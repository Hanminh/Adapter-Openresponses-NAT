# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`cardbridge` — Open Responses adapter: box write_todos (không thinking) + item `script_output`.

Giống todobridge (vẫn nhận file upload Open WebUI, chèn đường dẫn tuyệt đối), nhưng:
  * KHÔNG bọc hộp Thinking quanh danh sách todo.
  * Output của script `package_details` -> item type RIÊNG (`script_output`) cho client.

Yêu cầu: workflow bọc graph bằng
`agenticskills.deep_agents.stream_passthrough.as_nat_graph_passthrough` (đẩy step write_todos +
package_details) và deep agent bật todos + có tool `send_package_details`
(`build_native_optimize_deep_agent_passthrough`).

Routes (như todobridge):
  POST /v1/responses        -> Open Responses (stream + non-stream)
  POST /v1/chat/completions -> OpenAI-compatible (todos thành reasoning; script bỏ qua)
  POST /api/v1/files/       -> nhận multipart upload
  GET  /v1/models, /health
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import uuid

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from natbridge import openai_translator
from natbridge.open_responses import extract_input_text

from filebridge.files import (
    UploadRegistry,
    augment_user_text,
    referenced_file_ids,
    resolve_owui_files,
    save_upload_stream,
    save_uploaded_files,
)

from cardbridge import translator
from cardbridge.config import CardAdapterConfig

logger = logging.getLogger("cardbridge.app")

_SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive"}


def _bad_request(message: str, param: str | None = None) -> JSONResponse:
    return JSONResponse(status_code=400, content={
        "error": {"message": message, "type": "invalid_request", "param": param, "code": "invalid_request"}})


def _conversation_id(request: Request, body: dict, *, fallback: str) -> str:
    """conversation-id ỔN ĐỊNH để forward sang nat serve (không dùng response_id đổi mỗi lượt)."""
    meta = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    for cand in (
        request.headers.get("conversation-id"),
        request.headers.get("x-conversation-id"),
        body.get("conversation_id"),
        body.get("conversation"),
        body.get("chat_id"),
        meta.get("conversation_id"),
        meta.get("chat_id"),
    ):
        if cand:
            return str(cand)
    return fallback


def create_app(config: CardAdapterConfig) -> FastAPI:
    app = FastAPI(title="cardbridge — write_todos + script_output → nat serve")
    registry = UploadRegistry()

    def _handle_files(body: dict, user_text: str, user_id: str | None) -> str:
        paths: list[str] = []
        try:
            user_text, owui_paths = resolve_owui_files(
                user_text, config.owui_upload_dir, config.upload_dir, config.max_file_bytes,
                strip_rag=config.strip_rag_context)
            paths += owui_paths
            paths += save_uploaded_files(body, config.upload_dir, config.max_file_bytes)
            paths += registry.resolve_ids(referenced_file_ids(body))
            if not paths:
                paths += registry.take_pending(user_id)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to resolve uploaded files; forwarding text only")
            return user_text
        paths = list(dict.fromkeys(paths))
        if paths:
            logger.info("Attached %d file(s): %s", len(paths), paths)
        return augment_user_text(user_text, paths, config.attachment_label)

    @app.post("/api/v1/files/")
    async def upload_file(file: UploadFile = File(...), metadata: str | None = Form(None)):
        try:
            file_id, abs_path, size = await save_upload_stream(
                file, config.upload_dir, max_bytes=config.max_file_bytes)
        except ValueError as exc:
            return _bad_request(str(exc))
        registry.add(file_id, abs_path, None)
        original = os.path.basename(file.filename or "unnamed")
        return {"id": file_id, "filename": original, "path": abs_path,
                "meta": {"name": original, "content_type": file.content_type, "size": size}}

    @app.post("/v1/responses")
    async def create_response(request: Request):
        try:
            body = await request.json()
        except Exception:
            return _bad_request("Invalid JSON body")

        model = body.get("model") or config.model_id
        user_id = request.headers.get("x-user-id")
        user_text = _handle_files(body, extract_input_text(body.get("input")), user_id)
        stream = bool(body.get("stream", False))
        tools = body.get("tools") or []
        instructions = body.get("instructions")
        previous_response_id = body.get("previous_response_id")
        response_id = previous_response_id or f"resp_{uuid.uuid4().hex}"
        conversation_id = _conversation_id(request, body, fallback=response_id)

        if not user_text:
            return _bad_request("No user text or file found in `input`.", param="input")

        if not stream:
            try:
                obj = await translator.build_non_streaming_response(
                    config, user_text, response_id=response_id, model=model, user_id=user_id,
                    conversation_id=conversation_id, tools=tools, instructions=instructions,
                    previous_response_id=previous_response_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Non-streaming run failed")
                return JSONResponse(status_code=500, content={
                    "error": {"message": str(exc), "type": "model_error", "param": None, "code": "model_error"}})
            return JSONResponse(content=obj)

        return StreamingResponse(
            translator.stream_open_responses(
                config, user_text, response_id=response_id, model=model, user_id=user_id,
                conversation_id=conversation_id, tools=tools, instructions=instructions,
                previous_response_id=previous_response_id),
            media_type="text/event-stream", headers=_SSE_HEADERS)

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": [
            {"id": config.model_id, "object": "model", "created": int(time.time()), "owned_by": "nat"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except Exception:
            return _bad_request("Invalid JSON")

        model = body.get("model") or config.model_id
        user_id = request.headers.get("x-user-id")
        user_text = _handle_files(body, openai_translator.last_user_text(body.get("messages", [])), user_id)
        stream = bool(body.get("stream", False))
        conversation_id = body.get("conversation_id") or request.headers.get("conversation-id")

        if not user_text:
            return _bad_request("No user message or file found.")

        if stream:
            return StreamingResponse(
                translator.stream_openai(config, user_text, model=model, user_id=user_id,
                                         conversation_id=conversation_id),
                media_type="text/event-stream", headers=_SSE_HEADERS)

        text = await translator.build_answer(config, user_text, user_id=user_id,
                                             conversation_id=conversation_id)
        return JSONResponse(content={
            "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

    @app.get("/health")
    async def health():
        return {"status": "healthy", "bridge": "cardbridge",
                "nat_chat_stream_url": config.nat_chat_stream_url,
                "script_output_type": config.script_output_type,
                "rule": f"step write_todos -> box | step {config.script_step_marker} -> item "
                        f"{config.script_output_type} | token -> answer"}

    return app


def build_config_from_args(argv: list[str] | None = None) -> tuple[CardAdapterConfig, str, int]:
    ap = argparse.ArgumentParser(
        description="cardbridge — Open Responses adapter: box write_todos + item script_output, trước nat serve")
    ap.add_argument("--nat-url", default="http://localhost:8000/chat/stream")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8005)
    ap.add_argument("--model-id", default="nat")
    ap.add_argument("--upload-dir", default="/tmp/nat_uploads")
    ap.add_argument("--max-file-mb", type=float, default=25.0)
    ap.add_argument("--owui-upload-dir", default=None)
    ap.add_argument("--keep-rag-context", action="store_true")
    ap.add_argument("--hide-answer", action="store_true", help="Chỉ hiện box/item, KHÔNG hiện đáp án")
    ap.add_argument("--script-output-type", default="carousel",
                    help="`type` của item output script gửi client (mặc định carousel)")
    ap.add_argument("--script-output-name", default="package_details",
                    help="`name` gắn kèm item output script (mặc định package_details)")
    args = ap.parse_args(argv)

    config = CardAdapterConfig(
        nat_chat_stream_url=args.nat_url, model_id=args.model_id,
        upload_dir=args.upload_dir, max_file_bytes=int(args.max_file_mb * 1024 * 1024),
        owui_upload_dir=args.owui_upload_dir, strip_rag_context=not args.keep_rag_context,
        show_answer=not args.hide_answer,
        script_output_type=args.script_output_type, script_output_name=args.script_output_name)
    return config, args.host, args.port


def main(argv: list[str] | None = None) -> None:
    config, host, port = build_config_from_args(argv)
    logging.basicConfig(level=logging.INFO)
    logger.info("cardbridge -> %s (box write_todos + item script_output '%s')",
                config.nat_chat_stream_url, config.script_output_type)
    uvicorn.run(create_app(config), host=host, port=port)


if __name__ == "__main__":
    main()
