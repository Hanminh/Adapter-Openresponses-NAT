# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`thinkbridge` — Open Responses adapter: "mọi thứ không phải tool và không phải đáp án = thinking".

Giống `filebridge` (giữ nguyên chuẩn Open Responses, vẫn nhận file upload từ Open WebUI và chèn
đường dẫn tuyệt đối vào câu hỏi), nhưng thay cách phân loại frame:

    natbridge/filebridge : đáp án = dò chuỗi "Final Answer:" trong `**Output:**`
                           -> CHỈ chạy với workflow `_type: react_agent`.
    thinkbridge          : đáp án = các token trên dòng `data:` của nat serve
                           -> chạy với CẢ `react_agent` LẪN `langgraph_wrapper`.

Mọi intermediate step không phải tool đều vào hộp Thinking. Model cứ "nghĩ" cho tới khi token đáp
án đầu tiên xuất hiện — đúng lúc đó thinking đóng lại và câu trả lời bắt đầu chảy.

Routes (y hệt filebridge):
  POST /v1/responses        -> Open Responses (stream + non-stream) + xử lý file
  POST /v1/chat/completions -> OpenAI-compatible + xử lý file
  POST /api/v1/files/       -> nhận multipart upload từ Open WebUI
  GET  /v1/models, /health
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import uuid

import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse

from natbridge import openai_translator
from natbridge.open_responses import extract_input_text

from filebridge.files import UploadRegistry
from filebridge.files import augment_user_text
from filebridge.files import referenced_file_ids
from filebridge.files import resolve_owui_files
from filebridge.files import save_upload_stream
from filebridge.files import save_uploaded_files

from thinkbridge import translator
from thinkbridge.config import ThinkAdapterConfig
from thinkbridge.frames import ThinkFrameParser

logger = logging.getLogger("thinkbridge.app")

_SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive"}


def _bad_request(message: str, param: str | None = None) -> JSONResponse:
    return JSONResponse(status_code=400, content={
        "error": {"message": message, "type": "invalid_request", "param": param, "code": "invalid_request"}})


def _conversation_id(request: Request, body: dict, *, fallback: str) -> str:
    """Lấy conversation-id ỔN ĐỊNH của hội thoại để forward sang nat serve.

    Ưu tiên header (Open WebUI thường gửi `conversation-id`), rồi các field trong body, cuối
    cùng mới dùng `fallback` (response_id). Dùng response_id làm conversation-id là SAI vì nó đổi
    mỗi lượt -> NAT coi mỗi lượt là hội thoại mới, mất memory/multi-turn.
    """
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


def create_app(config: ThinkAdapterConfig) -> FastAPI:
    app = FastAPI(title="thinkbridge — Open Responses adapter → nat serve")
    parser = ThinkFrameParser(config)
    registry = UploadRegistry()

    def _handle_files(body: dict, user_text: str, user_id: str | None) -> str:
        """Y hệt filebridge: gom file của lượt này rồi gắn đường dẫn tuyệt đối vào câu hỏi."""
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
        except Exception:  # noqa: BLE001 - xử lý file không bao giờ được làm hỏng lượt chat
            logger.exception("Failed to resolve uploaded files; forwarding text only")
            return user_text
        paths = list(dict.fromkeys(paths))
        if paths:
            logger.info("Attached %d file(s): %s", len(paths), paths)
        return augment_user_text(user_text, paths, config.attachment_label)

    # -- upload (Open WebUI multipart) --------------------------------------
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

    # -- Open Responses -----------------------------------------------------
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
        # conversation-id ổn định để forward sang nat serve (KHÔNG dùng response_id — đổi mỗi lượt).
        conversation_id = _conversation_id(request, body, fallback=response_id)

        if not user_text:
            return _bad_request("No user text or file found in `input`.", param="input")

        if not stream:
            try:
                obj = await translator.build_non_streaming_response(
                    config, parser, user_text, response_id=response_id, model=model, user_id=user_id,
                    conversation_id=conversation_id, tools=tools, instructions=instructions,
                    previous_response_id=previous_response_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Non-streaming run failed")
                return JSONResponse(status_code=500, content={
                    "error": {"message": str(exc), "type": "model_error", "param": None, "code": "model_error"}})
            return JSONResponse(content=obj)

        return StreamingResponse(
            translator.stream_open_responses(
                config, parser, user_text, response_id=response_id, model=model, user_id=user_id,
                conversation_id=conversation_id, tools=tools, instructions=instructions,
                previous_response_id=previous_response_id),
            media_type="text/event-stream", headers=_SSE_HEADERS)

    # -- OpenAI-compatible --------------------------------------------------
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
        # `last_user_text` của natbridge là thuần đọc messages[] -> dùng lại được.
        user_text = _handle_files(body, openai_translator.last_user_text(body.get("messages", [])), user_id)
        stream = bool(body.get("stream", False))
        conversation_id = body.get("conversation_id") or request.headers.get("conversation-id")

        if not user_text:
            return _bad_request("No user message or file found.")

        if stream:
            return StreamingResponse(
                translator.stream_openai(config, parser, user_text, model=model, user_id=user_id,
                                         conversation_id=conversation_id),
                media_type="text/event-stream", headers=_SSE_HEADERS)

        text = await translator.build_answer(config, parser, user_text, user_id=user_id,
                                             conversation_id=conversation_id)
        return JSONResponse(content={
            "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

    @app.get("/health")
    async def health():
        return {"status": "healthy", "bridge": "thinkbridge",
                "nat_chat_stream_url": config.nat_chat_stream_url,
                "upload_dir": config.upload_dir,
                "rule": "tool -> tool box | data: tokens -> final answer | everything else -> thinking"}

    return app


def build_config_from_args(argv: list[str] | None = None) -> tuple[ThinkAdapterConfig, str, int]:
    ap = argparse.ArgumentParser(
        description="thinkbridge — Open Responses adapter (everything-else-is-thinking) in front of nat serve")
    ap.add_argument("--nat-url", default="http://localhost:8000/chat/stream")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--model-id", default="nat")
    ap.add_argument("--upload-dir", default="/tmp/nat_uploads")
    ap.add_argument("--max-file-mb", type=float, default=25.0)
    ap.add_argument("--owui-upload-dir", default=None,
                    help="Thư mục uploads của Open WebUI ({DATA_DIR}/uploads) để đọc file gốc theo id")
    ap.add_argument("--keep-rag-context", action="store_true",
                    help="Giữ khối <context> RAG của Open WebUI (mặc định: cắt bỏ)")
    ap.add_argument("--keep-final-in-thinking", action="store_true",
                    help="Giữ nguyên phần sau 'Final Answer:' trong hộp Thinking (mặc định: cắt, "
                         "vì đáp án đã được stream riêng -> tránh hiện hai lần)")
    ap.add_argument("--keep-thinking-after-answer", action="store_true",
                    help="Vẫn phát khối thinking dở dang khi đáp án bắt đầu (mặc định: vứt — với "
                         "langgraph_wrapper khối đó CHÍNH LÀ đáp án)")
    ap.add_argument("--hide-tools", action="store_true", help="Không hiện box tool")
    args = ap.parse_args(argv)

    config = ThinkAdapterConfig(
        nat_chat_stream_url=args.nat_url, model_id=args.model_id,
        upload_dir=args.upload_dir, max_file_bytes=int(args.max_file_mb * 1024 * 1024),
        owui_upload_dir=args.owui_upload_dir, strip_rag_context=not args.keep_rag_context,
        strip_final_answer_from_thinking=not args.keep_final_in_thinking,
        stop_thinking_on_first_token=not args.keep_thinking_after_answer,
        show_tools=not args.hide_tools)
    return config, args.host, args.port


def main(argv: list[str] | None = None) -> None:
    config, host, port = build_config_from_args(argv)
    logging.basicConfig(level=logging.INFO)
    logger.info("thinkbridge -> %s (upload_dir=%s)", config.nat_chat_stream_url, config.upload_dir)
    logger.info("rule: tool -> tool box | `data:` tokens -> final answer | everything else -> thinking")
    uvicorn.run(create_app(config), host=host, port=port)


if __name__ == "__main__":
    main()
