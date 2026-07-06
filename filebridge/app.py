# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""File-aware Open Responses adapter in front of `nat serve`.

Same Open Responses API standard as `natbridge` (it reuses natbridge's translators verbatim), but
adds a front step: any file uploaded by Open WebUI is saved to local disk and its ABSOLUTE PATH is
appended to the user message before the turn is forwarded to `nat serve`. The agent then sees a
plain-text request that names the file paths on disk.

Routes (identical surface to natbridge):
  POST /v1/responses        -> Open Responses (streaming + non-streaming) + file handling
  POST /v1/chat/completions -> OpenAI-compatible chat + file handling
  GET  /v1/models           -> model list
  GET  /health              -> liveness + nat serve URL + upload dir
"""

from __future__ import annotations

import argparse
import logging
import time
import uuid

import os
import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse

from natbridge import openai_translator
from natbridge import responses_translator
from natbridge.frames import NatFrameParser
from natbridge.open_responses import extract_input_text

from filebridge.config import FileAdapterConfig
from filebridge.files import UploadRegistry
from filebridge.files import augment_user_text
from filebridge.files import referenced_file_ids
from filebridge.files import resolve_owui_files
from filebridge.files import save_upload_stream
from filebridge.files import save_uploaded_files

logger = logging.getLogger("filebridge.app")

_SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive"}


def _bad_request(message: str, param: str | None = None) -> JSONResponse:
    return JSONResponse(status_code=400, content={
        "error": {"message": message, "type": "invalid_request", "param": param, "code": "invalid_request"}})


def create_app(config: FileAdapterConfig) -> FastAPI:
    app = FastAPI(title="File-aware Open Responses adapter → nat serve")
    parser = NatFrameParser(config)
    registry = UploadRegistry()   # files received via POST /api/v1/files/

    def _handle_files(body: dict, user_text: str, user_id: str | None) -> str:
        """Resolve every file for this turn and append their absolute paths to the user text.

        Sources, in order:
          1. inline base64 / data-URI files in the chat payload (images, some clients),
          2. files uploaded to POST /api/v1/files/ and referenced by id (metadata.files),
          3. fallback: files this user uploaded but that weren't matched by id (pending queue).
        """
        paths: list[str] = []
        try:
            # 0) Open WebUI same-machine: read the ORIGINAL file from OWUI's uploads by id
            #    (the `<file .../files/{id}/content .../>` tag in the message text).
            user_text, owui_paths = resolve_owui_files(
                user_text, config.owui_upload_dir, config.upload_dir, config.max_file_bytes,
                strip_rag=config.strip_rag_context)
            paths += owui_paths
            # 1) inline base64 / data-URI files in the payload (images, some clients)
            paths += save_uploaded_files(body, config.upload_dir, config.max_file_bytes)
            # 2) files uploaded to POST /api/v1/files/ and referenced by id
            paths += registry.resolve_ids(referenced_file_ids(body))
            if not paths:  # nothing found -> consume this user's recent multipart uploads
                paths += registry.take_pending(user_id)
        except Exception:  # noqa: BLE001 - file handling must never break the turn
            logger.exception("Failed to resolve uploaded files; forwarding text only")
            return user_text
        # de-duplicate, keep order
        paths = list(dict.fromkeys(paths))
        if paths:
            logger.info("Attached %d file(s) to the request: %s", len(paths), paths)
        return augment_user_text(user_text, paths, config.attachment_label)

    # -- File upload (Open WebUI multipart) --------------------------------
    @app.post("/api/v1/files/")
    async def upload_file(file: UploadFile = File(...), metadata: str | None = Form(None)):
        """Receive a browser file upload (multipart/form-data) and save it locally.

        Mirrors Open WebUI's own POST /api/v1/files/ contract so uploads routed here are stored on
        disk and made available to the next chat turn (by id, or via the per-user pending queue).
        """
        try:
            file_id, abs_path, size = await save_upload_stream(
                file, config.upload_dir, max_bytes=config.max_file_bytes)
        except ValueError as exc:
            return _bad_request(str(exc))
        registry.add(file_id, abs_path, None)  # multipart form has no x-user-id header for OWUI
        original = os.path.basename(file.filename or "unnamed")
        return {"id": file_id, "filename": original, "path": abs_path,
                "meta": {"name": original, "content_type": file.content_type, "size": size}}

    # -- Open Responses (the standard is preserved via natbridge translators) --
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

        if not user_text:
            return _bad_request("No user text or file found in `input`.", param="input")

        if not stream:
            try:
                obj = await responses_translator.build_non_streaming(
                    config, parser, user_text, response_id=response_id, model=model, user_id=user_id,
                    tools=tools, instructions=instructions, previous_response_id=previous_response_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Non-streaming run failed")
                return JSONResponse(status_code=500, content={
                    "error": {"message": str(exc), "type": "model_error", "param": None, "code": "model_error"}})
            return JSONResponse(content=obj)

        return StreamingResponse(
            responses_translator.stream_open_responses(
                config, parser, user_text, response_id=response_id, model=model, user_id=user_id,
                tools=tools, instructions=instructions, previous_response_id=previous_response_id),
            media_type="text/event-stream", headers=_SSE_HEADERS)

    # -- OpenAI-compatible -------------------------------------------------
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
                openai_translator.stream_openai(config, parser, user_text, model=model,
                                                user_id=user_id, conversation_id=conversation_id),
                media_type="text/event-stream", headers=_SSE_HEADERS)

        text = await openai_translator.build_non_streaming(
            config, parser, user_text, user_id=user_id, conversation_id=conversation_id)
        return JSONResponse(content={
            "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})

    @app.get("/health")
    async def health():
        return {"status": "healthy", "nat_chat_stream_url": config.nat_chat_stream_url,
                "upload_dir": config.upload_dir}

    return app


def build_config_from_args(argv: list[str] | None = None) -> tuple[FileAdapterConfig, str, int]:
    ap = argparse.ArgumentParser(description="File-aware Open Responses adapter in front of nat serve")
    ap.add_argument("--nat-url", default="http://localhost:8000/chat/stream")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model-id", default="nat")
    ap.add_argument("--upload-dir", default="/tmp/nat_uploads", help="Where files are copied/saved locally")
    ap.add_argument("--max-file-mb", type=float, default=25.0, help="Per-file size cap in MB (0 = no limit)")
    ap.add_argument("--owui-upload-dir", default=None,
                    help="Open WebUI uploads dir ({DATA_DIR}/uploads) to read original files by id "
                         "(same machine). Needed to turn attachments into local file paths.")
    ap.add_argument("--keep-rag-context", action="store_true",
                    help="Keep Open WebUI's RAG <context> block (default: strip it, forwarding only "
                         "the real user question + file path).")
    args = ap.parse_args(argv)
    config = FileAdapterConfig(
        nat_chat_stream_url=args.nat_url, model_id=args.model_id,
        upload_dir=args.upload_dir, max_file_bytes=int(args.max_file_mb * 1024 * 1024),
        owui_upload_dir=args.owui_upload_dir, strip_rag_context=not args.keep_rag_context)
    return config, args.host, args.port


def main(argv: list[str] | None = None) -> None:
    config, host, port = build_config_from_args(argv)
    logging.basicConfig(level=logging.INFO)
    logger.info("File-aware adapter -> %s (upload_dir=%s)", config.nat_chat_stream_url, config.upload_dir)
    uvicorn.run(create_app(config), host=host, port=port)


if __name__ == "__main__":
    main()
