# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI app factory + CLI for the Open Responses adapter.

Routes:
  POST /v1/responses        -> Open Responses (streaming + non-streaming). The special Open WebUI
                               boxes (Thinking / Tool Executed / answer) live here.
  POST /v1/chat/completions -> OpenAI-compatible chat (Open WebUI's default mode).
  GET  /v1/models           -> minimal model list so clients discover the model.
  GET  /health              -> liveness + the configured nat serve URL.
"""

from __future__ import annotations

import argparse
import logging
import time
import uuid

import uvicorn
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse

from natbridge import openai_translator
from natbridge import responses_translator
from natbridge.config import AdapterConfig
from natbridge.frames import NatFrameParser
from natbridge.open_responses import extract_input_text

logger = logging.getLogger("natbridge.app")

_SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive"}


def _bad_request(message: str, param: str | None = None) -> JSONResponse:
    return JSONResponse(status_code=400, content={
        "error": {"message": message, "type": "invalid_request", "param": param, "code": "invalid_request"}})


def create_app(config: AdapterConfig) -> FastAPI:
    """Build the FastAPI app bound to a given `AdapterConfig`."""
    app = FastAPI(title="Open Responses adapter → nat serve /chat/stream")
    parser = NatFrameParser(config)

    # -- Open Responses ----------------------------------------------------

    @app.post("/v1/responses")
    async def create_response(request: Request):
        try:
            body = await request.json()
        except Exception:
            return _bad_request("Invalid JSON body")

        model = body.get("model") or config.model_id
        user_text = extract_input_text(body.get("input"))
        stream = bool(body.get("stream", False))
        tools = body.get("tools") or []
        instructions = body.get("instructions")
        previous_response_id = body.get("previous_response_id")
        user_id = request.headers.get("x-user-id")
        response_id = previous_response_id or f"resp_{uuid.uuid4().hex}"

        if not user_text:
            return _bad_request("No user text found in `input`.", param="input")

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
        user_text = openai_translator.last_user_text(body.get("messages", []))
        stream = bool(body.get("stream", False))
        user_id = request.headers.get("x-user-id")
        conversation_id = body.get("conversation_id") or request.headers.get("conversation-id")

        if not user_text:
            return _bad_request("No user message found.")

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
        return {"status": "healthy", "nat_chat_stream_url": config.nat_chat_stream_url}

    return app


def build_config_from_args(argv: list[str] | None = None) -> tuple[AdapterConfig, str, int]:
    """Parse CLI args into an `AdapterConfig` plus (host, port)."""
    ap = argparse.ArgumentParser(description="Open Responses adapter in front of nat serve /chat/stream")
    ap.add_argument("--nat-url", default="http://localhost:8000/chat/stream",
                    help="URL of the running nat serve /chat/stream endpoint")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model-id", default="nat", help="Model id advertised to clients")
    ap.add_argument("--step-payload-max", type=int, default=1000,
                    help="Max chars kept per step payload (bounds SSE line size). 0 = no limit.")
    ap.add_argument("--max-steps", type=int, default=50, help="Max thinking blocks surfaced.")
    ap.add_argument("--reasoning-total-max", type=int, default=8000,
                    help="Max total chars in the accumulated Thinking box.")
    args = ap.parse_args(argv)
    config = AdapterConfig(
        nat_chat_stream_url=args.nat_url, model_id=args.model_id,
        step_payload_max=args.step_payload_max, max_steps=args.max_steps,
        reasoning_total_max=args.reasoning_total_max)
    return config, args.host, args.port


def main(argv: list[str] | None = None) -> None:
    config, host, port = build_config_from_args(argv)
    logging.basicConfig(level=logging.INFO)
    logger.info("Open Responses adapter -> %s (step_payload_max=%s, max_steps=%s)",
                config.nat_chat_stream_url, config.step_payload_max, config.max_steps)
    uvicorn.run(create_app(config), host=host, port=port)


if __name__ == "__main__":
    main()
