# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate NAT `/chat/stream` into OpenAI `chat.completion.chunk` SSE (the
`/v1/chat/completions` endpoint — Open WebUI's default mode).

Mapping:
  KIỂU 1 thinking + KIỂU 2 tool -> `delta.reasoning_content` (Open WebUI "Thinking" box).
  KIỂU 3 final answer           -> `delta.content` (streamed as deltas).

Tools deliberately go into `reasoning_content`, NOT `delta.tool_calls`: in OpenAI chat semantics a
`tool_calls` delta means "client, run this tool", so Open WebUI would try to RE-EXECUTE a tool NAT
already ran. For a SEPARATE "Tool Executed" box, use `/v1/responses` instead.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from natbridge.config import AdapterConfig
from natbridge.extractors import FinalAnswerStreamer
from natbridge.extractors import ThinkingExtractor
from natbridge.frames import NatFrameParser
from natbridge.nat_client import call_chat_stream
from natbridge.util import ToolTracker
from natbridge.util import format_block

logger = logging.getLogger("natbridge.openai")


def openai_chunk(chat_id: str, model: str, *, content: str | None = None,
                 reasoning_content: str | None = None, role: str | None = None,
                 finish_reason: str | None = None) -> str:
    delta: dict = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    chunk = {
        "id": chat_id, "object": "chat.completion.chunk", "created": int(time.time()),
        "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def last_user_text(messages: list) -> str:
    """OpenAI `messages[]` -> the last user message's text."""
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(p.get("text", "") for p in content
                                 if isinstance(p, dict) and p.get("type") in ("text", "input_text"))
    return ""


async def stream_openai(config: AdapterConfig, parser: NatFrameParser, user_text: str, *,
                        model: str, user_id: str | None,
                        conversation_id: str | None) -> AsyncIterator[str]:
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    thinking = ThinkingExtractor()      # KIỂU 1
    answer = FinalAnswerStreamer()      # KIỂU 3
    tools_t = ToolTracker()             # KIỂU 2
    n_blocks = 0
    answer_started = False
    got_answer = False
    final_fallback = ""
    data_fallback = ""
    yield openai_chunk(chat_id, model, role="assistant")

    def block_chunk(block: str | None) -> str | None:
        nonlocal n_blocks
        if not block or n_blocks >= config.max_steps:
            return None
        n_blocks += 1
        return openai_chunk(chat_id, model, reasoning_content=format_block(block, n_blocks, config.step_payload_max))

    def tool_chunk(name: str, args: str, result: str | None) -> str | None:
        _call_id, is_new = tools_t.on_call(name, args)
        if is_new:                                       # lần đầu -> "tool đang chạy"
            block = f"🔧 **Tool:** `{name}`\n- Input: `{args}`"
            if result and tools_t.on_result(name, args):
                block += f"\n- Result: {result}"
            return block_chunk(block)
        if result and tools_t.on_result(name, args):     # kết quả tới sau -> bổ sung dòng kết quả
            return block_chunk(f"🔧 **Result** `{name}`: {result}")
        return None

    def maybe_attach(step: dict) -> str | None:
        if tools_t.pending is None:
            return None
        obs = parser.last_observation(step, user_text)
        if not obs:
            return None
        return tool_chunk(tools_t.pending[0], tools_t.pending[1], obs)

    try:
        async for kind, obj in call_chat_stream(config, user_text, user_id=user_id,
                                                conversation_id=conversation_id):
            if kind == "step":
                cls = parser.classify(obj)
                if cls == "skip":
                    continue
                if cls == "tool":                                    # KIỂU 2
                    name, args, result = parser.parse_tool(obj)
                    chunk = tool_chunk(name, args, result)
                    if chunk:
                        yield chunk
                    continue
                if cls == "final":                                   # KIỂU 3
                    chunk = maybe_attach(obj)
                    if chunk:
                        yield chunk
                    fa = parser.final_answer_text(obj)
                    if fa:
                        final_fallback = fa
                    if not answer_started:
                        flush = block_chunk(thinking.flush())        # dọn nốt hộp thinking
                        if flush:
                            yield flush
                        answer_started = True
                    delta = answer.feed(fa)
                    if delta:
                        got_answer = True
                        yield openai_chunk(chat_id, model, content=delta)
                    continue
                # KIỂU 1
                chunk = maybe_attach(obj)
                if chunk:
                    yield chunk
                if not answer_started:
                    block = block_chunk(thinking.feed(parser.thinking_text(obj)))
                    if block:
                        yield block
            elif kind == "token":
                for choice in obj.get("choices", []):
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        data_fallback += content

        if not answer_started:
            flush = block_chunk(thinking.flush())
            if flush:
                yield flush
            answer_started = True
        if not got_answer:
            text = final_fallback or data_fallback
            if text:
                yield openai_chunk(chat_id, model, content=text)
        yield openai_chunk(chat_id, model, finish_reason="stop")
        yield "data: [DONE]\n\n"

    except Exception as exc:  # noqa: BLE001
        logger.exception("OpenAI stream failed")
        yield f"data: {json.dumps({'error': {'message': str(exc), 'type': 'server_error', 'code': 'server_error'}})}\n\n"
        yield "data: [DONE]\n\n"


async def build_non_streaming(config: AdapterConfig, parser: NatFrameParser, user_text: str, *,
                              user_id: str | None, conversation_id: str | None) -> str:
    """Drain the stream and return the answer text (for a non-streaming chat.completion)."""
    text = ""
    final_fallback = ""
    async for kind, obj in call_chat_stream(config, user_text, user_id=user_id,
                                            conversation_id=conversation_id):
        if kind == "token":
            for choice in obj.get("choices", []):
                text += (choice.get("delta") or {}).get("content") or ""
        else:
            fa = parser.final_answer_text(obj)
            if fa:
                final_fallback = fa
    return text or final_fallback
