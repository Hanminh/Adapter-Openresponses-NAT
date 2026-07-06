# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate NAT `/chat/stream` into Open Responses events (the `/v1/responses` endpoint).

This is the PRIMARY path: it produces the special Open WebUI boxes. `output[]` ends up as
`[reasoning, function_call, function_call_output, message]`, rendered as:
    Thinking box  ->  Tool Executed box (with result)  ->  clear answer
The reasoning item is always closed before a tool box (or the message) opens, so items never
overlap and render in the right order.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator

import httpx

from natbridge.config import AdapterConfig
from natbridge.extractors import FinalAnswerStreamer
from natbridge.extractors import ThinkingExtractor
from natbridge.frames import NatFrameParser
from natbridge.nat_client import call_chat_stream
from natbridge.open_responses import OpenResponsesEmitter
from natbridge.util import ToolTracker
from natbridge.util import format_block

logger = logging.getLogger("natbridge.responses")


async def stream_open_responses(config: AdapterConfig, parser: NatFrameParser, user_text: str, *,
                                response_id: str, model: str, user_id: str | None,
                                tools: list | None, instructions: str | None,
                                previous_response_id: str | None) -> AsyncIterator[str]:
    em = OpenResponsesEmitter(response_id, model, tools=tools, instructions=instructions,
                              previous_response_id=previous_response_id,
                              step_payload_max=config.step_payload_max, max_steps=config.max_steps,
                              reasoning_total_max=config.reasoning_total_max)
    msg_id = f"msg_{uuid.uuid4().hex}"
    thinking = ThinkingExtractor()      # KIỂU 1
    answer = FinalAnswerStreamer()      # KIỂU 3
    tools_t = ToolTracker()             # KIỂU 2
    reasoning_open = msg_open = False
    n_blocks = 0
    got_answer = False
    final_fallback = ""     # full "Final Answer:" text, used if nothing streamed
    data_fallback = ""      # `data:` content, used only if KIỂU 3 produced nothing

    def emit_block(block: str | None) -> list[str]:
        nonlocal reasoning_open, n_blocks
        if not block or n_blocks >= config.max_steps:
            return []
        n_blocks += 1
        frames: list[str] = []
        if not reasoning_open:
            frames.append(em.open_reasoning(f"rs_{uuid.uuid4().hex}"))
            reasoning_open = True
        f = em.reasoning_delta(format_block(block, n_blocks, config.step_payload_max))
        if f:
            frames.append(f)
        return frames

    def close_reasoning() -> list[str]:
        nonlocal reasoning_open
        frames = emit_block(thinking.flush())
        if reasoning_open:
            f = em.close_reasoning()
            if f:
                frames.append(f)
            reasoning_open = False
        return frames

    def open_message() -> list[str]:
        nonlocal msg_open
        frames = close_reasoning()
        frames += em.open_message(msg_id)
        msg_open = True
        return frames

    def emit_tool(name: str, args: str, result: str | None) -> list[str]:
        frames: list[str] = []
        call_id, is_new = tools_t.on_call(name, args)
        if is_new:                                   # lần đầu -> box "tool đang chạy"
            frames += em.tool_call(name, args, call_id)
        if result and (cid := tools_t.on_result(name, args)):
            frames += em.tool_output(cid, result)    # gắn kết quả vào box tool
        return frames

    def attach_observation(step: dict) -> list[str]:
        if tools_t.pending is None:
            return []
        obs = parser.last_observation(step, user_text)
        if not obs:
            return []
        return emit_tool(tools_t.pending[0], tools_t.pending[1], obs)

    yield em.created()
    yield em.in_progress()

    try:
        async for kind, obj in call_chat_stream(config, user_text, user_id=user_id,
                                                conversation_id=response_id):
            if kind == "step":
                cls = parser.classify(obj)
                if cls == "skip":
                    continue
                if cls == "tool":                                    # KIỂU 2
                    name, args, result = parser.parse_tool(obj)
                    for f in close_reasoning():                      # đóng Thinking trước box Tool
                        yield f
                    for f in emit_tool(name, args, result):
                        yield f
                    continue
                if cls == "final":                                   # KIỂU 3
                    for f in attach_observation(obj):                # vớt kết quả tool (nếu có)
                        yield f
                    if not msg_open:
                        for f in open_message():
                            yield f
                    fa = parser.final_answer_text(obj)
                    if fa:
                        final_fallback = fa
                    delta = answer.feed(fa)
                    if delta:
                        got_answer = True
                        yield em.message_delta(delta)
                    continue
                # KIỂU 1
                for f in attach_observation(obj):
                    yield f
                if not msg_open:
                    for f in emit_block(thinking.feed(parser.thinking_text(obj))):
                        yield f
            elif kind == "token":
                for choice in obj.get("choices", []):
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        data_fallback += content

        if not msg_open:
            for f in open_message():
                yield f
        if not got_answer:
            text = final_fallback or data_fallback
            if text:
                yield em.message_delta(text)
        for f in em.close_message():
            yield f
        yield em.completed()
        yield em.done()

    except httpx.HTTPStatusError as exc:
        for f in em.failed(f"nat serve returned {exc.response.status_code}",
                           err_type="server_error", code="server_error"):
            yield f
        yield em.done()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Open Responses stream failed")
        for f in em.failed(str(exc)):
            yield f
        yield em.done()


async def build_non_streaming(config: AdapterConfig, parser: NatFrameParser, user_text: str, *,
                              response_id: str, model: str, user_id: str | None,
                              tools: list | None, instructions: str | None,
                              previous_response_id: str | None) -> dict:
    """Drain the stream and assemble one final response object (reasoning item + message)."""
    em = OpenResponsesEmitter(response_id, model, tools=tools, instructions=instructions,
                              previous_response_id=previous_response_id,
                              step_payload_max=config.step_payload_max, max_steps=config.max_steps,
                              reasoning_total_max=config.reasoning_total_max)
    thinking = ThinkingExtractor()
    blocks: list[str] = []
    text = ""
    final_fallback = ""
    async for kind, obj in call_chat_stream(config, user_text, user_id=user_id,
                                            conversation_id=response_id):
        if kind == "token":
            for choice in obj.get("choices", []):
                text += (choice.get("delta") or {}).get("content") or ""
        elif parser.classify(obj) != "skip":
            fa = parser.final_answer_text(obj)
            if fa:
                final_fallback = fa
            if len(blocks) < config.max_steps:
                block = thinking.feed(parser.thinking_text(obj))
                if block:
                    blocks.append(block)
    final_block = thinking.flush()
    if final_block and len(blocks) < config.max_steps:
        blocks.append(final_block)
    if not text:
        text = final_fallback
    if blocks:
        em.open_reasoning(f"rs_{uuid.uuid4().hex}")
        for i, block in enumerate(blocks, start=1):
            em.reasoning_delta(format_block(block, i, config.step_payload_max))
        em.close_reasoning()
    em.open_message(f"msg_{uuid.uuid4().hex}")
    em.message_delta(text)
    em.close_message()
    return em.final_object()
