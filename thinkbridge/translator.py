# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dịch `/chat/stream` của NAT sang Open Responses + OpenAI chat, theo quy tắc:

    TOOL     -> box tool  (hoặc dòng 🔧 trong Thinking, tuỳ endpoint)
    `data:`  -> ĐÁP ÁN CUỐI, stream token-by-token vào `content`
    còn lại  -> THINKING

Hai chốt quan trọng để đáp án KHÔNG bị hiện hai lần
--------------------------------------------------
1. `strip_final_answer_from_thinking`: với `react_agent`, khối `**Output:**` cuối chứa cả
   "Thought: ... Final Answer: ...". Ta cắt tại marker, giữ Thought, bỏ đáp án (đáp án đã đến
   qua `data:`).

2. `stop_thinking_on_first_token`: với `langgraph_wrapper` KHÔNG có marker nào cả — khối
   `**Output:**` của lượt LLM cuối CHÍNH LÀ đáp án. Nhưng nó đang là khối thinking "dở dang"
   (ThinkingExtractor chỉ phát một khối khi khối MỚI bắt đầu). Vậy nên khi token `data:` đầu tiên
   xuất hiện, ta **vứt khối dở dang** và ngừng phát thinking. Đúng tinh thần: model cứ "nghĩ" cho
   tới khi có kết quả cuối cùng.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from natbridge.extractors import ThinkingExtractor
from natbridge.nat_client import call_chat_stream
from natbridge.open_responses import OpenResponsesEmitter
from natbridge.util import ToolTracker
from natbridge.util import format_block

from thinkbridge.config import ThinkAdapterConfig
from thinkbridge.frames import ThinkFrameParser
from thinkbridge.frames import ThinkStripper

logger = logging.getLogger("thinkbridge.translator")


def _token_text(obj: dict) -> str:
    """Nội dung của một ChatResponseChunk trên dòng `data:`."""
    out = ""
    for choice in obj.get("choices", []) or []:
        out += (choice.get("delta") or {}).get("content") or ""
    return out


# --------------------------------------------------------------------------- #
# /v1/chat/completions
# --------------------------------------------------------------------------- #
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
    chunk = {"id": chat_id, "object": "chat.completion.chunk", "created": int(time.time()),
             "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


async def stream_openai(config: ThinkAdapterConfig, parser: ThinkFrameParser, user_text: str, *,
                        model: str, user_id: str | None,
                        conversation_id: str | None) -> AsyncIterator[str]:
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    thinking = ThinkingExtractor()
    tools_t = ToolTracker()
    strip = ThinkStripper() if config.strip_think_from_answer else None
    n_blocks = 0
    answer_started = False       # đã có token `data:` đầu tiên chưa
    got_answer = False

    yield openai_chunk(chat_id, model, role="assistant")

    def block_chunk(block: str | None) -> str | None:
        nonlocal n_blocks
        if not block or n_blocks >= config.max_steps:
            return None
        n_blocks += 1
        return openai_chunk(chat_id, model,
                            reasoning_content=format_block(block, n_blocks, config.step_payload_max))

    def tool_chunk(name: str, args: str, result: str | None) -> str | None:
        if not config.show_tools:
            return None
        _call_id, is_new = tools_t.on_call(name, args)
        if is_new:
            block = f"🔧 **Tool:** `{name}`\n- Input: `{args}`"
            if result and tools_t.on_result(name, args):
                block += f"\n- Result: {result}"
            return block_chunk(block)
        if result and tools_t.on_result(name, args):
            return block_chunk(f"🔧 **Result** `{name}`: {result}")
        return None

    try:
        async for kind, obj in call_chat_stream(config, user_text, user_id=user_id,
                                                conversation_id=conversation_id):
            if kind == "token":
                text = _token_text(obj)
                if not text:
                    continue
                if strip is not None:                       # bỏ <think>...</think> khỏi đáp án
                    text = strip.feed(text)
                    if not text:
                        continue
                if not answer_started:
                    # Kết quả cuối bắt đầu -> ngừng thinking. KHÔNG flush khối dở dang: với
                    # langgraph_wrapper khối đó chính là đáp án này.
                    answer_started = True
                    if not config.stop_thinking_on_first_token:
                        flush = block_chunk(thinking.flush())
                        if flush:
                            yield flush
                got_answer = True
                yield openai_chunk(chat_id, model, content=text)
                continue

            # kind == "step"
            cls = parser.classify(obj)
            if cls == "skip":
                continue
            if cls == "tool":
                chunk = tool_chunk(*parser.parse_tool(obj))
                if chunk:
                    yield chunk
                continue
            if answer_started:           # đã ra đáp án -> mọi step sau đều bỏ qua
                continue
            block = block_chunk(thinking.feed(parser.thinking_text(obj)))
            if block:
                yield block

        if strip is not None:                               # xả nốt đệm đáp án của bộ lọc think
            tail = strip.flush()
            if tail:
                got_answer = True
                yield openai_chunk(chat_id, model, content=tail)

        if not got_answer:
            # nat serve không stream token nào (workflow không hỗ trợ streaming) -> đưa nốt khối
            # thinking cuối ra làm câu trả lời, còn hơn trả về rỗng.
            tail = thinking.flush()
            if tail:
                yield openai_chunk(chat_id, model, content=tail)

        yield openai_chunk(chat_id, model, finish_reason="stop")
        yield "data: [DONE]\n\n"

    except Exception as exc:  # noqa: BLE001
        logger.exception("thinkbridge OpenAI stream failed")
        yield (f"data: {json.dumps({'error': {'message': str(exc), 'type': 'server_error', 'code': 'server_error'}})}"
               "\n\n")
        yield "data: [DONE]\n\n"


# --------------------------------------------------------------------------- #
# /v1/responses  (Open Responses — box Tool riêng, box Thinking riêng)
# --------------------------------------------------------------------------- #
async def stream_open_responses(config: ThinkAdapterConfig, parser: ThinkFrameParser, user_text: str, *,
                                response_id: str, model: str, user_id: str | None,
                                tools: list | None, instructions: str | None,
                                previous_response_id: str | None,
                                conversation_id: str | None = None) -> AsyncIterator[str]:
    em = OpenResponsesEmitter(response_id, model, tools=tools, instructions=instructions,
                              previous_response_id=previous_response_id,
                              step_payload_max=config.step_payload_max, max_steps=config.max_steps,
                              reasoning_total_max=config.reasoning_total_max)
    msg_id = f"msg_{uuid.uuid4().hex}"
    thinking = ThinkingExtractor()
    tools_t = ToolTracker()
    strip = ThinkStripper() if config.strip_think_from_answer else None
    reasoning_open = msg_open = False
    n_blocks = 0
    answer_started = False
    got_answer = False

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

    def close_reasoning(*, flush: bool) -> list[str]:
        """Đóng hộp Thinking. `flush=False` -> VỨT khối dở dang (nó chính là đáp án cuối)."""
        nonlocal reasoning_open
        frames = emit_block(thinking.flush()) if flush else []
        if not flush:
            thinking.flush()                    # xả state, không phát ra
        if reasoning_open:
            f = em.close_reasoning()
            if f:
                frames.append(f)
            reasoning_open = False
        return frames

    def emit_tool(name: str, args: str, result: str | None) -> list[str]:
        if not config.show_tools:
            return []
        frames: list[str] = []
        call_id, is_new = tools_t.on_call(name, args)
        if is_new:
            frames += em.tool_call(name, args, call_id)
        if result and (cid := tools_t.on_result(name, args)):
            frames += em.tool_output(cid, result)
        return frames

    yield em.created()
    yield em.in_progress()

    try:
        async for kind, obj in call_chat_stream(config, user_text, user_id=user_id,
                                                conversation_id=conversation_id or response_id):
            if kind == "token":
                text = _token_text(obj)
                if not text:
                    continue
                if strip is not None:                       # bỏ <think>...</think> khỏi đáp án
                    text = strip.feed(text)
                    if not text:
                        continue
                if not answer_started:
                    answer_started = True
                    for f in close_reasoning(flush=not config.stop_thinking_on_first_token):
                        yield f
                if not msg_open:
                    for f in em.open_message(msg_id):
                        yield f
                    msg_open = True
                got_answer = True
                yield em.message_delta(text)
                continue

            # kind == "step"
            cls = parser.classify(obj)
            if cls == "skip":
                continue
            if cls == "tool":
                for f in close_reasoning(flush=True):      # đóng Thinking TRƯỚC box Tool
                    yield f
                for f in emit_tool(*parser.parse_tool(obj)):
                    yield f
                continue
            if answer_started:
                continue
            for f in emit_block(thinking.feed(parser.thinking_text(obj))):
                yield f

        if strip is not None and msg_open:                  # xả nốt đệm đáp án của bộ lọc think
            tail = strip.flush()
            if tail:
                got_answer = True
                yield em.message_delta(tail)

        if not got_answer:                                  # không có token -> dùng thinking cuối
            tail = thinking.flush()
            for f in close_reasoning(flush=False):
                yield f
            if tail:
                for f in em.open_message(msg_id):
                    yield f
                msg_open = True
                yield em.message_delta(tail)

        for f in close_reasoning(flush=True):
            yield f
        if msg_open:
            for f in em.close_message():
                yield f
        yield em.completed()
        yield em.done()

    except Exception as exc:  # noqa: BLE001
        logger.exception("thinkbridge Open Responses stream failed")
        for f in em.failed(str(exc)):
            yield f
        yield em.done()


# --------------------------------------------------------------------------- #
# Non-streaming
# --------------------------------------------------------------------------- #
async def build_answer(config: ThinkAdapterConfig, parser: ThinkFrameParser, user_text: str, *,
                       user_id: str | None, conversation_id: str | None) -> str:
    """Rút cạn stream, trả về đáp án (ghép từ các token `data:`)."""
    text = ""
    last_thinking = ""
    async for kind, obj in call_chat_stream(config, user_text, user_id=user_id,
                                            conversation_id=conversation_id):
        if kind == "token":
            text += _token_text(obj)
        elif parser.classify(obj) == "thinking":
            last_thinking = parser.thinking_text(obj) or last_thinking
    answer = text or last_thinking
    if config.strip_think_from_answer:                       # bỏ <think>...</think> nếu còn sót
        s = ThinkStripper()
        answer = s.feed(answer) + s.flush()
    return answer


async def build_non_streaming_response(config: ThinkAdapterConfig, parser: ThinkFrameParser,
                                       user_text: str, *, response_id: str, model: str,
                                       user_id: str | None, tools: list | None,
                                       instructions: str | None,
                                       previous_response_id: str | None,
                                       conversation_id: str | None = None) -> dict:
    """Đối tượng Open Responses hoàn chỉnh (non-streaming)."""
    em = OpenResponsesEmitter(response_id, model, tools=tools, instructions=instructions,
                              previous_response_id=previous_response_id,
                              step_payload_max=config.step_payload_max, max_steps=config.max_steps,
                              reasoning_total_max=config.reasoning_total_max)
    text = await build_answer(config, parser, user_text, user_id=user_id,
                              conversation_id=conversation_id or response_id)
    em.open_message(f"msg_{uuid.uuid4().hex}")
    em.message_delta(text)
    em.close_message()
    em.completed()
    return em.final_object()
