# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dịch `/chat/stream` của nat serve -> Open Responses / OpenAI cho `cardbridge`.

Khác todobridge:
  * KHÔNG có hộp Thinking bọc quanh danh sách todo.
  * Step `package_details` (output script passthrough) -> phát ITEM TYPE RIÊNG (`script_output`)
    để client tách khỏi câu trả lời và tự render (thẻ gói).

Ánh xạ:
  step write_todos     -> box tiến trình (mở khi in_progress, hoàn tất khi completed).
  step package_details -> item `script_output` (name=package_details, output_text=list JSON).
  token `data:`        -> câu trả lời cuối (agent vẫn trả lời bằng chữ như hiện tại).
  step khác            -> BỎ QUA.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from natbridge.nat_client import call_chat_stream
from natbridge.open_responses import OpenResponsesEmitter

from thinkbridge.frames import ThinkStripper   # tái dùng bộ lọc <think>

from todobridge.translator import TodoResponsesEmitter, _all_completed, _token_text

from cardbridge.config import CardAdapterConfig
from cardbridge.frames import (
    TodoBoxTracker,
    is_script_step,
    is_todo_step,
    parse_script_step,
    parse_todo_step,
    status_icon,
    step_name,
)

logger = logging.getLogger("cardbridge.translator")


class CardResponsesEmitter(TodoResponsesEmitter):
    """TodoResponsesEmitter + item `script_output` cho output của script passthrough."""

    def emit_script_output(self, text: str, *, name: str, item_type: str) -> list[str]:
        idx = len(self._output)
        item = {
            "id": f"so_{uuid.uuid4().hex}",
            "type": item_type,          # vd "script_output" -> client key theo đây
            "status": "completed",
            "name": name,               # vd "package_details"
            "output": [{"type": "output_text", "text": text}],
        }
        self._output.append(item)
        return [
            self._frame("response.output_item.added", {"output_index": idx, "item": item}),
            self._frame("response.output_item.done", {"output_index": idx, "item": item}),
        ]


def _classify(obj: dict, config: CardAdapterConfig):
    """Phân loại step: ("todo", todos) | ("script", text) | None (bỏ qua)."""
    if not isinstance(obj, dict):
        return None
    name = step_name(obj)
    if is_script_step(name, config.script_step_marker):
        text = parse_script_step(obj)
        return ("script", text) if text else None
    if is_todo_step(name):
        todos = parse_todo_step(obj)
        return ("todo", todos) if todos is not None else None
    return None


# --------------------------------------------------------------------------- #
# /v1/responses — box todos + item script_output + message đáp án (KHÔNG thinking)
# --------------------------------------------------------------------------- #
async def stream_open_responses(config: CardAdapterConfig, user_text: str, *, response_id: str,
                                model: str, user_id: str | None, tools: list | None,
                                instructions: str | None, previous_response_id: str | None,
                                conversation_id: str | None = None) -> AsyncIterator[str]:
    em = CardResponsesEmitter(response_id, model, tools=tools, instructions=instructions,
                              previous_response_id=previous_response_id,
                              step_payload_max=config.step_payload_max, max_steps=config.max_steps,
                              reasoning_total_max=config.reasoning_total_max)
    tracker = TodoBoxTracker()
    strip = ThinkStripper() if config.strip_think_from_answer else None
    msg_id = f"msg_{uuid.uuid4().hex}"
    msg_open = False
    got_answer = False

    def emit_todo_events(events) -> list[str]:
        frames: list[str] = []
        for kind, content, call_id, status in events:
            if kind == "open":
                frames += em.open_todo_box(call_id, content, status)
            elif kind == "complete":
                frames += em.complete_todo_box(call_id, config.completed_result)
        return frames

    yield em.created()
    yield em.in_progress()
    try:
        async for kind, obj in call_chat_stream(config, user_text, user_id=user_id,
                                                conversation_id=conversation_id or response_id):
            if kind == "step":
                info = _classify(obj, config)
                if info is None:
                    continue                          # step khác -> BỎ QUA (không thinking chen giữa)
                if info[0] == "script":
                    text = info[1]
                    if config.script_payload_max:
                        text = text[: config.script_payload_max]
                    for f in em.emit_script_output(text, name=config.script_output_name,
                                                   item_type=config.script_output_type):
                        yield f
                    continue
                # info[0] == "todo"
                for f in emit_todo_events(tracker.update(info[1], open_pending=config.todo_show_pending)):
                    yield f
                continue

            # kind == "token" -> câu trả lời
            text = _token_text(obj)
            if not text or not config.show_answer:
                continue
            if strip is not None:
                text = strip.feed(text)
                if not text:
                    continue
            if not msg_open:
                for f in em.open_message(msg_id):
                    yield f
                msg_open = True
            got_answer = True
            yield em.message_delta(text)

        if strip is not None and msg_open:
            tail = strip.flush()
            if tail:
                got_answer = True
                yield em.message_delta(tail)

        if config.complete_open_boxes_on_end:
            for f in emit_todo_events(tracker.pending_completions()):
                yield f

        if msg_open:
            for f in em.close_message():
                yield f
        elif not got_answer:
            for f in em.open_message(msg_id):
                yield f
            yield em.message_delta("")
            for f in em.close_message():
                yield f

        yield em.completed()
        yield em.done()
    except Exception as exc:  # noqa: BLE001
        logger.exception("cardbridge Open Responses stream failed")
        for f in em.failed(str(exc)):
            yield f
        yield em.done()


# --------------------------------------------------------------------------- #
# /v1/chat/completions — không có item riêng -> todos thành reasoning; script BỎ QUA
# --------------------------------------------------------------------------- #
def _openai_chunk(chat_id: str, model: str, *, content: str | None = None,
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


async def stream_openai(config: CardAdapterConfig, user_text: str, *, model: str,
                        user_id: str | None, conversation_id: str | None) -> AsyncIterator[str]:
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    tracker = TodoBoxTracker()
    strip = ThinkStripper() if config.strip_think_from_answer else None
    yield _openai_chunk(chat_id, model, role="assistant")
    try:
        async for kind, obj in call_chat_stream(config, user_text, user_id=user_id,
                                                conversation_id=conversation_id):
            if kind == "step":
                info = _classify(obj, config)
                if info is None or info[0] == "script":
                    continue                          # OpenAI mode không có item riêng -> bỏ script
                for ev_kind, content, _cid, status in tracker.update(
                        info[1], open_pending=config.todo_show_pending):
                    icon = "✅" if ev_kind == "complete" else status_icon(status)
                    yield _openai_chunk(chat_id, model, reasoning_content=f"{icon} {content}\n")
                continue
            text = _token_text(obj)
            if not text or not config.show_answer:
                continue
            if strip is not None:
                text = strip.feed(text)
                if not text:
                    continue
            yield _openai_chunk(chat_id, model, content=text)

        if strip is not None:
            tail = strip.flush()
            if tail:
                yield _openai_chunk(chat_id, model, content=tail)
        yield _openai_chunk(chat_id, model, finish_reason="stop")
        yield "data: [DONE]\n\n"
    except Exception as exc:  # noqa: BLE001
        logger.exception("cardbridge OpenAI stream failed")
        yield (f"data: {json.dumps({'error': {'message': str(exc), 'type': 'server_error', 'code': 'server_error'}})}"
               "\n\n")
        yield "data: [DONE]\n\n"


# --------------------------------------------------------------------------- #
# Non-streaming
# --------------------------------------------------------------------------- #
async def build_answer(config: CardAdapterConfig, user_text: str, *, user_id: str | None,
                       conversation_id: str | None) -> str:
    text = ""
    async for kind, obj in call_chat_stream(config, user_text, user_id=user_id,
                                            conversation_id=conversation_id):
        if kind == "token":
            text += _token_text(obj)
    if config.strip_think_from_answer:
        s = ThinkStripper()
        text = s.feed(text) + s.flush()
    return text


async def build_non_streaming_response(config: CardAdapterConfig, user_text: str, *,
                                       response_id: str, model: str, user_id: str | None,
                                       tools: list | None, instructions: str | None,
                                       previous_response_id: str | None,
                                       conversation_id: str | None = None) -> dict:
    em = OpenResponsesEmitter(response_id, model, tools=tools, instructions=instructions,
                              previous_response_id=previous_response_id,
                              step_payload_max=config.step_payload_max, max_steps=config.max_steps,
                              reasoning_total_max=config.reasoning_total_max)
    text = await build_answer(config, user_text, user_id=user_id,
                              conversation_id=conversation_id or response_id)
    em.open_message(f"msg_{uuid.uuid4().hex}")
    em.message_delta(text)
    em.close_message()
    em.completed()
    return em.final_object()
