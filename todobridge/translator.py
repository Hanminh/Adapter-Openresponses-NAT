# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dịch `/chat/stream` của nat serve -> Open Responses / OpenAI, CHỈ hiển thị box `write_todos`.

Giống thinkbridge: hiển thị DỰA TRÊN LỜI GỌI TOOL (NAT intermediate step). Cụ thể:

    intermediate step "Tool: write_todos"  -> dựng/cập nhật BOX (một box mỗi `content`)
    token `data:`                          -> ĐÁP ÁN, stream vào message (nếu show_answer)
    step khác (thinking/tool khác)         -> BỎ QUA

Ánh xạ theo cơ chế TodoListMiddleware (mỗi write_todos THAY nguyên danh sách với status mới):
mỗi `content` = một box; MỞ khi xuất hiện lần đầu, HOÀN TẤT khi status='completed'.
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

from todobridge.config import TodoAdapterConfig
from todobridge.frames import (
    TodoBoxTracker,
    is_todo_step,
    parse_todo_step,
    status_icon,
    status_label,
    step_name,
)

logger = logging.getLogger("todobridge.translator")


def _token_text(obj: dict) -> str:
    out = ""
    for choice in obj.get("choices", []) or []:
        out += (choice.get("delta") or {}).get("content") or ""
    return out


def _box_args(status: str) -> str:
    return json.dumps({"status": status, "label": status_label(status)}, ensure_ascii=False)


class TodoResponsesEmitter(OpenResponsesEmitter):
    """Emitter con: box tool CÓ TRẠNG THÁI ĐỘNG.

    `natbridge.OpenResponsesEmitter.tool_call` phát `function_call` với `status="completed"` +
    `done` NGAY -> Open WebUI hiện box "đã xong" tức thì. Ở đây tách vòng đời như hộp reasoning:

      * `open_todo_box`     -> `response.output_item.added` với item `status="in_progress"`
                               (CHƯA `done`) => box hiển thị "ĐANG CHẠY" (spinner).
      * `complete_todo_box` -> `response.output_item.done` (status="completed") + một
                               `function_call_output` (kết quả ✅) => box chuyển "ĐÃ HOÀN THÀNH".

    Mọi item vẫn được ghi vào `self._output` để object `response.completed` cuối cùng nhất quán.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._todo_boxes: dict[str, tuple[int, dict]] = {}   # call_id -> (output_index, item)

    def open_todo_box(self, call_id: str, name: str, status: str) -> list[str]:
        idx = len(self._output)
        item = {"id": f"fc_{uuid.uuid4().hex}", "type": "function_call", "status": "in_progress",
                "call_id": call_id, "name": name, "arguments": _box_args(status)}
        self._output.append(item)
        self._todo_boxes[call_id] = (idx, item)
        return [self._frame("response.output_item.added", {"output_index": idx, "item": item})]

    def complete_todo_box(self, call_id: str, result: str) -> list[str]:
        entry = self._todo_boxes.get(call_id)
        if entry is None:
            return []
        idx, item = entry
        item["status"] = "completed"
        item["arguments"] = _box_args("completed")
        frames = [self._frame("response.output_item.done", {"output_index": idx, "item": item})]
        out_idx = len(self._output)
        out_item = {"id": f"fco_{uuid.uuid4().hex}", "type": "function_call_output",
                    "status": "completed", "call_id": call_id,
                    "output": [{"type": "output_text", "text": result}]}
        self._output.append(out_item)
        frames += [
            self._frame("response.output_item.added", {"output_index": out_idx, "item": out_item}),
            self._frame("response.output_item.done", {"output_index": out_idx, "item": out_item}),
        ]
        return frames


def _todos_from_step(obj: dict) -> list[dict] | None:
    """Nếu obj là step write_todos -> trả todos, ngược lại None."""
    if not isinstance(obj, dict) or not is_todo_step(step_name(obj)):
        return None
    return parse_todo_step(obj)


# --------------------------------------------------------------------------- #
# /v1/responses — mỗi todo là một box (function_call + function_call_output)
# --------------------------------------------------------------------------- #
async def stream_open_responses(config: TodoAdapterConfig, user_text: str, *, response_id: str,
                                model: str, user_id: str | None, tools: list | None,
                                instructions: str | None, previous_response_id: str | None,
                                conversation_id: str | None = None) -> AsyncIterator[str]:
    em = TodoResponsesEmitter(response_id, model, tools=tools, instructions=instructions,
                              previous_response_id=previous_response_id,
                              step_payload_max=config.step_payload_max, max_steps=config.max_steps,
                              reasoning_total_max=config.reasoning_total_max)
    tracker = TodoBoxTracker()
    strip = ThinkStripper() if config.strip_think_from_answer else None
    msg_id = f"msg_{uuid.uuid4().hex}"
    msg_open = False
    got_answer = False

    def emit_events(events) -> list[str]:
        frames: list[str] = []
        for kind, content, call_id, status in events:
            if kind == "open":                       # in_progress / pending -> box ĐANG CHẠY
                frames += em.open_todo_box(call_id, content, status)
            elif kind == "complete":                 # completed -> box ĐÃ HOÀN THÀNH
                frames += em.complete_todo_box(call_id, config.completed_result)
        return frames

    yield em.created()
    yield em.in_progress()
    try:
        async for kind, obj in call_chat_stream(config, user_text, user_id=user_id,
                                                conversation_id=conversation_id or response_id):
            if kind == "step":
                todos = _todos_from_step(obj)
                if todos is not None:
                    for f in emit_events(tracker.update(todos)):
                        yield f
                continue                              # step KHÁC write_todos -> bỏ qua

            # kind == "token" -> đáp án
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
            for f in emit_events(tracker.pending_completions()):
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
        logger.exception("todobridge Open Responses stream failed")
        for f in em.failed(str(exc)):
            yield f
        yield em.done()


# --------------------------------------------------------------------------- #
# /v1/chat/completions — không có box tool -> render todos thành dòng reasoning
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


async def stream_openai(config: TodoAdapterConfig, user_text: str, *, model: str,
                        user_id: str | None, conversation_id: str | None) -> AsyncIterator[str]:
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    tracker = TodoBoxTracker()
    strip = ThinkStripper() if config.strip_think_from_answer else None
    yield _openai_chunk(chat_id, model, role="assistant")
    try:
        async for kind, obj in call_chat_stream(config, user_text, user_id=user_id,
                                                conversation_id=conversation_id):
            if kind == "step":
                todos = _todos_from_step(obj)
                if todos is not None:
                    for ev_kind, content, _cid, status in tracker.update(todos):
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
        logger.exception("todobridge OpenAI stream failed")
        yield (f"data: {json.dumps({'error': {'message': str(exc), 'type': 'server_error', 'code': 'server_error'}})}"
               "\n\n")
        yield "data: [DONE]\n\n"


# --------------------------------------------------------------------------- #
# Non-streaming
# --------------------------------------------------------------------------- #
async def build_answer(config: TodoAdapterConfig, user_text: str, *, user_id: str | None,
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


async def build_non_streaming_response(config: TodoAdapterConfig, user_text: str, *,
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
