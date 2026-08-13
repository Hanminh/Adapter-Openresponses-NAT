# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dịch `/chat/stream` -> Open Responses, hiển thị MỘT checklist SỐNG (thay vì mỗi todo một box).

Ý tưởng như todobridge nhưng KHÁC cách hiển thị:
    todobridge : mỗi `content` = một box tool riêng (nhiều box).
    planbridge : TOÀN BỘ todos = MỘT khối checklist duy nhất, RE-RENDER tại chỗ mỗi khi status đổi
                 (✅ completed / 🔧 in_progress / ⬜ pending) -> giống trải nghiệm "to-do list".

Cơ chế cập-nhật-tại-chỗ: dùng ĐÚNG thủ thuật todobridge đã kiểm chứng — RE-EMIT
`response.output_item.done` cho CÙNG một item (cùng `output_index`/`id`) với nội dung mới; Open WebUI
render lại item đó thay vì tạo item mới. Ở đây item là MỘT hộp `reasoning` (khối "suy nghĩ/tiến
trình") chứa checklist.

Đọc todos: GIỐNG todobridge — NAT intermediate step tên `write_todos` (do
`agenticskills.deep_agents.todo_event_stream.as_nat_graph_todo_events` đẩy từ STATE `{todos:[...]}`).
State này GIỐNG NHAU cho cả `TodoListMiddleware` (nested) lẫn `FlatTodoMiddleware` (v3, phẳng) — nên
planbridge chạy với cả hai; khuyến nghị v3 (flat) để không lỗi 'Extra data' khi streaming.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from natbridge.nat_client import call_chat_stream
from natbridge.open_responses import OpenResponsesEmitter

from thinkbridge.frames import ThinkStripper

from todobridge.frames import is_todo_step, parse_todo_step, step_name  # đọc todos GIỐNG todobridge

from planbridge.config import PlanAdapterConfig

logger = logging.getLogger("planbridge.translator")

_ICON = {"completed": "✅", "in_progress": "🔧", "pending": "⬜"}


def _token_text(obj: dict) -> str:
    out = ""
    for choice in obj.get("choices", []) or []:
        out += (choice.get("delta") or {}).get("content") or ""
    return out


def _todos_from_step(obj: dict) -> list[dict] | None:
    if not isinstance(obj, dict) or not is_todo_step(step_name(obj)):
        return None
    return parse_todo_step(obj)


def _all_completed(todos: list[dict]) -> bool:
    return bool(todos) and all(str(t.get("status")) == "completed" for t in todos)


def render_checklist(todos: list[dict], title: str) -> str:
    """Toàn bộ todos -> một khối checklist người-đọc-được."""
    lines = [title]
    for t in todos:
        icon = _ICON.get(str(t.get("status")), "•")
        lines.append(f"{icon} {str(t.get('content', '')).strip()}")
    return "\n".join(lines)


class ChecklistEmitter(OpenResponsesEmitter):
    """Một hộp `reasoning` duy nhất, RE-RENDER tại chỗ mỗi khi checklist đổi.

    `open_checklist` phát `output_item.added` (reasoning, in_progress) MỘT LẦN. `update_checklist`
    đặt lại `summary[0].text` = checklist mới rồi RE-EMIT `output_item.done` cho CÙNG item -> Open
    WebUI cập nhật khối tại chỗ (đúng cơ chế todobridge dùng cho box tool). Item cuối `status=completed`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cl_idx: int | None = None
        self._cl_item: dict | None = None

    def open_checklist(self, item_id: str) -> list[str]:
        self._cl_idx = len(self._output)
        self._cl_item = {"id": item_id, "type": "reasoning", "status": "in_progress",
                         "summary": [{"type": "summary_text", "text": ""}]}
        self._output.append(self._cl_item)
        added = {"id": item_id, "type": "reasoning", "status": "in_progress", "summary": []}
        return [self._frame("response.output_item.added", {"output_index": self._cl_idx, "item": added})]

    def update_checklist(self, text: str, *, done: bool = False) -> list[str]:
        if self._cl_item is None:
            return []
        if self._step_payload_max:
            text = text[: self._reasoning_total_max or self._step_payload_max]
        self._cl_item["summary"][0]["text"] = text
        self._cl_item["status"] = "completed" if done else "in_progress"
        # RE-EMIT done cho cùng item_index -> Open WebUI render lại khối (không tạo khối mới).
        return [self._frame("response.output_item.done", {"output_index": self._cl_idx, "item": self._cl_item})]


# --------------------------------------------------------------------------- #
# /v1/responses — MỘT checklist sống (reasoning re-render) + đáp án
# --------------------------------------------------------------------------- #
async def stream_open_responses(config: PlanAdapterConfig, user_text: str, *, response_id: str,
                                model: str, user_id: str | None, tools: list | None,
                                instructions: str | None, previous_response_id: str | None,
                                conversation_id: str | None = None) -> AsyncIterator[str]:
    em = ChecklistEmitter(response_id, model, tools=tools, instructions=instructions,
                          previous_response_id=previous_response_id,
                          step_payload_max=config.step_payload_max, max_steps=config.max_steps,
                          reasoning_total_max=config.reasoning_total_max)
    strip = ThinkStripper() if config.strip_think_from_answer else None
    msg_id = f"msg_{uuid.uuid4().hex}"
    msg_open = False
    got_answer = False
    cl_open = False
    last_repr = None

    yield em.created()
    yield em.in_progress()
    try:
        async for kind, obj in call_chat_stream(config, user_text, user_id=user_id,
                                                conversation_id=conversation_id or response_id):
            if kind == "step":
                todos = _todos_from_step(obj)
                if todos is None:
                    continue                                   # step khác write_todos -> bỏ qua
                rep = json.dumps(todos, ensure_ascii=False, sort_keys=True)
                if rep == last_repr:                           # không đổi -> khỏi render lại
                    continue
                last_repr = rep
                if not cl_open:                                # mở hộp checklist lần đầu
                    for f in em.open_checklist(f"rs_{uuid.uuid4().hex}"):
                        yield f
                    cl_open = True
                for f in em.update_checklist(render_checklist(todos, config.checklist_title),
                                             done=_all_completed(todos)):
                    yield f
                continue

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

        # chốt checklist ở trạng thái cuối (đánh dấu hoàn tất nếu còn dở).
        if cl_open and config.complete_checklist_on_end and last_repr is not None:
            todos = json.loads(last_repr)
            for f in em.update_checklist(render_checklist(todos, config.checklist_title), done=True):
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
        logger.exception("planbridge Open Responses stream failed")
        for f in em.failed(str(exc)):
            yield f
        yield em.done()


# --------------------------------------------------------------------------- #
# /v1/chat/completions — không có "hộp" tách biệt -> checklist qua reasoning_content
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


async def stream_openai(config: PlanAdapterConfig, user_text: str, *, model: str,
                        user_id: str | None, conversation_id: str | None) -> AsyncIterator[str]:
    """/v1/chat/completions: reasoning_content chỉ APPEND (không update tại chỗ) -> phát mỗi snapshot
    checklist mới (có tiêu đề) mỗi khi đổi; đáp án vào content."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    strip = ThinkStripper() if config.strip_think_from_answer else None
    last_repr = None
    yield _openai_chunk(chat_id, model, role="assistant")
    try:
        async for kind, obj in call_chat_stream(config, user_text, user_id=user_id,
                                                conversation_id=conversation_id):
            if kind == "step":
                todos = _todos_from_step(obj)
                if todos is None:
                    continue
                rep = json.dumps(todos, ensure_ascii=False, sort_keys=True)
                if rep == last_repr:
                    continue
                last_repr = rep
                yield _openai_chunk(chat_id, model,
                                    reasoning_content="\n" + render_checklist(todos, config.checklist_title) + "\n")
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
        logger.exception("planbridge OpenAI stream failed")
        yield (f"data: {json.dumps({'error': {'message': str(exc), 'type': 'server_error', 'code': 'server_error'}})}"
               "\n\n")
        yield "data: [DONE]\n\n"


# --------------------------------------------------------------------------- #
# Non-streaming
# --------------------------------------------------------------------------- #
async def build_answer(config: PlanAdapterConfig, user_text: str, *, user_id: str | None,
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


async def build_non_streaming_response(config: PlanAdapterConfig, user_text: str, *,
                                       response_id: str, model: str, user_id: str | None,
                                       tools: list | None, instructions: str | None,
                                       previous_response_id: str | None,
                                       conversation_id: str | None = None) -> dict:
    em = ChecklistEmitter(response_id, model, tools=tools, instructions=instructions,
                          previous_response_id=previous_response_id,
                          step_payload_max=config.step_payload_max, max_steps=config.max_steps,
                          reasoning_total_max=config.reasoning_total_max)
    # Rút cạn: gom checklist cuối + đáp án.
    last_todos: list[dict] | None = None
    text = ""
    async for kind, obj in call_chat_stream(config, user_text, user_id=user_id,
                                            conversation_id=conversation_id or response_id):
        if kind == "step":
            todos = _todos_from_step(obj)
            if todos is not None:
                last_todos = todos
        elif kind == "token":
            text += _token_text(obj)
    if config.strip_think_from_answer:
        s = ThinkStripper()
        text = s.feed(text) + s.flush()
    if last_todos:
        em.open_checklist(f"rs_{uuid.uuid4().hex}")
        em.update_checklist(render_checklist(last_todos, config.checklist_title), done=True)
    em.open_message(f"msg_{uuid.uuid4().hex}")
    em.message_delta(text)
    em.close_message()
    em.completed()
    return em.final_object()
