# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Đọc NAT intermediate step của `write_todos` cho `todobridge` — GIỐNG CÁCH thinkbridge đọc box tool.

Emitter phía workflow (`agenticskills/deep_agents/todo_event_stream.py`) đẩy mỗi lần `write_todos`
thành một cặp TOOL_START/TOOL_END. Front-end NAT (`StepAdaptor`) phát ra `intermediate_data:` với:

    name    = "Tool: write_todos"
    payload = markdown chứa   **Input:**  ```json  {"todos":[...]}  ```   (+ **Output:** tương tự)

Module này:
  * `is_todo_step(name)`         -> step này có phải write_todos không.
  * `parse_todo_step(payload)`   -> danh sách todos bóc từ khối `**Input:**` (json.loads).
  * `TodoBoxTracker`             -> map mỗi `content` sang MỘT box ổn định (call_id), theo cơ chế
    TodoListMiddleware (mỗi write_todos THAY nguyên danh sách với status mới): MỞ box khi content
    xuất hiện lần đầu, HOÀN TẤT khi status -> "completed"; không tạo box trùng.
"""

from __future__ import annotations

import ast
import html
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Literal

TODO_TOOL_MARKER = "write_todos"     # khớp tên tool trong emitter
_INPUT_RE = re.compile(r"\*\*Input:\*\*\s*```[a-zA-Z]*\n?(.*?)```", re.DOTALL)
_OUTPUT_RE = re.compile(r"\*\*Output:\*\*\s*```[a-zA-Z]*\n?(.*?)```", re.DOTALL)

_STATUS_ICON = {"pending": "⬜", "in_progress": "🔧", "completed": "✅"}
_STATUS_LABEL = {"pending": "⬜ chờ", "in_progress": "🔧 đang làm", "completed": "✅ hoàn tất"}


def status_icon(status: str) -> str:
    return _STATUS_ICON.get(status, "•")


def status_label(status: str) -> str:
    return _STATUS_LABEL.get(status, status or "?")


def step_name(step: dict) -> str:
    return str(step.get("name") or "")


def is_todo_step(name: str) -> bool:
    """True nếu name của step là tool write_todos (vd 'Tool: write_todos')."""
    return TODO_TOOL_MARKER in name


def _loads_loose(raw: str):
    """Parse cả JSON (nháy kép) LẪN dict Python repr (nháy đơn).

    NAT `StepAdaptor` render `data.input`/`data.output` bằng `str(...)`. Với write_todos, input
    là dict args -> ra `{'todos': [...]}` DẤU NHÁY ĐƠN (không phải JSON). `json.loads` sẽ fail;
    `ast.literal_eval` đọc được. Thử JSON trước (nhanh), rồi literal_eval.
    """
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(raw)
        except (ValueError, SyntaxError):
            continue
    return None


def _todos_from_obj(data) -> list[dict] | None:
    if isinstance(data, dict):
        todos = data.get("todos")
        return todos if isinstance(todos, list) else None
    if isinstance(data, list):
        return data
    return None


def parse_todo_step(step: dict) -> list[dict] | None:
    """Bóc danh sách todos từ payload step.

    Ưu tiên khối **Input:** (chính là args của write_todos: `{'todos': [...]}`). Khối **Output:**
    thường là `Command(update={...})` KHÔNG parse literal được, nên chỉ dùng làm phương án cuối
    bằng cách bóc phần `{'todos': [...]}` bên trong.
    """
    payload = html.unescape(str(step.get("payload") or ""))

    # 1) Ưu tiên **Input:** — sạch nhất.
    m = _INPUT_RE.search(payload)
    if m:
        todos = _todos_from_obj(_loads_loose(m.group(1).strip()))
        if todos is not None:
            return todos

    # 2) Fallback **Output:** — thường là `Command(update={'todos':[...], 'messages':[...]})`.
    #    Không literal_eval cả Command được, nên cắt riêng list sau `'todos':` bằng cách dò dấu
    #    ngoặc vuông cân bằng rồi literal_eval phần list đó.
    m = _OUTPUT_RE.search(payload)
    if m:
        text = m.group(1)
        km = re.search(r"['\"]todos['\"]\s*:\s*\[", text)
        if km:
            start = text.index("[", km.start())
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        lst = _loads_loose(text[start:i + 1])
                        if isinstance(lst, list):
                            return lst
                        break
    return None


# --------------------------------------------------------------------------- #
# Theo dõi box theo `content`
# --------------------------------------------------------------------------- #
BoxEvent = tuple[Literal["open", "complete"], str, str, str]   # (kind, content, call_id, status)


@dataclass(slots=True)
class _Box:
    call_id: str
    status: str
    completed_emitted: bool = False


@dataclass(slots=True)
class TodoBoxTracker:
    """Biến các snapshot todos (mỗi write_todos) thành sự kiện box gia tăng.

    - Lần đầu thấy `content` -> ("open", content, call_id, status).
    - Khi `content` -> "completed" (chưa phát) -> ("complete", ...).
    - Cùng content ở snapshot sau KHÔNG mở lại (dedup theo content).
    """

    _boxes: dict[str, _Box] = field(default_factory=dict)

    def update(self, todos: list[dict], *, open_pending: bool = False) -> list[BoxEvent]:
        """Sinh sự kiện box từ một snapshot todos.

        `open_pending=False` (mặc định): CHỈ mở box khi todo đã `in_progress`/`completed` -> các
        box xuất hiện TUẦN TỰ (không mở loạt box `pending` cùng lúc), tránh Open WebUI gộp nhóm.
        `open_pending=True`: mở box ngay cả khi `pending` (hiện đủ kế hoạch từ đầu).
        """
        events: list[BoxEvent] = []
        for t in todos:
            content = str(t.get("content", "")).strip()
            if not content:
                continue
            status = str(t.get("status", "pending"))
            box = self._boxes.get(content)
            if box is None:
                should_open = status in ("in_progress", "completed") or open_pending
                if not should_open:
                    continue                      # pending & chưa tới lượt -> chưa mở
                box = _Box(call_id=f"todo_{uuid.uuid4().hex[:12]}", status=status)
                self._boxes[content] = box
                events.append(("open", content, box.call_id, status))
            else:
                box.status = status
            if status == "completed" and not box.completed_emitted:
                box.completed_emitted = True
                events.append(("complete", content, box.call_id, status))
        return events

    def pending_completions(self) -> list[BoxEvent]:
        """Đóng nốt các box chưa completed khi stream kết thúc."""
        out: list[BoxEvent] = []
        for content, box in self._boxes.items():
            if not box.completed_emitted:
                box.completed_emitted = True
                out.append(("complete", content, box.call_id, box.status))
        return out
