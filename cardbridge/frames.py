# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Đọc NAT intermediate step cho `cardbridge`.

Hai loại step quan tâm (đều do `stream_passthrough` đẩy ra):
  * `write_todos`     -> box tiến trình (tái dùng nguyên bộ parse của todobridge).
  * `package_details` -> output của script `package_details.py` (list JSON), phát thành item type
    RIÊNG cho client.

Bộ parse todos dùng lại từ `todobridge.frames` (KHÔNG lặp code). Ở đây chỉ thêm phần nhận diện +
bóc payload của step script.
"""

from __future__ import annotations

import html
import re

# Tái dùng toàn bộ logic todo của todobridge.
from todobridge.frames import (  # noqa: F401  (re-export cho translator)
    TodoBoxTracker,
    is_todo_step,
    parse_todo_step,
    status_icon,
    status_label,
    step_name,
)

# Tên step JSON mặc định (khớp SCRIPT_STEP_NAME phía workflow / tool_passthrough).
# PHẢI khác tên tool `send_package_details` — nếu không sẽ khớp nhầm step "lời gọi tool".
SCRIPT_STEP_MARKER = "package_details_payload"

# Bóc khối trong step payload. NAT StepAdaptor render `data.input`/`data.output` trong code fence.
_INPUT_RE = re.compile(r"\*\*Input:\*\*\s*```[a-zA-Z]*\n?(.*?)```", re.DOTALL)
_OUTPUT_RE = re.compile(r"\*\*Output:\*\*\s*```[a-zA-Z]*\n?(.*?)```", re.DOTALL)


def is_script_step(name: str, marker: str = SCRIPT_STEP_MARKER) -> bool:
    """True nếu step là output script passthrough — KHỚP CHÍNH XÁC theo tên step.

    Front-end NAT đặt name = "Tool: <step_name>". Ta so khớp phần sau "Tool: " BẰNG marker (không
    dùng substring) để `send_package_details` (lời gọi tool) KHÔNG bị nhận nhầm là step JSON.
    """
    if not marker or not name:
        return False
    bare = name[len("Tool: "):] if name.startswith("Tool: ") else name
    return bare.strip() == marker


def parse_script_step(step: dict) -> str | None:
    """Bóc text output (list JSON dạng chuỗi) từ payload step script.

    Ưu tiên khối **Input:** (chính là payload_text ta đẩy ở `_push_script_step`), fallback **Output:**.
    Trả None nếu không bóc được.
    """
    payload = html.unescape(str(step.get("payload") or ""))
    for rx in (_INPUT_RE, _OUTPUT_RE):
        m = rx.search(payload)
        if m:
            text = m.group(1).strip()
            if text:
                return text
    return None
