# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config cho `planbridge` — hiển thị MỘT checklist sống (kế thừa todobridge/filebridge)."""

from __future__ import annotations

from dataclasses import dataclass

from todobridge.config import TodoAdapterConfig


@dataclass(slots=True)
class PlanAdapterConfig(TodoAdapterConfig):
    """Như `TodoAdapterConfig` nhưng render TOÀN BỘ todos thành MỘT khối checklist re-render tại chỗ.

    Kế thừa mọi cờ file (upload_dir, owui_upload_dir, ...) + show_answer/strip_think_from_answer.
    """

    # Tiêu đề dòng đầu của khối checklist.
    checklist_title: str = "📋 Kế hoạch:"

    # Khi stream kết thúc, đánh dấu checklist là hoàn tất (status khối -> completed).
    complete_checklist_on_end: bool = True
