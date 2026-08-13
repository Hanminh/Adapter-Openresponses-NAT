# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""planbridge — Open Responses adapter hiển thị MỘT checklist SỐNG (ý tưởng như todobridge).

Module map:
  config.py      — PlanAdapterConfig (kế thừa TodoAdapterConfig, thêm checklist_title).
  translator.py  — ChecklistEmitter: MỘT hộp reasoning, re-render tại chỗ mỗi khi todos đổi.
  app.py         — FastAPI app + CLI (`python -m planbridge`).
Đọc todos GIỐNG todobridge (NAT step `write_todos`) -> chạy với deep agent v2 (nested) lẫn v3 (flat).
"""

from planbridge.app import create_app, main
from planbridge.config import PlanAdapterConfig

__all__ = ["PlanAdapterConfig", "create_app", "main"]
