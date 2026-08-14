# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config cho `cardbridge` — như todobridge (box write_todos) NHƯNG:

  * KHÔNG có hộp Thinking bọc quanh danh sách todo (`show_thinking=False`).
  * Thêm item type RIÊNG cho output của script `package_details` (`script_output`) — client nhận
    biết để render thẻ gói.
"""

from __future__ import annotations

from dataclasses import dataclass

from todobridge.config import TodoAdapterConfig


@dataclass(slots=True)
class CardAdapterConfig(TodoAdapterConfig):
    """Cấu hình cardbridge (kế thừa todobridge)."""

    # BỎ hộp thinking bọc quanh todo (yêu cầu: không thêm thinking trước/sau list todo).
    show_thinking: bool = False

    # --- Item output của script passthrough ---
    # Tên step JSON phía workflow (khớp SCRIPT_STEP_NAME của tool_passthrough). PHẢI khác tên tool
    # `send_package_details` để bridge KHÔNG tóm nhầm step "lời gọi tool" (Input = args).
    script_step_marker: str = "package_details_payload"
    # `type` của item gửi client -> client key theo đây để tách khỏi câu trả lời.
    script_output_type: str = "carousel"
    # `name` gắn kèm item (để client biết là gói cước).
    script_output_name: str = "package_details"
    # Giới hạn ký tự phần TEXT thô của item script. 0 = KHÔNG cắt (mặc định) -> item luôn chứa
    # TOÀN BỘ output của script. Trường `data` (list object JSON đã parse) KHÔNG bao giờ bị cắt.
    script_payload_max: int = 0
