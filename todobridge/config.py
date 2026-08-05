# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config cho `todobridge` — adapter CHỈ hiển thị các box `write_todos`.

Giống `thinkbridge`/`filebridge` (vẫn nhận file upload của Open WebUI, chèn đường dẫn tuyệt đối),
nhưng KHÁC ở cách hiển thị: bỏ hết box Thinking / box tool thường; chỉ dựng MỖI todo (từ
TodoListMiddleware) thành một box, phản ánh status pending/in_progress/completed.
"""

from __future__ import annotations

from dataclasses import dataclass

from filebridge.config import FileAdapterConfig


@dataclass(slots=True)
class TodoAdapterConfig(FileAdapterConfig):
    """Cấu hình todobridge.

    Nguồn todos: emitter phía workflow (`todo_event_stream.as_nat_graph_todo_events`) đẩy mỗi
    lần `write_todos` thành một NAT intermediate step "Tool: write_todos". Bridge đọc step này
    (GIỐNG cách thinkbridge đọc box tool) rồi dựng box theo từng `content`.
    """

    # Có stream luôn câu trả lời cuối (token `data:` KHÔNG phải todos) vào message không.
    # True (mặc định) -> vừa hiện box todos, vừa hiện đáp án. False -> CHỈ box todos.
    show_answer: bool = True

    # Có tự đóng (đánh dấu hoàn tất) các box còn dở khi stream kết thúc không.
    complete_open_boxes_on_end: bool = True

    # Nội dung phần "kết quả" của box khi todo hoàn tất.
    completed_result: str = "✅ Hoàn tất"

    # Lọc `<think>...</think>` khỏi ĐÁP ÁN (lưới an toàn; emitter thường đã lọc sẵn).
    strip_think_from_answer: bool = True
