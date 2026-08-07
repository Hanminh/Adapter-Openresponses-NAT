# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config cho `todobridge` — adapter hiển thị tiến trình `write_todos` + hộp Thinking.

Giống `thinkbridge`/`filebridge` (vẫn nhận file upload của Open WebUI, chèn đường dẫn tuyệt đối).
Cách hiển thị:
  * Mỗi todo (từ TodoListMiddleware) -> MỘT box, mở khi `in_progress` (đang chạy) và hoàn tất
    khi `completed`. Các box xuất hiện TUẦN TỰ -> danh sách phẳng, không bị Open WebUI gộp nhóm.
  * Mọi hoạt động KHÁC (LLM suy nghĩ, chạy bash...) -> hộp "Thinking" (reasoning) để lấp khoảng
    trễ TRƯỚC khi write_todos đầu tiên xuất hiện và SAU khi write_todos cuối hoàn tất.
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

    # --- HỘP THINKING lấp khoảng trễ TRƯỚC/SAU write_todos ---
    # Với mọi intermediate step KHÔNG phải write_todos (LLM đang suy nghĩ, chạy bash, đọc kết
    # quả...), gom vào một hộp "Thinking" (reasoning). Nhờ đó khoảng trễ trước khi write_todos
    # đầu tiên xuất hiện và sau khi write_todos cuối hoàn tất KHÔNG còn "đứng hình".
    show_thinking: bool = True
    # Cắt phần sau "Final Answer:" khỏi hộp thinking (đáp án đã stream riêng qua data:).
    strip_final_answer_from_thinking: bool = True

    # --- KIỂU HIỂN THỊ DANH SÁCH TODO ---
    # False (mặc định): mở box khi todo chuyển `in_progress`, hoàn tất khi `completed` -> các box
    #   XUẤT HIỆN TUẦN TỰ (mỗi lúc chỉ 1 box đang chạy) -> Open WebUI KHÔNG gộp thành 1 nhóm
    #   "tool song song" -> danh sách PHẲNG, không có hộp bọc ngoài.
    # True: mở luôn cả todo đang `pending` ngay từ đầu (hiện đủ kế hoạch, nhưng nhiều box mở cùng
    #   lúc có thể bị Open WebUI bọc chung 1 nhóm).
    todo_show_pending: bool = False
