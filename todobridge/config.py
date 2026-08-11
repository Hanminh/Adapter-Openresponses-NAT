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

    # --- HAI HỘP THINKING lấp khoảng trễ: 1 ở ĐẦU (trước write_todos), 1 ở CUỐI (sau khi mọi
    #     todo completed, trước đáp án). CHỈ hiện chỉ báo "Thinking", KHÔNG in nội dung thought,
    #     và KHÔNG có hộp thinking xen giữa các bước write_todos.
    show_thinking: bool = True
    # Nhãn (nội dung summary) của MỖI hộp thinking. PHẢI khác rỗng thì Open WebUI mới render hộp:
    # reasoning item rỗng (không có `reasoning_summary_text.delta`) thường KHÔNG hiện thành hộp nào.
    # Mặc định "Thinking" -> cả hộp ĐẦU (trước todo) lẫn hộp CUỐI (sau khi mọi todo xong) đều hiện
    # cùng một nhãn "Thinking". Đổi chuỗi nếu muốn nhãn khác (vd "Đang xử lý…").
    thinking_label: str = "Thinking"

    # --- KIỂU HIỂN THỊ DANH SÁCH TODO ---
    # False (mặc định): mở box khi todo chuyển `in_progress`, hoàn tất khi `completed` -> các box
    #   XUẤT HIỆN TUẦN TỰ (mỗi lúc chỉ 1 box đang chạy) -> Open WebUI KHÔNG gộp thành 1 nhóm
    #   "tool song song" -> danh sách PHẲNG, không có hộp bọc ngoài.
    # True: mở luôn cả todo đang `pending` ngay từ đầu (hiện đủ kế hoạch, nhưng nhiều box mở cùng
    #   lúc có thể bị Open WebUI bọc chung 1 nhóm).
    todo_show_pending: bool = False
