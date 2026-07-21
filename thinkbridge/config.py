# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config for `thinkbridge` — the "everything else is thinking" adapter.

Kế thừa toàn bộ phần xử lý file của `filebridge` (upload Open WebUI -> lưu đĩa -> chèn đường dẫn
tuyệt đối vào câu hỏi), chỉ thêm các knob cho cách phân loại frame mới.
"""

from __future__ import annotations

from dataclasses import dataclass

from filebridge.config import FileAdapterConfig


@dataclass(slots=True)
class ThinkAdapterConfig(FileAdapterConfig):
    """Cấu hình thinkbridge.

    Quy tắc phân loại của bridge này (xem `thinkbridge.frames`):

        TOOL   = intermediate step có name bắt đầu bằng `tool_name_prefix` ("Tool:")
        FINAL  = các token trên dòng `data:` (ChatResponseChunk) của nat serve
        THINKING = MỌI THỨ CÒN LẠI

    Nhờ FINAL lấy từ `data:` chứ không dò text marker, bridge chạy được cho CẢ HAI kiểu workflow
    (`react_agent` và `langgraph_wrapper`) — xem README.
    """

    # Cắt phần sau "Final Answer:" ra khỏi khối thinking (chỉ có ý nghĩa với react_agent, vốn nhét
    # cả suy nghĩ lẫn đáp án vào cùng một **Output:**). Giữ "Thought:", bỏ đáp án — vì đáp án đã
    # được stream qua `data:` rồi, để lại sẽ bị lặp hai lần.
    strip_final_answer_from_thinking: bool = True

    # Khi token đáp án đầu tiên xuất hiện -> ngừng phát thinking và VỨT khối thinking đang dở.
    # Với `langgraph_wrapper`, khối đang dở đó chính là đáp án cuối (LLM_NEW_TOKEN của lượt cuối),
    # nên không vứt là lặp nội dung.
    stop_thinking_on_first_token: bool = True

    # Hiện box tool riêng (Open Responses) / dòng 🔧 trong Thinking (chat/completions).
    show_tools: bool = True

    # Lọc khối `<think>...</think>` inline ra khỏi ĐÁP ÁN cuối. Lưới an toàn: nếu workflow không
    # tự lọc (VD react_agent với Qwen bật thinking, không đi qua StreamSafeGraph), phần reasoning
    # sẽ lẫn vào `data:` -> ta cắt tại đây. Với langgraph_wrapper + StreamSafeGraph thì đáp án đã
    # sạch sẵn nên bộ lọc này chỉ chạy không.
    strip_think_from_answer: bool = True
