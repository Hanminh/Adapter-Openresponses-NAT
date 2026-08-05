# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner cho `todobridge` — adapter CHỈ hiển thị box `write_todos` của deep agent.

Điều kiện: workflow là deep agent v2 (TodoListMiddleware) và graph được bọc bằng
`agenticskills.deep_agents.todo_event_stream.as_nat_graph_todo_events` để đẩy todos ra sentinel
trong luồng `data:`.

Giữ NGUYÊN chức năng file của filebridge/thinkbridge: nhận upload Open WebUI (POST /api/v1/files/),
đọc file gốc theo id trong thư mục uploads của Open WebUI, và chèn ĐƯỜNG DẪN TUYỆT ĐỐI của file
vào câu hỏi trước khi gửi nat serve (các cờ --upload-dir / --owui-upload-dir / --max-file-mb /
--keep-rag-context giống hệt thinkbridge).

Chạy (adapter :8003, nat serve :8000) — kèm chức năng file như run_thinkbridge:

    python my_example/openresponses/run_todobridge.py \
        --nat-url http://localhost:8000/chat/stream --port 8003 \
        --upload-dir /home/minhth11/Projects/Data_Upload \
        --owui-upload-dir /home/minhth11/Projects/API_UI/open-webui/.venv/lib/python3.13/site-packages/open_webui/data/uploads

Trong Open WebUI: thêm connection tới http://localhost:8003/v1, API Type = Responses
(để mỗi todo hiện thành một box "Tool Executed" riêng). `--hide-answer` nếu chỉ muốn box.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from todobridge.app import main  # noqa: E402

if __name__ == "__main__":
    main()
