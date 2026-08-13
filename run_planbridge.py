# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner cho `planbridge` — hiển thị MỘT checklist SỐNG của deep agent (thay vì nhiều box như todobridge).

Điều kiện: workflow là deep agent v3 (khuyến nghị: FlatTodoMiddleware -> không lỗi 'Extra data' khi
streaming) HOẶC v2, và graph bọc bằng
`agenticskills.deep_agents.todo_event_stream.as_nat_graph_todo_events` (đẩy todos thành NAT step).

Chạy (adapter :8004, nat serve :8000) — kèm chức năng file như run_todobridge:

    python my_example/openresponses/run_planbridge.py \
        --nat-url http://localhost:8000/chat/stream --port 8004 \
        --upload-dir /home/minhth11/Projects/Data_Upload \
        --owui-upload-dir /home/minhth11/Projects/API_UI/open-webui/.venv/lib/python3.13/site-packages/open_webui/data/uploads

Trong Open WebUI: thêm connection tới http://localhost:8004/v1, API Type = Responses.
`--hide-answer` nếu chỉ muốn checklist; `--title` để đổi tiêu đề.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planbridge.app import main  # noqa: E402

if __name__ == "__main__":
    main()
