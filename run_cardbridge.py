# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner cho `cardbridge` — box write_todos (không thinking) + item `script_output`.

Điều kiện: workflow là deep agent passthrough (build_native_optimize_deep_agent_passthrough) và
graph được bọc bằng `agenticskills.deep_agents.stream_passthrough.as_nat_graph_passthrough` (đẩy
step write_todos + package_details).

Giữ NGUYÊN chức năng file của filebridge/todobridge (nhận upload Open WebUI, chèn đường dẫn tuyệt
đối). Cờ --upload-dir / --owui-upload-dir / --max-file-mb / --keep-rag-context giống todobridge.

Chạy (adapter :8005, nat serve :8000):

    python my_example/openresponses/run_cardbridge.py \
        --nat-url http://localhost:8000/chat/stream --port 8005 \
        --upload-dir /home/minhth11/Projects/Data_Upload \
        --owui-upload-dir /home/minhth11/Projects/API_UI/open-webui/.venv/lib/python3.13/site-packages/open_webui/data/uploads

Trong Open WebUI: thêm connection tới http://localhost:8005/v1, API Type = Responses.
Output của script package_details tới client dưới item `type: "script_output"` (name=package_details).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cardbridge.app import main  # noqa: E402

if __name__ == "__main__":
    main()
