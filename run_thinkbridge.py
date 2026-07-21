# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner cho `thinkbridge` — adapter "mọi thứ còn lại đều là thinking".

Khác `run_filebridge.py`: bridge này lấy ĐÁP ÁN CUỐI từ luồng token `data:` của nat serve (thay vì
dò chuỗi "Final Answer:" trong payload), nên chạy được cho CẢ HAI kiểu workflow:

    workflow._type: react_agent          (langgraph_wrapper khai báo như một tool)
    workflow._type: langgraph_wrapper    (agent LangGraph làm workflow luôn)

Mọi intermediate step không phải tool đều rơi vào hộp Thinking. Model "nghĩ" cho tới khi token đáp
án đầu tiên xuất hiện.

Chạy (adapter :8002, nat serve :8000):

    python my_example/openresponses/run_thinkbridge.py \
        --nat-url http://localhost:8000/chat/stream --port 8002 \
        --upload-dir /home/minhth11/Projects/Data_Upload \
        --owui-upload-dir /home/minhth11/Projects/API_UI/open-webui/.venv/lib/python3.13/site-packages/open_webui/data/uploads

Trong Open WebUI: thêm connection tới http://localhost:8002/v1
  * API Type = Responses  -> tool hiện thành box "Tool Executed" riêng (khuyến nghị)
  * API Type = OpenAI     -> tool hiện dưới dạng dòng 🔧 trong hộp Thinking
"""

import os
import sys

# Cho phép import cả `natbridge`, `filebridge` lẫn `thinkbridge` dù chạy từ thư mục nào.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from thinkbridge.app import main  # noqa: E402

if __name__ == "__main__":
    main()
