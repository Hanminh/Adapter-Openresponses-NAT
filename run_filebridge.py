# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convenience runner for the `filebridge` file-aware adapter.

Run (adapter on :8001, nat serve on :8000):
    python my_example/openresponses/run_filebridge.py \
        --nat-url http://localhost:8000/chat/stream --port 8001 --upload-dir /home/minhth11/Projects/Data_Upload
        
python my_example/openresponses/run_filebridge.py \
  --nat-url http://localhost:8000/chat/stream --port 8001 \
  --upload-dir /home/minhth11/Projects/Data_Upload \
  --owui-upload-dir /home/minhth11/Projects/API_UI/open-webui/.venv/lib/python3.13/site-packages/open_webui/data/uploads    

"""

import os
import sys

# Make both `natbridge` and `filebridge` importable regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from filebridge.app import main  # noqa: E402

if __name__ == "__main__":
    main()
