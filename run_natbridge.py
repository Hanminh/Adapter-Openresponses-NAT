# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convenience runner for the `natbridge` adapter package.

Run (adapter on :8001, nat serve on :8000):
    python my_example/openresponses/run_natbridge.py \
        --nat-url http://localhost:8000/chat/stream --port 8001

Equivalent to `python -m natbridge ...` when run from this directory.
"""

import os
import sys

# Make the `natbridge` package importable regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from natbridge.app import main  # noqa: E402

if __name__ == "__main__":
    main()
