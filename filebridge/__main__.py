# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entrypoint: `python -m filebridge --nat-url ... --port 8001 --upload-dir /data/uploads`."""

from filebridge.app import main

if __name__ == "__main__":
    main()
