# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cardbridge — Open Responses adapter: box write_todos (không thinking) + item script_output.

Module map:
  config.py      — CardAdapterConfig (kế thừa todobridge; show_thinking=False + script_output_*).
  frames.py      — tái dùng parse todos của todobridge + parse step script (package_details).
  translator.py  — dịch `/chat/stream` -> Open Responses (box todos + item script_output) / OpenAI.
  app.py         — FastAPI app + CLI (`python -m cardbridge`).
"""

from cardbridge.app import create_app, main
from cardbridge.config import CardAdapterConfig

__all__ = ["CardAdapterConfig", "create_app", "main"]
