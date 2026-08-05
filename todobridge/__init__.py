# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""todobridge — Open Responses adapter CHỈ hiển thị box `write_todos` của deep agent.

Module map:
  config.py      — TodoAdapterConfig (kế thừa filebridge, thêm show_answer/...).
  frames.py      — parse_token (tách sentinel todos khỏi token đáp án) + TodoBoxTracker.
  translator.py  — dịch `/chat/stream` -> Open Responses (box todos) / OpenAI (reasoning).
  app.py         — FastAPI app + CLI (`python -m todobridge`).
"""

from todobridge.app import create_app, main
from todobridge.config import TodoAdapterConfig

__all__ = ["TodoAdapterConfig", "create_app", "main"]
