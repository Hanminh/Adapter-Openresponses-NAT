# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""File-aware Open Responses adapter.

Reuses `natbridge`'s Open Responses / OpenAI translators (API standard preserved) and adds a
front step that saves Open WebUI file uploads to local disk and appends their absolute paths to
the user message before forwarding to `nat serve`.

Modules:
  config.py  -> FileAdapterConfig (AdapterConfig + upload settings)
  files.py   -> extract uploads from the request, save them, append abs paths to the text
  app.py     -> create_app(config) + CLI main()
"""

from filebridge.app import create_app
from filebridge.app import main
from filebridge.config import FileAdapterConfig

__all__ = ["FileAdapterConfig", "create_app", "main"]
