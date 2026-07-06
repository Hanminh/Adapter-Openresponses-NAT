# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""natbridge — a clean, modular Open Responses adapter in front of `nat serve /chat/stream`.

Module map:
  config.py               — AdapterConfig (all runtime + frame-format knobs).
  frames.py               — NatFrameParser: classify/parse the 3 NAT step kinds (thinking/tool/final).
  extractors.py           — ThinkingExtractor / FinalAnswerStreamer (cumulative-snapshot dedup).
  nat_client.py           — call_chat_stream: async client for nat serve.
  open_responses.py       — OpenResponsesEmitter: spec-conformant SSE + 31-key response object.
  util.py                 — format_block, ToolTracker (tool dedup state).
  responses_translator.py — /v1/responses path (the special Open WebUI boxes).
  openai_translator.py    — /v1/chat/completions path (OpenAI default mode).
  app.py                  — FastAPI app factory + CLI (`python -m natbridge`).
"""

from natbridge.app import create_app
from natbridge.app import main
from natbridge.config import AdapterConfig

__all__ = ["AdapterConfig", "create_app", "main"]
