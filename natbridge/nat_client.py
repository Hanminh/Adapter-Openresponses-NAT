# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin async client for a running `nat serve` /chat/stream endpoint."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from natbridge.config import AdapterConfig

# Yielded tuples: ("step", ResponseIntermediateStep) | ("token", ChatResponseChunk).
NatEvent = tuple[str, dict]


async def call_chat_stream(config: AdapterConfig, user_text: str, *, user_id: str | None,
                           conversation_id: str | None) -> AsyncIterator[NatEvent]:
    """POST to nat serve /chat/stream and yield decoded SSE frames.

    NAT emits two line kinds:
      - `intermediate_data: {ResponseIntermediateStep}` -> ("step", obj)
      - `data: {ChatResponseChunk}` / `data: [DONE]`    -> ("token", obj) / end
    Headers `x-user-id` / `conversation-id` are forwarded so memory & multi-turn keep working.
    """
    body = {"messages": [{"role": "user", "content": user_text}], "stream": True}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if user_id:
        headers["x-user-id"] = user_id
    if conversation_id:
        headers["conversation-id"] = conversation_id

    async with httpx.AsyncClient(timeout=config.http_timeout) as client:
        async with client.stream("POST", config.nat_chat_stream_url, json=body, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("intermediate_data:"):
                    raw = line[len("intermediate_data:"):].strip()
                    try:
                        yield "step", json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                elif line.startswith("data:"):
                    raw = line[len("data:"):].strip()
                    if raw == "[DONE]":
                        return
                    try:
                        yield "token", json.loads(raw)
                    except json.JSONDecodeError:
                        continue
