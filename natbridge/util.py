# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small shared helpers used by both translators."""

from __future__ import annotations

import uuid


def format_block(text: str, n: int, max_chars: int) -> str:
    """Format one reasoning/thinking block for display (numbered + truncated)."""
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + " …[truncated]"
    return f"**Step {n}**\n{text}\n\n"


class ToolTracker:
    """Tracks tool calls so each `(name, args)` is surfaced once, with its result attached later.

    A NAT tool result often arrives AFTER the call (in the next LLM dump), so the call and its
    result are emitted at different times. This tracker holds the dedup state; the caller does the
    actual (emitter-specific) emission based on the booleans/ids returned here.
    """

    def __init__(self) -> None:
        self._seen: dict[tuple[str, str], dict] = {}   # (name, args) -> {"call_id", "result": bool}
        self.pending: tuple[str, str] | None = None    # a call still awaiting its result

    def on_call(self, name: str, args: str) -> tuple[str, bool]:
        """Register a call. Returns (call_id, is_new); emit the function_call only when is_new."""
        key = (name, args)
        info = self._seen.get(key)
        if info is None:
            call_id = f"call_{uuid.uuid4().hex[:12]}"
            self._seen[key] = {"call_id": call_id, "result": False}
            self.pending = key
            return call_id, True
        return info["call_id"], False

    def on_result(self, name: str, args: str) -> str | None:
        """Attach a result to a prior call. Returns the call_id if newly attached, else None."""
        key = (name, args)
        info = self._seen.get(key)
        if info is None or info["result"]:
            return None
        info["result"] = True
        if self.pending == key:
            self.pending = None
        return info["call_id"]
