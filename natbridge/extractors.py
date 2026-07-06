# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""De-duplicate NAT's CUMULATIVE snapshots (grow token-by-token, reset per LLM call).

NAT re-sends the whole reasoning/answer so far on every token. These two small state machines turn
that firehose into clean, incremental output. They operate on already-extracted TEXT (see
`natbridge.frames`), so they're independent of the frame format.
"""

from __future__ import annotations


class ThinkingExtractor:
    """Collapse cumulative THINKING snapshots into one block per completed snapshot."""

    def __init__(self) -> None:
        self._cur = ""

    def feed(self, snapshot: str | None) -> str | None:
        """Feed a thinking snapshot; return a COMPLETED block when a new one starts, else None."""
        if snapshot is None:
            return None
        if snapshot.startswith(self._cur):              # same block still growing (or identical)
            self._cur = snapshot
            return None
        completed, self._cur = self._cur, snapshot      # a new block started -> previous is complete
        return completed or None

    def flush(self) -> str | None:
        """Return the final pending block."""
        cur, self._cur = self._cur, ""
        return cur or None


class FinalAnswerStreamer:
    """Stream cumulative FINAL-answer snapshots as incremental deltas (token-by-token)."""

    def __init__(self) -> None:
        self._cur = ""

    def feed(self, full: str | None) -> str | None:
        """Feed the latest cumulative answer text; return only the NEW tail, else None."""
        if not full or full == self._cur:
            return None
        if full.startswith(self._cur):                  # grew -> emit only the appended part
            delta, self._cur = full[len(self._cur):], full
            return delta or None
        self._cur = full                                # rare reset -> emit the new full text
        return full
