# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classify & parse NAT `/chat/stream` `intermediate_data:` steps into the 3 known kinds.

A verbose `react_agent` behind `nat serve` emits 3 kinds of step frame:

  KIỂU 1 — THINKING : name == "<model_name>"; the model's `**Output:**` reasoning (Thought/Action)
                      WITHOUT "Final Answer:". Pure `**Input:**` dumps (the system prompt) are noise.
  KIỂU 2 — TOOL     : name starts with "Tool: <tool_name>"; `**Input:**` carries the args. The
                      result may be inline (`**Output:**`) OR arrive as a HumanMessage in the next
                      LLM dump (current config) — see `last_observation`.
  KIỂU 3 — FINAL    : name == "<model_name>"; the `**Output:**` section contains "Final Answer:".
                      The text after it is the answer, streamed as cumulative snapshots.

⚠️ The system prompt embedded in every `**Input:**` dump literally contains the line
   'Final Answer: the final answer to the original input question'. So "Final Answer:" must be
   matched ONLY inside the `**Output:**` section, never over the whole payload, or every frame
   would be misclassified and the Thinking box would stay empty.

All format markers come from `AdapterConfig`, so a different agent's wire format is a config change.
"""

from __future__ import annotations

import html
import re
from typing import Literal
from typing import NamedTuple

from natbridge.config import AdapterConfig

FrameKind = Literal["tool", "final", "thinking", "skip"]


class ToolCall(NamedTuple):
    """A parsed KIỂU 2 step. `result` is None when the frame carries only the args (Input)."""
    name: str
    arguments: str
    result: str | None


class NatFrameParser:
    """Stateless classifier/parser for NAT step frames, driven by `AdapterConfig` markers."""

    def __init__(self, config: AdapterConfig) -> None:
        self._cfg = config
        # Matches a fenced block after **Input:** / **Output:** (``` optional language ```).
        self._input_re = re.compile(re.escape(config.input_marker) + r"\s*```[a-zA-Z]*\n?(.*?)```",
                                    re.DOTALL)
        self._output_re = re.compile(re.escape(config.output_marker) + r"\s*```[a-zA-Z]*\n?(.*?)```",
                                     re.DOTALL)
        # A HumanMessage(content=...) entry in an LLM input dump (used to recover a tool result).
        self._human_re = re.compile(
            r"HumanMessage\(content=(['\"])(.*?)\1,\s*(?:additional_kwargs|response_metadata)=",
            re.DOTALL)

    # -- low-level helpers --------------------------------------------------

    @staticmethod
    def _payload(step: dict) -> str:
        return html.unescape(str(step.get("payload") or ""))

    def output_section(self, step: dict) -> str | None:
        """The model's real output (text after `**Output:**`), or None for input-only dumps."""
        payload = self._payload(step)
        marker = self._cfg.output_marker
        if marker not in payload:
            return None
        return payload.split(marker, 1)[1].strip() or None

    def is_noise(self, name: str) -> bool:
        """Internal wrapper frames that aren't one of the 3 user-facing kinds -> skip."""
        return any(m in name for m in self._cfg.noise_name_markers)

    def is_tool(self, name: str) -> bool:
        return name.startswith(self._cfg.tool_name_prefix)

    # -- classification -----------------------------------------------------

    def classify(self, step: dict) -> FrameKind:
        name = step.get("name") or ""
        if self.is_tool(name):
            return "tool"
        if self.is_noise(name):
            return "skip"
        if self.final_answer_text(step) is not None:
            return "final"
        return "thinking"

    def thinking_text(self, step: dict) -> str | None:
        """KIỂU 1 content: the `**Output:**` reasoning, or None (input-only / final / tool / noise)."""
        name = step.get("name") or ""
        if self.is_tool(name) or self.is_noise(name):
            return None
        out = self.output_section(step)
        if out is None or self._cfg.final_answer_marker in out:
            return None
        return out

    def final_answer_text(self, step: dict) -> str | None:
        """KIỂU 3: the (cumulative) text AFTER 'Final Answer:' inside `**Output:**`, else None."""
        name = step.get("name") or ""
        if self.is_tool(name) or self.is_noise(name):
            return None
        out = self.output_section(step)
        marker = self._cfg.final_answer_marker
        if out and marker in out:
            return out.split(marker, 1)[1].strip() or None
        return None

    def parse_tool(self, step: dict) -> ToolCall:
        """KIỂU 2: (tool_name, args, result_or_None). Always returns a ToolCall (never None) so the
        tool box can show as soon as the call starts ("tool đang chạy"); the result is attached
        later via `last_observation` when it isn't inline."""
        raw = step.get("name") or ""
        name = raw.split(":", 1)[1].strip() if ":" in raw else raw
        payload = self._payload(step)
        mi = self._input_re.search(payload)
        args = (mi.group(1).strip() if mi else "") or "{}"
        if "   (" in args:                                  # strip trailing prose after the JSON
            args = args.split("   (")[0].strip()
        result: str | None = None
        if self._cfg.output_marker in payload:
            mo = self._output_re.search(payload)
            result = (mo.group(1).strip() if mo
                      else payload.split(self._cfg.output_marker, 1)[1].strip()) or None
        return ToolCall(name, args, result)

    def last_observation(self, step: dict, user_text: str) -> str | None:
        """Best-effort tool RESULT: in the current config the result comes back as the LAST
        HumanMessage (not the user's question) in the next LLM input dump."""
        payload = self._payload(step)
        q = (user_text or "").strip()
        for _, content in reversed(self._human_re.findall(payload)):
            if "Message(content=" in content:               # regex bridged two messages -> skip
                continue
            if q and q in content:                           # the user's question wrapper
                continue
            text = content.replace("\\n", "\n").strip()
            if text:
                return text
        return None
