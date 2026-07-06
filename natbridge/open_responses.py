# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conformant Open Responses building blocks (SSE events + the full response resource object).

Field shapes follow the authoritative Zod schemas in the spec repo
(`openresponses/src/generated/kubb/zod/*`). Two facts a minimal implementation gets wrong and this
module encodes:

  1. The `response` object carried by created/in_progress/completed/failed MUST contain all 31
     required keys (most nullable). `_response()` always emits the full object.
  2. A `function_call` item + a matching `function_call_output` item (paired by `call_id`) render
     in Open WebUI as a DISPLAY-ONLY "Tool Executed" box — the client does NOT execute it. This is
     how we surface NAT's already-run tools without triggering re-execution.

Open WebUI renders `output[]` IN ORDER: `reasoning` -> "Thinking" box, `function_call`(+output) ->
"Tool Executed" box, `message` -> the answer. So the high-level helpers keep a strict order:
open/stream/close the reasoning item BEFORE any tool box, and the message item LAST.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any


def extract_input_text(input_field: Any) -> str:
    """Open Responses `input` (string OR item array) -> the last user message's text."""
    if isinstance(input_field, str):
        return input_field
    if not isinstance(input_field, list):
        return ""
    last = ""
    for item in input_field:
        if not isinstance(item, dict) or item.get("type", "message") != "message":
            continue
        if item.get("role") not in (None, "user"):
            continue
        content = item.get("content")
        if isinstance(content, str):
            last = content
        elif isinstance(content, list):
            parts = [p.get("text", "") for p in content
                     if isinstance(p, dict) and p.get("type") in ("input_text", "output_text", "text")]
            if parts:
                last = "\n".join(parts)
    return last


class OpenResponsesEmitter:
    """Emits a spec-conformant Open Responses SSE stream (framing + ordering + full objects)."""

    def __init__(self, response_id: str, model: str, *, tools: list | None = None,
                 instructions: str | None = None, previous_response_id: str | None = None,
                 step_payload_max: int = 1000, max_steps: int = 50, reasoning_total_max: int = 8000):
        self._id = response_id
        self._model = model
        self._tools = tools or []
        self._instructions = instructions
        self._previous_response_id = previous_response_id
        self._seq = 0
        self._created_at = int(time.time())
        self._output: list[dict] = []   # finalized output[] items
        self._error: dict | None = None
        self._step_payload_max = step_payload_max
        self._max_steps = max_steps
        self._reasoning_total_max = reasoning_total_max
        # State for the streaming helpers (one accumulating reasoning item, then one message item).
        self._reasoning_item: dict | None = None
        self._reasoning_idx: int | None = None
        self._reasoning_total = 0
        self._msg_item: dict | None = None
        self._msg_idx: int | None = None

    # -- framing ------------------------------------------------------------

    def _frame(self, event_type: str, payload: dict) -> str:
        body = {"type": event_type, "sequence_number": self._seq, **payload}
        self._seq += 1
        return f"event: {event_type}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"

    def _response(self, status: str) -> dict:
        """The full response resource object — ALL 31 required keys (responseResourceSchema)."""
        completed = status == "completed"
        return {
            "id": self._id, "object": "response", "created_at": self._created_at,
            "completed_at": self._created_at if completed else None, "status": status,
            "incomplete_details": None, "model": self._model,
            "previous_response_id": self._previous_response_id, "instructions": self._instructions,
            "output": list(self._output), "error": self._error, "tools": self._tools,
            "tool_choice": "auto", "truncation": "disabled", "parallel_tool_calls": True,
            "text": {"format": {"type": "text"}}, "top_p": None, "presence_penalty": None,
            "frequency_penalty": None, "top_logprobs": None, "temperature": None, "reasoning": None,
            "usage": None, "max_output_tokens": None, "max_tool_calls": None, "store": False,
            "background": False, "service_tier": "auto", "metadata": {}, "safety_identifier": None,
            "prompt_cache_key": None,
        }

    # -- response-level envelope --------------------------------------------

    def created(self) -> str:
        return self._frame("response.created", {"response": self._response("in_progress")})

    def in_progress(self) -> str:
        return self._frame("response.in_progress", {"response": self._response("in_progress")})

    def completed(self) -> str:
        return self._frame("response.completed", {"response": self._response("completed")})

    def failed(self, message: str, *, err_type: str = "model_error", code: str = "model_error") -> list[str]:
        self._error = {"code": code, "message": message}
        payload = {"message": message, "type": err_type, "param": None, "code": code}
        return [
            self._frame("error", {"error": payload}),
            self._frame("response.failed", {"response": self._response("failed")}),
        ]

    @staticmethod
    def done() -> str:
        return "data: [DONE]\n\n"  # terminal sentinel, not a JSON event

    def final_object(self) -> dict:
        """Non-streaming: the completed response object."""
        return self._response("completed")

    # -- `reasoning` item (Open WebUI "Thinking" box) -----------------------

    def open_reasoning(self, item_id: str) -> str:
        self._reasoning_idx = len(self._output)
        self._reasoning_item = {"id": item_id, "type": "reasoning", "status": "in_progress",
                                "summary": [{"type": "summary_text", "text": ""}]}
        self._output.append(self._reasoning_item)
        added = {"id": item_id, "type": "reasoning", "status": "in_progress", "summary": []}
        return self._frame("response.output_item.added", {"output_index": self._reasoning_idx, "item": added})

    def reasoning_delta(self, text: str) -> str | None:
        """Append text to the Thinking box (bounded so the terminal event stays small)."""
        if self._reasoning_item is None:
            return None
        if self._reasoning_total_max and self._reasoning_total >= self._reasoning_total_max:
            return None
        if self._reasoning_total_max:
            text = text[: self._reasoning_total_max - self._reasoning_total]
        if not text:
            return None
        self._reasoning_total += len(text)
        self._reasoning_item["summary"][0]["text"] += text
        return self._frame("response.reasoning_summary_text.delta", {
            "item_id": self._reasoning_item["id"], "output_index": self._reasoning_idx,
            "summary_index": 0, "delta": text})

    def close_reasoning(self) -> str | None:
        if self._reasoning_item is None:
            return None
        self._reasoning_item["status"] = "completed"
        return self._frame("response.output_item.done",
                           {"output_index": self._reasoning_idx, "item": self._reasoning_item})

    # -- `message` item (the clear final answer) ----------------------------

    def open_message(self, item_id: str) -> list[str]:
        self._msg_idx = len(self._output)
        self._msg_item = {"id": item_id, "type": "message", "role": "assistant", "status": "in_progress",
                          "content": [{"type": "output_text", "text": "", "annotations": []}]}
        self._output.append(self._msg_item)
        added = {"id": item_id, "type": "message", "role": "assistant", "status": "in_progress", "content": []}
        return [
            self._frame("response.output_item.added", {"output_index": self._msg_idx, "item": added}),
            self._frame("response.content_part.added", {
                "item_id": item_id, "output_index": self._msg_idx, "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []}}),
        ]

    def message_delta(self, delta: str) -> str:
        self._msg_item["content"][0]["text"] += delta
        return self._frame("response.output_text.delta", {
            "item_id": self._msg_item["id"], "output_index": self._msg_idx, "content_index": 0, "delta": delta})

    def close_message(self) -> list[str]:
        text = self._msg_item["content"][0]["text"]
        self._msg_item["status"] = "completed"
        return [
            self._frame("response.output_text.done", {
                "item_id": self._msg_item["id"], "output_index": self._msg_idx, "content_index": 0, "text": text}),
            self._frame("response.content_part.done", {
                "item_id": self._msg_item["id"], "output_index": self._msg_idx, "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []}}),
            self._frame("response.output_item.done", {"output_index": self._msg_idx, "item": self._msg_item}),
        ]

    # -- tool items (Open WebUI "Tool Executed" box; DISPLAY ONLY) -----------

    def tool_call(self, name: str, arguments: str, call_id: str) -> list[str]:
        """Emit ONLY the `function_call` item (the call). Lets the tool box show immediately;
        the result (`function_call_output`, same call_id) can follow later via `tool_output`."""
        if self._step_payload_max:
            arguments = arguments[: self._step_payload_max]
        item = {"id": f"fc_{uuid.uuid4().hex}", "type": "function_call", "status": "completed",
                "call_id": call_id, "name": name, "arguments": arguments}
        idx = len(self._output)
        self._output.append(item)
        return [
            self._frame("response.output_item.added", {"output_index": idx, "item": item}),
            self._frame("response.output_item.done", {"output_index": idx, "item": item}),
        ]

    def tool_output(self, call_id: str, result: str) -> list[str]:
        """Emit ONLY the `function_call_output` item (the result), paired to a prior `tool_call`."""
        if self._step_payload_max:
            result = result[: self._step_payload_max]
        item = {"id": f"fco_{uuid.uuid4().hex}", "type": "function_call_output", "status": "completed",
                "call_id": call_id, "output": [{"type": "output_text", "text": result}]}
        idx = len(self._output)
        self._output.append(item)
        return [
            self._frame("response.output_item.added", {"output_index": idx, "item": item}),
            self._frame("response.output_item.done", {"output_index": idx, "item": item}),
        ]

    def tool_call_items(self, name: str, arguments: str, result: str | None,
                        call_id: str | None = None) -> list[str]:
        """Convenience: function_call (+ function_call_output) in one go (when result known)."""
        call_id = call_id or f"call_{uuid.uuid4().hex[:12]}"
        frames = self.tool_call(name, arguments, call_id)
        if result:
            frames += self.tool_output(call_id, result)
        return frames
