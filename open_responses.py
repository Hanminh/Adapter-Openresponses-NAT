# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conformant Open Responses building blocks (events + the full response resource object).

Field shapes are taken from the authoritative Zod schemas in the spec repo
(/home/minhth11/Projects/API_UI/openresponses/src/generated/kubb/zod/*):

  - responseResourceSchema.ts            -> the response object (ALL 31 keys are required)
  - response*StreamingEventSchema.ts     -> the event envelopes
  - messageSchema.ts / functionCallSchema.ts / outputTextContentSchema.ts  -> item/content shapes
  - errorPayloadSchema.ts                -> {message, type, param, code}

Two facts this module encodes that a minimal implementation gets wrong:
  1. The `response` carried by created/in_progress/completed/failed MUST contain all 31
     required keys (most are nullable). We always emit the full object.
  2. NAT runs tools INTERNALLY. Per the spec, internally-hosted tools are surfaced as a
     vendor-namespaced *item* "receipt" (cf. the spec's own `openai:web_search_call`
     example). We therefore emit each NAT intermediate step as a `nat:intermediate_step`
     item inside `output[]`, framed by output_item.added/done — not as a `function_call`
     (which would wrongly ask the client to execute it).
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
        self._output: list[dict] = []   # finalized output items
        self._error: dict | None = None
        # Size bounds: keep every SSE `data:` line (esp. the terminal response.completed, which
        # embeds output[]) well under client line-length limits (e.g. 128KB). A single huge step
        # payload, or hundreds of step items, would otherwise blow past it.
        self._step_payload_max = step_payload_max
        self._max_steps = max_steps
        self._step_count = 0
        self._reasoning_total_max = reasoning_total_max
        # State for the high-level streaming helpers (reasoning Thinking box + message item).
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
        """The full response resource object — ALL required keys (responseResourceSchema)."""
        completed = status == "completed"
        return {
            "id": self._id,
            "object": "response",
            "created_at": self._created_at,
            "completed_at": self._created_at if completed else None,
            "status": status,
            "incomplete_details": None,
            "model": self._model,
            "previous_response_id": self._previous_response_id,
            "instructions": self._instructions,
            "output": list(self._output),
            "error": self._error,
            "tools": self._tools,
            "tool_choice": "auto",
            "truncation": "disabled",
            "parallel_tool_calls": True,
            "text": {"format": {"type": "text"}},
            "top_p": None,
            "presence_penalty": None,
            "frequency_penalty": None,
            "top_logprobs": None,
            "temperature": None,
            "reasoning": None,
            "usage": None,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "store": False,
            "background": False,
            "service_tier": "auto",
            "metadata": {},
            "safety_identifier": None,
            "prompt_cache_key": None,
        }

    # -- response-level envelope --------------------------------------------

    def created(self) -> str:
        return self._frame("response.created", {"response": self._response("in_progress")})

    def in_progress(self) -> str:
        return self._frame("response.in_progress", {"response": self._response("in_progress")})

    def completed(self) -> str:
        return self._frame("response.completed", {"response": self._response("completed")})

    def incomplete(self) -> str:
        return self._frame("response.incomplete", {"response": self._response("incomplete")})

    def failed(self, message: str, *, err_type: str = "model_error", code: str = "model_error") -> list[str]:
        payload = {"message": message, "type": err_type, "param": None, "code": code}
        self._error = {"code": code, "message": message}
        return [
            self._frame("error", {"error": payload}),
            self._frame("response.failed", {"response": self._response("failed")}),
        ]

    @staticmethod
    def done() -> str:
        return "data: [DONE]\n\n"  # terminal sentinel, not a JSON event

    # -- assistant `message` item (streamable output_text) ------------------

    def message_added(self, output_index: int, item_id: str) -> str:
        item = {"id": item_id, "type": "message", "role": "assistant", "status": "in_progress", "content": []}
        return self._frame("response.output_item.added", {"output_index": output_index, "item": item})

    def text_part_added(self, output_index: int, item_id: str, content_index: int) -> str:
        return self._frame("response.content_part.added", {
            "item_id": item_id, "output_index": output_index, "content_index": content_index,
            "part": {"type": "output_text", "text": "", "annotations": []},
        })

    def text_delta(self, output_index: int, item_id: str, content_index: int, delta: str) -> str:
        return self._frame("response.output_text.delta", {
            "item_id": item_id, "output_index": output_index, "content_index": content_index, "delta": delta,
        })

    def text_done(self, output_index: int, item_id: str, content_index: int, text: str) -> str:
        return self._frame("response.output_text.done", {
            "item_id": item_id, "output_index": output_index, "content_index": content_index, "text": text,
        })

    def text_part_done(self, output_index: int, item_id: str, content_index: int, text: str) -> str:
        return self._frame("response.content_part.done", {
            "item_id": item_id, "output_index": output_index, "content_index": content_index,
            "part": {"type": "output_text", "text": text, "annotations": []},
        })

    def message_done(self, output_index: int, item_id: str, text: str) -> str:
        item = {"id": item_id, "type": "message", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}]}
        self._output.append(item)
        return self._frame("response.output_item.done", {"output_index": output_index, "item": item})

    # -- `function_call` item (a tool the model wants the client to run) ----

    def function_call(self, output_index: int, name: str, arguments: str, call_id: str) -> list[str]:
        """Full function_call lifecycle (use for client-executed / externally-hosted tools)."""
        item_id = f"fc_{uuid.uuid4().hex}"
        added = {"id": item_id, "type": "function_call", "name": name, "call_id": call_id,
                 "arguments": "", "status": "in_progress"}
        out = [self._frame("response.output_item.added", {"output_index": output_index, "item": added})]
        out.append(self._frame("response.function_call_arguments.delta",
                               {"item_id": item_id, "output_index": output_index, "delta": arguments}))
        out.append(self._frame("response.function_call_arguments.done",
                               {"item_id": item_id, "output_index": output_index, "arguments": arguments}))
        done = {**added, "status": "completed", "arguments": arguments}
        self._output.append(done)
        out.append(self._frame("response.output_item.done", {"output_index": output_index, "item": done}))
        return out

    # -- `nat:intermediate_step` item (internally-hosted tool/step receipt) -

    def _truncate(self, text: Any) -> Any:
        if not isinstance(text, str) or self._step_payload_max <= 0 or len(text) <= self._step_payload_max:
            return text
        return text[: self._step_payload_max] + "…[truncated]"

    def intermediate_step_item(self, step: dict) -> list[str]:
        """A vendor-namespaced item receipt for a NAT step (cf. spec `openai:web_search_call`).

        The output_index is computed from the current output[] length so it always matches the
        item's real position (capped/skipped steps never leave a gap). Framed by
        output_item.added/done like any item; clients that don't know the type ignore it but
        still reconstruct the response. The payload is truncated and the number of step items is
        capped so neither these events nor the terminal response.completed (which embeds
        output[]) can exceed client SSE line-length limits.
        """
        if self._max_steps >= 0 and self._step_count >= self._max_steps:
            return []  # cap reached: stop surfacing further step items (keeps output[] bounded)
        self._step_count += 1
        output_index = len(self._output)
        item = {
            "id": step.get("id") or f"nat_{uuid.uuid4().hex}",
            "type": "nat:intermediate_step",
            "status": "completed",
            "name": step.get("name"),
            "step_type": step.get("type"),
            "parent_id": step.get("parent_id"),
            "payload": self._truncate(step.get("payload")),
        }
        self._output.append(item)
        return [
            self._frame("response.output_item.added", {"output_index": output_index, "item": item}),
            self._frame("response.output_item.done", {"output_index": output_index, "item": item}),
        ]

    # -- high-level streaming helpers (Open WebUI-friendly) -----------------
    #
    # Open WebUI's Responses-API renderer (utils/middleware.py) renders output[] items as
    # special blocks: a `reasoning` item -> a collapsible "Thinking" box; a `message` item's
    # output_text -> the clear final answer; a `function_call` item -> a "Tool Executed" box.
    # Items render in output[] ORDER, so the reasoning item MUST precede the message item.
    # These helpers manage one accumulating reasoning item, then one message item.

    def open_reasoning(self, item_id: str) -> str:
        """Start the single `reasoning` item (the Thinking box). Call before any message text."""
        self._reasoning_idx = len(self._output)
        self._reasoning_item = {
            "id": item_id, "type": "reasoning", "status": "in_progress",
            "summary": [{"type": "summary_text", "text": ""}],
        }
        self._output.append(self._reasoning_item)
        added = {"id": item_id, "type": "reasoning", "status": "in_progress", "summary": []}
        return self._frame("response.output_item.added", {"output_index": self._reasoning_idx, "item": added})

    def reasoning_delta(self, text: str) -> str | None:
        """Append text to the Thinking box (bounded total so the terminal event stays small)."""
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
            "summary_index": 0, "delta": text,
        })

    def close_reasoning(self) -> str | None:
        """Finalize the Thinking box (renders as 'Thought for…' collapsed once the answer starts)."""
        if self._reasoning_item is None:
            return None
        self._reasoning_item["status"] = "completed"
        return self._frame("response.output_item.done",
                           {"output_index": self._reasoning_idx, "item": self._reasoning_item})

    def open_message(self, item_id: str) -> list[str]:
        """Start the assistant `message` item (the clear final answer)."""
        self._msg_idx = len(self._output)
        self._msg_item = {
            "id": item_id, "type": "message", "role": "assistant", "status": "in_progress",
            "content": [{"type": "output_text", "text": "", "annotations": []}],
        }
        self._output.append(self._msg_item)
        added = {"id": item_id, "type": "message", "role": "assistant", "status": "in_progress", "content": []}
        return [
            self._frame("response.output_item.added", {"output_index": self._msg_idx, "item": added}),
            self._frame("response.content_part.added", {
                "item_id": item_id, "output_index": self._msg_idx, "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            }),
        ]

    def message_delta(self, delta: str) -> str:
        """Append a token to the final answer."""
        self._msg_item["content"][0]["text"] += delta # type: ignore
        return self._frame("response.output_text.delta", {
            "item_id": self._msg_item["id"], "output_index": self._msg_idx, "content_index": 0, "delta": delta, # type: ignore
        })

    def close_message(self) -> list[str]:
        """Finalize the message item."""
        text = self._msg_item["content"][0]["text"] # type: ignore
        self._msg_item["status"] = "completed" # type: ignore
        return [
            self._frame("response.output_text.done", {
                "item_id": self._msg_item["id"], "output_index": self._msg_idx, "content_index": 0, "text": text}), # type: ignore
            self._frame("response.content_part.done", {
                "item_id": self._msg_item["id"], "output_index": self._msg_idx, "content_index": 0, # type: ignore
                "part": {"type": "output_text", "text": text, "annotations": []}}),
            self._frame("response.output_item.done", {"output_index": self._msg_idx, "item": self._msg_item}),
        ]

    # -- tool call (renders as Open WebUI's "Tool Executed" box) ------------
    #
    # A `function_call` item + a matching `function_call_output` item (paired by call_id) render
    # in Open WebUI as `<details type="tool_calls" done="true">Tool Executed … </details>`. This
    # is DISPLAY ONLY (the output is already present), so Open WebUI does NOT try to execute it.

    def tool_call(self, name: str, arguments: str, call_id: str) -> list[str]:
        """Emit ONLY the `function_call` item (the call itself / "tool running").

        Emitting the call as soon as the `Tool:` frame arrives lets Open WebUI show the tool box
        immediately; the matching result (`function_call_output`, paired by call_id) can follow
        later via `tool_output` once it's known. Open WebUI pairs them regardless of the gap.
        """
        if self._step_payload_max:
            arguments = arguments[: self._step_payload_max]
        fc_item = {"id": f"fc_{uuid.uuid4().hex}", "type": "function_call", "status": "completed",
                   "call_id": call_id, "name": name, "arguments": arguments}
        idx = len(self._output)
        self._output.append(fc_item)
        return [
            self._frame("response.output_item.added", {"output_index": idx, "item": fc_item}),
            self._frame("response.output_item.done", {"output_index": idx, "item": fc_item}),
        ]

    def tool_output(self, call_id: str, result: str) -> list[str]:
        """Emit ONLY the `function_call_output` item (the result), paired to a prior `tool_call`."""
        if self._step_payload_max:
            result = result[: self._step_payload_max]
        out_item = {"id": f"fco_{uuid.uuid4().hex}", "type": "function_call_output",
                    "status": "completed", "call_id": call_id,
                    "output": [{"type": "output_text", "text": result}]}
        idx = len(self._output)
        self._output.append(out_item)
        return [
            self._frame("response.output_item.added", {"output_index": idx, "item": out_item}),
            self._frame("response.output_item.done", {"output_index": idx, "item": out_item}),
        ]

    def tool_call_items(self, name: str, arguments: str, result: str | None, call_id: str | None = None) -> list[str]:
        """Emit a completed function_call item (+ function_call_output) in one go."""
        call_id = call_id or f"call_{uuid.uuid4().hex[:12]}"
        frames = self.tool_call(name, arguments, call_id)
        if result:
            frames += self.tool_output(call_id, result)
        return frames

    # -- non-streaming final object -----------------------------------------

    def final_object(self) -> dict:
        return self._response("completed")
