# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Open Responses-compliant streaming server for NeMo Agent Toolkit workflows.

Exposes `POST /v1/responses` (Open Responses standard, see
/home/minhth11/Projects/API_UI/openresponses/instruction/streaming-api-guide.md) and runs a
NAT workflow behind it, translating NAT's output into Open Responses *semantic events*:

    response.created -> response.in_progress
      -> response.output_item.added            (assistant `message` item)
        -> response.content_part.added         (output_text part, text="")
          -> response.output_text.delta * N    (streamed answer chunks)
        -> response.output_text.done
      -> response.content_part.done
      -> response.output_item.done
      [-> function_call item(s) for tools the agent used]   (optional)
    -> response.completed
    data: [DONE]

This is the concrete realization of the bridge sketched in
`my_instruction/OPEN_RESPONSES_VS_NAT.md` §6: NAT runs the agentic loop; this server
*serializes* it as Open Responses items + events.

Run:

    python my_example/openresponses/adapter.py \
        --nat-url http://localhost:8000/chat/stream \
            --port 8001 \
                --max-steps 50 \
                    --step-payload-max 1000
                    
Then (Open Responses streaming request):
    curl -N http://localhost:8001/v1/responses -H 'Content-Type: application/json' \
      -d '{"model":"nat","stream":true,
           "input":[{"type":"message","role":"user",
                     "content":[{"type":"input_text","text":"Hello"}]}]}'
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse

from nat.builder.context import Context
from nat.data_models.intermediate_step import IntermediateStep
from nat.runtime.loader import load_workflow

logger = logging.getLogger("openresponses-nat")

# Set by main() before the app serves.
CONFIG_FILE: str = ""

# ---------------------------------------------------------------------------
# Open Responses request parsing
# ---------------------------------------------------------------------------


def _extract_input_text(input_field: Any) -> str:
    """Turn an Open Responses `input` (string OR item array) into the workflow's text input.

    Per the spec, `input` is either a plain string (a user message shorthand) or an ordered
    list of items. For a message item, `content` is a string or a list of content parts
    ({type: input_text|output_text|text, text}). We use the LAST user message's text as the
    workflow input (NAT agents are single-turn; use a memory middleware + previous_response_id
    for continuity).
    """
    if isinstance(input_field, str):
        return input_field

    if not isinstance(input_field, list):
        return ""

    last_user_text = ""
    for item in input_field:
        if not isinstance(item, dict) or item.get("type", "message") != "message":
            continue
        if item.get("role") not in (None, "user"):
            continue
        content = item.get("content")
        if isinstance(content, str):
            last_user_text = content
        elif isinstance(content, list):
            parts = [
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") in ("input_text", "output_text", "text")
            ]
            if parts:
                last_user_text = "\n".join(parts)
    return last_user_text


# ---------------------------------------------------------------------------
# Open Responses event emitter (handles framing, sequence_number, indices)
# ---------------------------------------------------------------------------


class ResponseStream:
    """Builds the Open Responses SSE event sequence with correct framing and ordering."""

    def __init__(self, response_id: str, model: str):
        self._response_id = response_id
        self._model = model
        self._seq = 0
        self._created_at = int(time.time())
        self._output: list[dict] = []  # finalized items, for the terminal response object

    # -- low-level framing --------------------------------------------------

    def _frame(self, event_type: str, payload: dict) -> str:
        """One SSE frame: `event: <type>\\n data: <json>\\n\\n`. event MUST match body.type."""
        body = {"type": event_type, "sequence_number": self._seq, **payload}
        self._seq += 1
        return f"event: {event_type}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"

    def _response_obj(self, status: str) -> dict:
        return {
            "id": self._response_id,
            "object": "response",
            "created_at": self._created_at,
            "model": self._model,
            "status": status,
            "output": list(self._output),
        }

    # -- envelope -----------------------------------------------------------

    def created(self) -> str:
        return self._frame("response.created", {"response": self._response_obj("in_progress")})

    def in_progress(self) -> str:
        return self._frame("response.in_progress", {"response": self._response_obj("in_progress")})

    def completed(self) -> str:
        return self._frame("response.completed", {"response": self._response_obj("completed")})

    def failed(self, message: str, err_type: str = "model_error", code: str = "model_error") -> list[str]:
        # On a mid-stream error: emit `error` then `response.failed` (spec §8).
        err = {"message": message, "type": err_type, "param": None, "code": code}
        out = [self._frame("error", {"error": err})]
        resp = self._response_obj("failed")
        resp["error"] = {"code": code, "message": message}
        out.append(self._frame("response.failed", {"response": resp}))
        return out

    @staticmethod
    def done() -> str:
        # Terminal sentinel — NOT a JSON event.
        return "data: [DONE]\n\n"

    def custom(self, event_type: str, payload: dict) -> str:
        """A vendor-namespaced extension event (spec §8 — MUST be `vendor:...`).

        Compliant clients ignore unknown namespaced events without breaking reconstruction,
        so this safely carries NAT intermediate steps inside the stream.
        """
        if ":" not in event_type:
            raise ValueError("custom event types MUST be vendor-namespaced, e.g. 'nat:foo'")
        return self._frame(event_type, payload)

    # -- message item (assistant text) -------------------------------------

    def message_added(self, output_index: int, item_id: str) -> str:
        # All non-nullable fields present; zero values where unknown.
        item = {"id": item_id, "type": "message", "role": "assistant", "status": "in_progress", "content": []}
        return self._frame("response.output_item.added", {"output_index": output_index, "item": item})

    def text_part_added(self, output_index: int, item_id: str, content_index: int) -> str:
        return self._frame(
            "response.content_part.added",
            {
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        )

    def text_delta(self, output_index: int, item_id: str, content_index: int, delta: str) -> str:
        return self._frame(
            "response.output_text.delta",
            {"item_id": item_id, "output_index": output_index, "content_index": content_index, "delta": delta},
        )

    def text_done(self, output_index: int, item_id: str, content_index: int, text: str) -> str:
        return self._frame(
            "response.output_text.done",
            {"item_id": item_id, "output_index": output_index, "content_index": content_index, "text": text},
        )

    def text_part_done(self, output_index: int, item_id: str, content_index: int, text: str) -> str:
        return self._frame(
            "response.content_part.done",
            {
                "item_id": item_id,
                "output_index": output_index,
                "content_index": content_index,
                "part": {"type": "output_text", "text": text, "annotations": []},
            },
        )

    def message_done(self, output_index: int, item_id: str, text: str) -> str:
        item = {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        self._output.append(item)
        return self._frame("response.output_item.done", {"output_index": output_index, "item": item})

    # -- function_call item (a tool the agent used) ------------------------

    def function_call_item(self, output_index: int, name: str, arguments: str, call_id: str) -> list[str]:
        """Emit a complete function_call item lifecycle for a tool NAT executed.

        NAT runs tools internally (internally-hosted), so we surface each as a finalized
        function_call item: added -> arguments.delta -> arguments.done -> output_item.done.
        Arguments use their own delta family (no content_part / content_index) per spec §5.
        """
        item_id = f"fc_{uuid.uuid4().hex}"
        added = {"id": item_id, "type": "function_call", "name": name, "status": "in_progress",
                 "arguments": "", "call_id": call_id}
        out = [self._frame("response.output_item.added", {"output_index": output_index, "item": added})]
        out.append(self._frame("response.function_call_arguments.delta",
                               {"item_id": item_id, "output_index": output_index, "delta": arguments}))
        out.append(self._frame("response.function_call_arguments.done",
                               {"item_id": item_id, "output_index": output_index, "arguments": arguments}))
        done_item = {**added, "status": "completed", "arguments": arguments}
        self._output.append(done_item)
        out.append(self._frame("response.output_item.done", {"output_index": output_index, "item": done_item}))
        return out


# ---------------------------------------------------------------------------
# Running a NAT workflow and translating its output into the event stream
# ---------------------------------------------------------------------------


def _tool_args(step: IntermediateStep) -> str:
    data = step.data
    val = getattr(data, "input", None) if data else None
    if val is None:
        return "{}"
    return val if isinstance(val, str) else json.dumps(val, ensure_ascii=False, default=str)


async def _stream_workflow(user_text: str, *, response_id: str, model: str, user_id: str | None,
                           emit_tool_calls: bool) -> AsyncIterator[str]:
    """Yield Open Responses SSE frames produced from one NAT workflow run."""
    rs = ResponseStream(response_id, model)
    msg_index = 0
    msg_id = f"msg_{uuid.uuid4().hex}"

    yield rs.created()
    yield rs.in_progress()
    yield rs.message_added(msg_index, msg_id)
    yield rs.text_part_added(msg_index, msg_id, 0)

    emitted = ""          # text already sent as deltas (to compute true deltas)
    tool_steps: list[IntermediateStep] = []

    try:
        async with load_workflow(CONFIG_FILE) as session_manager: # type: ignore
            # conversation_id == response_id so a memory middleware can chain turns when the
            # client passes previous_response_id (we echo response_id back as the chain key).
            async with session_manager.session(user_id=user_id, conversation_id=response_id) as session:
                async with session.run(user_text) as runner:

                    # Capture tool calls from the per-request IntermediateStep stream.
                    def _on_step(step: IntermediateStep) -> None:
                        if str(step.event_type) == "TOOL_END":
                            tool_steps.append(step)

                    subscription = Context.get().intermediate_step_manager.subscribe(on_next=_on_step)
                    try:
                        # Prefer streaming; fall back to a single result for non-streaming workflows.
                        try:
                            stream = runner.result_stream(to_type=str)
                            async for chunk in stream:
                                text = chunk if isinstance(chunk, str) else str(chunk)
                                # Workflows may yield cumulative snapshots OR true deltas; derive the delta.
                                if text.startswith(emitted):
                                    delta = text[len(emitted):]
                                    emitted = text
                                else:
                                    delta = text
                                    emitted += text
                                if delta:
                                    yield rs.text_delta(msg_index, msg_id, 0, delta)
                        except ValueError:
                            # Workflow does not support streaming output -> single result.
                            emitted = await runner.result(to_type=str)
                            if emitted:
                                yield rs.text_delta(msg_index, msg_id, 0, emitted)
                    finally:
                        subscription.unsubscribe()

        # Close the message item.
        yield rs.text_done(msg_index, msg_id, 0, emitted)
        yield rs.text_part_done(msg_index, msg_id, 0, emitted)
        yield rs.message_done(msg_index, msg_id, emitted)

        # Surface the tools the agent used as additional function_call items.
        if emit_tool_calls:
            idx = msg_index + 1
            for step in tool_steps:
                for frame in rs.function_call_item(idx, step.name or "tool", _tool_args(step),
                                                   call_id=f"call_{step.UUID}"):
                    yield frame
                idx += 1

        yield rs.completed()
        yield rs.done()

    except Exception as exc:  # noqa: BLE001 - surface any failure as spec-compliant error events
        logger.exception("Workflow run failed")
        for frame in rs.failed(str(exc)):
            yield frame
        yield rs.done()


async def _run_non_streaming(user_text: str, *, response_id: str, model: str,
                             user_id: str | None) -> dict:
    """Non-streaming branch: a single Open Responses `response` object."""
    async with load_workflow(CONFIG_FILE) as session_manager: # type: ignore
        async with session_manager.session(user_id=user_id, conversation_id=response_id) as session:
            async with session.run(user_text) as runner:
                text = await runner.result(to_type=str)

    item = {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "status": "completed",
        "output": [item],
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Open Responses ↔ NeMo Agent Toolkit")


@app.post("/v1/responses")
async def create_response(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Invalid JSON body", "type": "invalid_request",
                               "param": None, "code": "invalid_request"}},
        )

    model = body.get("model") or "nat"
    user_text = _extract_input_text(body.get("input"))
    stream = bool(body.get("stream", False))
    emit_tool_calls = bool(body.get("stream_options", {}).get("include_tool_calls", True)) \
        if isinstance(body.get("stream_options"), dict) else True

    # Identity: x-user-id header -> NAT Context.user_id; previous_response_id -> conversation chain.
    user_id = request.headers.get("x-user-id")
    response_id = body.get("previous_response_id") or f"resp_{uuid.uuid4().hex}"

    if not user_text:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "No user text found in `input`.", "type": "invalid_request",
                               "param": "input", "code": "invalid_request"}},
        )

    if stream:
        generator = _stream_workflow(user_text, response_id=response_id, model=model,
                                     user_id=user_id, emit_tool_calls=emit_tool_calls)
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    try:
        result = await _run_non_streaming(user_text, response_id=response_id, model=model, user_id=user_id)
        return JSONResponse(content=result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Non-streaming run failed")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": "model_error", "param": None, "code": "model_error"}},
        )


@app.get("/health")
async def health():
    return {"status": "healthy", "config_file": CONFIG_FILE}

MODEL_ID = "nat"
@app.get("/v1/models")
async def list_models():
    """Minimal OpenAI model list so clients (Open WebUI) can discover the model."""
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": int(time.time()), "owned_by": "nat"}],
    }


def main() -> None:
    global CONFIG_FILE
    parser = argparse.ArgumentParser(description="Open Responses streaming server for a NAT workflow")
    parser.add_argument("--config", required=True, help="Path to the NAT workflow YAML config")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    CONFIG_FILE = args.config
    logging.basicConfig(level=logging.INFO)
    logger.info("Serving Open Responses /v1/responses for NAT config: %s", CONFIG_FILE)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
