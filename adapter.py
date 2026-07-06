# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Open Responses adapter in FRONT of a running `nat serve`.

Flow:  user (Open Responses `POST /v1/responses`)  ->  this adapter  ->  nat serve `/chat/stream`
       and back, translating NAT's SSE into Open Responses semantic events.

Unlike `server.py` (which runs the workflow in-process via `load_workflow`), this adapter is a
pure HTTP↔HTTP translator: it forwards to an already-running `nat serve` and re-encodes the
stream. NAT's `/chat/stream` emits two frame kinds (verified in
nat/front_ends/fastapi/response_helpers.py):
  - `data: {ChatResponseChunk}`               -> answer tokens (choices[].delta.content)
  - `intermediate_data: {ResponseIntermediateStep}` -> reasoning/tool steps
  - `data: [DONE]`                            -> end

Translation to Open Responses (tuned so Open WebUI shows Thinking + clear answer):
  intermediate steps    -> a single `reasoning` item FIRST (Open WebUI: a collapsible "Thinking"
                           box). NAT's verbose react_agent emits a flood of steps (full
                           system-prompt dumps + cumulative LLM-output snapshots); we DROP the
                           noise and de-duplicate the snapshots into clean ReAct reasoning blocks
                           (see `_ThinkingExtractor`).
  answer tokens         -> a `message` item with `output_text` deltas AFTER the reasoning item
                           (Open WebUI: the clear final answer).
Open WebUI renders output[] items IN ORDER: `reasoning` -> Thinking box, `message` -> answer. So
the reasoning item MUST precede the message; we close the Thinking box when the first answer
token arrives.

Also exposes OpenAI-compatible `GET /v1/models` + `POST /v1/chat/completions` (Open WebUI's
default mode): steps -> `delta.reasoning_content` (also a Thinking box), answer -> `delta.content`.

Run (adapter on :8001, nat serve on :8000):
    uv run --extra langchain python my_example/openresponses/adapter.py \
        --nat-url http://localhost:8000/chat/stream --port 8001
    
    python my_example/openresponses/adapter.py \
        --nat-url http://localhost:8000/chat/stream --port 8001 \
        --max-steps 50 --step-payload-max 1000
    
Call:
    curl -N http://localhost:8001/v1/responses -H 'Content-Type: application/json' \
      -d '{"model":"nat","stream":true,
           "input":[{"type":"message","role":"user",
                     "content":[{"type":"input_text","text":"What is the date today?"}]}]}'
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse

# Conformant Open Responses building blocks (full response object + item/content/error events).
from open_responses import OpenResponsesEmitter
from open_responses import extract_input_text

logger = logging.getLogger("openresponses-adapter")

# Set by main().
NAT_CHAT_STREAM_URL = "http://localhost:8000/chat/stream"
HTTP_TIMEOUT = 120.0
# Size bounds so no single SSE `data:` line (esp. the terminal response.completed, which embeds
# output[]) exceeds client line-length limits (Open WebUI errors at 131072 bytes / 128KB).
STEP_PAYLOAD_MAX = 1500      # max chars kept per thinking block
MAX_STEPS = 40              # max number of thinking blocks surfaced
REASONING_TOTAL_MAX = 8000  # max total chars in the accumulated Thinking box


# ---------------------------------------------------------------------------
# Classify a NAT /chat/stream `intermediate_data:` step into one of 3 states
# (per the NAT react_agent frame format):
#
#   1. TOOL CALL  -> name starts with "Tool: <tool_name>"; payload has **Input:** (args) and,
#                    on the END frame, **Output:** (the tool result).
#   2. THINKING   -> name == "<model_name>"; payload has **Output:** WITHOUT "Final Answer:"
#                    (the agent's Thought/Action reasoning). Pure **Input:** dumps (the full
#                    system prompt) and "Function Start/Complete: <workflow>" are NOISE -> skip.
#   3. FINAL      -> name == "<model_name>"; payload **Output:** contains "Final Answer:".
#                    The answer itself streams cleanly via the `data:` frames, so we DROP these
#                    from the Thinking box (keeping only `data:` as the answer; with a fallback).
#
# THINKING and FINAL are emitted as CUMULATIVE snapshots (growing token by token, resetting per
# LLM call). `_ThinkingExtractor` de-duplicates them into one block per completed snapshot.
# ---------------------------------------------------------------------------


def _format_tool(tool_name: str, payload: str) -> str:
    """Render a NAT 'Tool: X' step as a clear Thinking-box entry (input args + result)."""
    block = f"🔧 **Tool call:** `{tool_name}`"
    mi = re.search(r"\*\*Input:\*\*\s*```[a-zA-Z]*\n?(.*?)```", payload, re.DOTALL)
    if mi and mi.group(1).strip():
        block += f"\n- Input: `{mi.group(1).strip()}`"
    if "**Output:**" in payload:
        mo = re.search(r"\*\*Output:\*\*\s*```[a-zA-Z]*\n?(.*?)```", payload, re.DOTALL)
        result = (mo.group(1).strip() if mo else payload.split("**Output:**", 1)[1].strip())
        if result:
            block += f"\n- Result: {result}"
    return block    


def _thinking_text(step: dict) -> str | None:
    """Text to show in the Thinking box for a step (TOOL or THINKING), or None to skip."""
    name = step.get("name") or ""
    if "<workflow>" in name:                       # workflow wrapper -> noise
        return None
    payload = html.unescape(str(step.get("payload") or ""))
    if name.startswith("Tool:"):                   # (1) TOOL CALL
        return _format_tool(name.split(":", 1)[1].strip(), payload)
    if "**Output:**" in payload:
        out = payload.split("**Output:**", 1)[1].strip()
        if "Final Answer:" in out:                 # (3) FINAL answer -> not thinking (use data frames)
            return None
        return out or None                         # (2) THINKING (reasoning)
    return None                                    # pure **Input:** dump -> noise


def _final_answer_text(step: dict) -> str | None:
    """If this step is the streaming FINAL answer, return the text after 'Final Answer:'."""
    name = step.get("name") or ""
    if name.startswith("Tool:") or "<workflow>" in name:
        return None
    payload = html.unescape(str(step.get("payload") or ""))
    if "**Output:**" in payload:
        out = payload.split("**Output:**", 1)[1]
        if "Final Answer:" in out:
            return out.split("Final Answer:", 1)[1].strip() or None
    return None


class _ThinkingExtractor:
    """De-duplicates NAT's cumulative thinking/tool snapshots into discrete blocks."""

    def __init__(self) -> None:
        self._cur = ""

    def feed(self, step: dict) -> str | None:
        """Feed a step; return a COMPLETED block when a new one starts, else None."""
        snap = _thinking_text(step)
        if snap is None:
            return None
        if snap.startswith(self._cur):             # same block still growing (or identical)
            self._cur = snap
            return None
        completed, self._cur = self._cur, snap     # a new block started -> previous is complete
        return completed or None

    def flush(self) -> str | None:
        """Return the final pending block."""
        cur, self._cur = self._cur, ""
        return cur or None


async def _call_nat_chat_stream(user_text: str, *, user_id: str | None,
                                conversation_id: str | None) -> AsyncIterator[tuple[str, dict]]:
    """Call `nat serve` /chat/stream and yield ("token", chunk) / ("step", step) tuples."""
    body = {"messages": [{"role": "user", "content": user_text}], "stream": True}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if user_id:
        headers["x-user-id"] = user_id
    if conversation_id:
        headers["conversation-id"] = conversation_id

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        async with client.stream("POST", NAT_CHAT_STREAM_URL, json=body, headers=headers) as resp:
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


def _format_block(text: str, n: int) -> str:
    """Format one reasoning block for the Thinking box (truncated)."""
    if STEP_PAYLOAD_MAX and len(text) > STEP_PAYLOAD_MAX:
        text = text[:STEP_PAYLOAD_MAX] + " …[truncated]"
    return f"**Step {n}**\n{text}\n\n"


async def _stream_open_responses(user_text: str, *, response_id: str, model: str, user_id: str | None,
                                 tools: list | None, instructions: str | None,
                                 previous_response_id: str | None) -> AsyncIterator[str]:
    """Translate NAT /chat/stream into Open Responses events Open WebUI renders specially.

    output[] order (required by Open WebUI's renderer): a single `reasoning` item FIRST (the
    collapsible "Thinking" box, accumulating the agent's ReAct steps), then the `message` item
    (the clear final answer). The reasoning item is closed the moment the first answer token
    arrives, so Open WebUI collapses it to "Thought…" and shows the answer clearly.
    """
    em = OpenResponsesEmitter(response_id, model, tools=tools, instructions=instructions,
                              previous_response_id=previous_response_id,
                              step_payload_max=STEP_PAYLOAD_MAX, max_steps=MAX_STEPS,
                              reasoning_total_max=REASONING_TOTAL_MAX)
    reasoning_id = f"rs_{uuid.uuid4().hex}"
    msg_id = f"msg_{uuid.uuid4().hex}"
    extractor = _ThinkingExtractor()
    reasoning_open = msg_open = False
    n_blocks = 0
    got_answer = False           # did any `data:` frame carry answer text?
    final_answer_fallback = ""   # cumulative "Final Answer:" text, used only if no `data:` answer

    def _emit_block(block: str | None):
        nonlocal reasoning_open, n_blocks
        if not block or n_blocks >= MAX_STEPS:
            return []
        n_blocks += 1
        frames = []
        if not reasoning_open:
            frames.append(em.open_reasoning(reasoning_id))
            reasoning_open = True
        f = em.reasoning_delta(_format_block(block, n_blocks))
        if f:
            frames.append(f)
        return frames

    def _open_message_frames():
        nonlocal reasoning_open, msg_open
        frames = []
        for fr in _emit_block(extractor.flush()):     # flush last thinking block
            frames.append(fr)
        if reasoning_open:
            fr = em.close_reasoning()                 # close Thinking box before the answer
            if fr:
                frames.append(fr)
        frames.extend(em.open_message(msg_id))
        msg_open = True
        return frames

    yield em.created()
    yield em.in_progress()

    try:
        async for kind, obj in _call_nat_chat_stream(user_text, user_id=user_id, conversation_id=response_id):
            if kind == "step" and not msg_open:
                fa = _final_answer_text(obj)          # capture FINAL answer as a fallback
                if fa:
                    final_answer_fallback = fa
                for frame in _emit_block(extractor.feed(obj)):  # TOOL / THINKING block, if completed
                    yield frame
            elif kind == "token":
                if not msg_open:
                    for frame in _open_message_frames():
                        yield frame
                for choice in obj.get("choices", []):
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        got_answer = True
                        yield em.message_delta(delta)

        # End of stream: open the message if needed, and fall back to the captured Final Answer
        # when the `data:` frames carried no answer text.
        if not msg_open:
            for frame in _open_message_frames():
                yield frame
        if not got_answer and final_answer_fallback:
            yield em.message_delta(final_answer_fallback)
        for frame in em.close_message():
            yield frame

        yield em.completed()
        yield em.done()

    except httpx.HTTPStatusError as exc:
        for frame in em.failed(f"nat serve returned {exc.response.status_code}", err_type="server_error",
                               code="server_error"):
            yield frame
        yield em.done()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Adapter stream failed")
        for frame in em.failed(str(exc)):
            yield frame
        yield em.done()


app = FastAPI(title="Open Responses adapter → nat serve /chat/stream")


@app.post("/v1/responses")
async def create_response(request: Request):
    try:
        req_body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Invalid JSON body", "type": "invalid_request",
                               "param": None, "code": "invalid_request"}},
        )

    model = req_body.get("model") or "nat"
    user_text = extract_input_text(req_body.get("input"))
    stream = bool(req_body.get("stream", False))
    tools = req_body.get("tools") or []
    instructions = req_body.get("instructions")
    previous_response_id = req_body.get("previous_response_id")
    user_id = request.headers.get("x-user-id")
    # conversation_id chains turns; reuse previous_response_id when given so memory continues.
    response_id = previous_response_id or f"resp_{uuid.uuid4().hex}"

    if not user_text:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "No user text found in `input`.", "type": "invalid_request",
                               "param": "input", "code": "invalid_request"}},
        )

    if not stream:
        # Non-streaming: drain the translation; assemble reasoning item (steps) + message (answer).
        em = OpenResponsesEmitter(response_id, model, tools=tools, instructions=instructions,
                                  previous_response_id=previous_response_id,
                                  step_payload_max=STEP_PAYLOAD_MAX, max_steps=MAX_STEPS,
                                  reasoning_total_max=REASONING_TOTAL_MAX)
        extractor = _ThinkingExtractor()
        blocks: list[str] = []
        text = ""
        final_answer_fallback = ""
        try:
            async for kind, obj in _call_nat_chat_stream(user_text, user_id=user_id, conversation_id=response_id):
                if kind == "token":
                    for choice in obj.get("choices", []):
                        text += (choice.get("delta") or {}).get("content") or ""
                else:
                    fa = _final_answer_text(obj)
                    if fa:
                        final_answer_fallback = fa
                    if len(blocks) < MAX_STEPS:
                        block = extractor.feed(obj)
                        if block:
                            blocks.append(block)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Non-streaming run failed")
            return JSONResponse(status_code=500, content={
                "error": {"message": str(exc), "type": "model_error", "param": None, "code": "model_error"}})
        final_block = extractor.flush()
        if final_block and len(blocks) < MAX_STEPS:
            blocks.append(final_block)
        if not text:
            text = final_answer_fallback
        if blocks:
            em.open_reasoning(f"rs_{uuid.uuid4().hex}")
            for i, block in enumerate(blocks, start=1):
                em.reasoning_delta(_format_block(block, i))
            em.close_reasoning()
        em.open_message(f"msg_{uuid.uuid4().hex}")
        em.message_delta(text)
        em.close_message()
        return JSONResponse(content=em.final_object())

    return StreamingResponse(
        _stream_open_responses(user_text, response_id=response_id, model=model, user_id=user_id,
                               tools=tools, instructions=instructions,
                               previous_response_id=previous_response_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints (for Open WebUI and other OpenAI clients)
#
# Open WebUI speaks the OpenAI API — it probes GET /v1/models, then POST
# /v1/chat/completions. It does NOT speak Open Responses (/v1/responses). These two routes
# let Open WebUI talk to the same NAT backend through this adapter. Intermediate steps are
# dropped here (the OpenAI chat schema has no place for them; use /v1/responses to get them).
# ---------------------------------------------------------------------------

MODEL_ID = "nat"


@app.get("/v1/models")
async def list_models():
    """Minimal OpenAI model list so clients (Open WebUI) can discover the model."""
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": int(time.time()), "owned_by": "nat"}],
    }


def _openai_chunk(chat_id: str, model: str, *, content: str | None = None,
                  reasoning_content: str | None = None,
                  role: str | None = None, finish_reason: str | None = None) -> str:
    delta: dict = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        # Open WebUI renders delta.reasoning_content in a collapsible "Thinking" box.
        delta["reasoning_content"] = reasoning_content
    chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _last_user_text(messages: list) -> str:
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(p.get("text", "") for p in content
                                 if isinstance(p, dict) and p.get("type") in ("text", "input_text"))
    return ""


async def _stream_openai(user_text: str, *, model: str, user_id: str | None,
                         conversation_id: str | None) -> AsyncIterator[str]:
    """Translate NAT /chat/stream into OpenAI chat.completion.chunk SSE.

    Thinking steps -> `delta.reasoning_content` (Open WebUI "Thinking" box); answer tokens ->
    `delta.content` (the clear final answer). Same clean ReAct extraction as the Responses path.
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    extractor = _ThinkingExtractor()
    n_blocks = 0
    answer_started = False
    got_answer = False
    final_answer_fallback = ""
    yield _openai_chunk(chat_id, model, role="assistant")

    def _block_chunk(block: str | None):
        nonlocal n_blocks
        if not block or n_blocks >= MAX_STEPS:
            return None
        n_blocks += 1
        return _openai_chunk(chat_id, model, reasoning_content=_format_block(block, n_blocks))

    try:
        async for kind, obj in _call_nat_chat_stream(user_text, user_id=user_id, conversation_id=conversation_id):
            if kind == "step" and not answer_started:
                fa = _final_answer_text(obj)
                if fa:
                    final_answer_fallback = fa
                chunk = _block_chunk(extractor.feed(obj))
                if chunk:
                    yield chunk
            elif kind == "token":
                if not answer_started:
                    chunk = _block_chunk(extractor.flush())  # flush last thinking block
                    if chunk:
                        yield chunk
                    answer_started = True
                for choice in obj.get("choices", []):
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        got_answer = True
                        yield _openai_chunk(chat_id, model, content=content)
        if not got_answer and final_answer_fallback:   # `data:` frames carried no answer
            yield _openai_chunk(chat_id, model, content=final_answer_fallback)
        yield _openai_chunk(chat_id, model, finish_reason="stop")
        yield "data: [DONE]\n\n"
    except Exception as exc:  # noqa: BLE001
        logger.exception("OpenAI stream failed")
        err = {"error": {"message": str(exc), "type": "server_error", "code": "server_error"}}
        yield f"data: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "Invalid JSON", "type": "invalid_request"}})

    model = body.get("model") or MODEL_ID
    user_text = _last_user_text(body.get("messages", []))
    stream = bool(body.get("stream", False))
    user_id = request.headers.get("x-user-id")
    conversation_id = body.get("conversation_id") or request.headers.get("conversation-id")

    if not user_text:
        return JSONResponse(status_code=400,
                            content={"error": {"message": "No user message found.", "type": "invalid_request"}})

    if stream:
        return StreamingResponse(
            _stream_openai(user_text, model=model, user_id=user_id, conversation_id=conversation_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # Non-streaming: assemble a single chat.completion.
    text = ""
    async for kind, obj in _call_nat_chat_stream(user_text, user_id=user_id, conversation_id=conversation_id):
        if kind == "token":
            for choice in obj.get("choices", []):
                text += (choice.get("delta") or {}).get("content") or ""
    return JSONResponse(content={
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


@app.get("/health")
async def health():
    return {"status": "healthy", "nat_chat_stream_url": NAT_CHAT_STREAM_URL}


def main() -> None:
    global NAT_CHAT_STREAM_URL, STEP_PAYLOAD_MAX, MAX_STEPS
    parser = argparse.ArgumentParser(description="Open Responses adapter in front of nat serve /chat/stream")
    parser.add_argument("--nat-url", default="http://localhost:8000/chat/stream",
                        help="URL of the running nat serve /chat/stream endpoint")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--step-payload-max", type=int, default=1000,
                        help="Max chars kept per NAT step payload (bounds SSE line size). 0 = no limit.")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max nat:intermediate_step items surfaced (bounds the terminal event). -1 = unlimited.")
    args = parser.parse_args()

    NAT_CHAT_STREAM_URL = args.nat_url
    STEP_PAYLOAD_MAX = args.step_payload_max
    MAX_STEPS = args.max_steps
    logging.basicConfig(level=logging.INFO)
    logger.info("Open Responses adapter -> %s (step_payload_max=%s, max_steps=%s)",
                NAT_CHAT_STREAM_URL, STEP_PAYLOAD_MAX, MAX_STEPS)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
