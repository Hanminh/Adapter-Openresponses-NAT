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
    
    python my_example/openresponses/my_adapter.py \
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
# (DÙNG CHÍNH XÁC FORMAT 3 KIỂU TRẢ VỀ CỦA /chat/stream):
#
#   KIỂU 1 — THINKING : name == "<model_name>"; payload KHÔNG có "Final Answer:" trong phần
#                       **Output:**. Đây là lúc model đang suy nghĩ (Thought/Action). Khung
#                       **Input:** thuần (toàn bộ system prompt) là NHIỄU -> bỏ.
#   KIỂU 2 — TOOL CALL: name bắt đầu bằng "Tool: <tool_name>"; payload có **Input:** (đối số).
#                       Đây là lúc model GỌI TOOL ("tool đang chạy"). Một số cấu hình kèm
#                       **Output:** (kết quả) ngay trong frame; cấu hình hiện tại trả kết quả
#                       ở frame LLM kế tiếp (HumanMessage observation) -> ta vớt thêm ở đó.
#   KIỂU 3 — FINAL    : name == "<model_name>"; phần **Output:** chứa "Final Answer:".
#                       Văn bản sau "Final Answer:" là câu trả lời cuối, phát theo dạng CUMULATIVE
#                       (lớn dần từng token) -> ta stream phần delta.
#
#   (Bỏ qua: frame `data:` có finish_reason — KHÔNG quan tâm; chỉ dùng làm fallback nếu KIỂU 3 trống.)
#
# ⚠️ CHÚ Ý QUAN TRỌNG: system prompt trong **Input:** có sẵn dòng
#       'Final Answer: the final answer to the original input question'
#    nên KHÔNG được kiểm tra "Final Answer:" trên TOÀN payload (sẽ dính false-positive ở MỌI
#    frame). Phải chỉ xét "Final Answer:" BÊN TRONG phần **Output:** (đầu ra thật của model).
#
# THINKING & FINAL phát ra dạng CUMULATIVE snapshot (lớn dần, reset mỗi lần gọi LLM).
# `_ThinkingExtractor` gom snapshot THINKING thành block; `_FinalAnswerStreamer` stream FINAL.
# ---------------------------------------------------------------------------


def _output_section(step: dict) -> str | None:
    """Return the model's `**Output:**` text (its real output), or None for input-only dumps."""
    payload = html.unescape(str(step.get("payload") or ""))
    if "**Output:**" not in payload:
        return None
    return payload.split("**Output:**", 1)[1].strip() or None


def _is_noise(name: str) -> bool:
    """Internal wrapper frames that aren't one of the 3 user-facing kinds -> skip."""
    return ("<workflow>" in name or name.startswith("Function Start:")
            or name.startswith("Function Complete:"))


def _final_answer_text(step: dict) -> str | None:
    """KIỂU 3: if `**Output:**` contains 'Final Answer:', return the text AFTER it (cumulative)."""
    name = step.get("name") or ""
    if name.startswith("Tool:") or _is_noise(name):
        return None
    out = _output_section(step)
    if out and "Final Answer:" in out:
        return out.split("Final Answer:", 1)[1].strip() or None
    return None


def _thinking_text(step: dict) -> str | None:
    """KIỂU 1: the model's `**Output:**` reasoning (Thought/Action), or None to skip.

    Skip the `Tool: X` frames and the `Function Start/Complete:` wrappers (handled separately /
    noise), the FINAL answer frames (KIỂU 3), and pure `**Input:**` dumps (the system prompt).
    "Final Answer:" is checked ONLY inside **Output:** so the system prompt's own 'Final Answer:'
    line (always present in the input dump) never falsely suppresses the Thinking box.
    """
    name = step.get("name") or ""
    if name.startswith("Tool:") or _is_noise(name):
        return None
    out = _output_section(step)                    # None for input-only dumps -> skip (nhiễu)
    if out is None or "Final Answer:" in out:      # input-dump, or KIỂU 3 -> không phải thinking
        return None
    return out


def _classify(step: dict) -> str:
    """Map a step to one of: 'tool' | 'final' | 'thinking' | 'skip' (per the 3-kind spec)."""
    name = step.get("name") or ""
    if name.startswith("Tool:"):                   # KIỂU 2
        return "tool"
    if _is_noise(name):
        return "skip"
    if _final_answer_text(step) is not None:       # KIỂU 3 (Final Answer trong **Output:**)
        return "final"
    return "thinking"                              # KIỂU 1 (name == model, không phải Final Answer)


class _FinalAnswerStreamer:
    """Streams NAT's cumulative 'Final Answer:' snapshots as incremental deltas."""

    def __init__(self) -> None:
        self._cur = ""

    def feed(self, step: dict) -> str | None:
        full = _final_answer_text(step)
        if not full or full == self._cur:
            return None
        if full.startswith(self._cur):             # snapshot grew -> emit only the new tail
            delta, self._cur = full[len(self._cur):], full
            return delta or None
        self._cur = full                           # rare reset -> emit the new full text
        return full


_HUMAN_MSG_RE = re.compile(r"HumanMessage\(content=(['\"])(.*?)\1,\s*(?:additional_kwargs|response_metadata)=",
                           re.DOTALL)


def _last_observation(step: dict, user_text: str) -> str | None:
    """Best-effort tool RESULT for the current config (result arrives as a HumanMessage in the
    next LLM input dump, not in the `Tool:` frame). Return the LAST HumanMessage that isn't the
    user's question (i.e. the tool observation), or None."""
    payload = html.unescape(str(step.get("payload") or ""))
    q = (user_text or "").strip()
    for _, content in reversed(_HUMAN_MSG_RE.findall(payload)):
        if q and q in content:                     # the user's question wrapper, not a tool result
            continue
        text = content.replace("\\n", "\n").strip()
        if text:
            return text
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


def _parse_tool(step: dict) -> tuple[str, str, str | None]:
    """KIỂU 2: parse a NAT 'Tool: X' step into (tool_name, args, result_or_None).

    The `Tool:` frame always carries **Input:** (the args). It MAY carry **Output:** (the result)
    in some configs; in the current one the result instead comes back as a HumanMessage in the
    next LLM dump (see `_last_observation`). So `result` is None when this frame is Input-only —
    we still emit the tool box (showing the call / "tool đang chạy") and attach the result later.
    """
    raw = step.get("name") or ""
    name = raw.split(":", 1)[1].strip() if ":" in raw else raw
    payload = html.unescape(str(step.get("payload") or ""))
    mi = re.search(r"\*\*Input:\*\*\s*```[a-zA-Z]*\n?(.*?)```", payload, re.DOTALL)
    args = (mi.group(1).strip() if mi else "") or "{}"
    if "   (" in args:                                 # strip trailing prose comments after the JSON
        args = args.split("   (")[0].strip()
    result: str | None = None
    if "**Output:**" in payload:
        mo = re.search(r"\*\*Output:\*\*\s*```[a-zA-Z]*\n?(.*?)```", payload, re.DOTALL)
        result = (mo.group(1).strip() if mo else payload.split("**Output:**", 1)[1].strip()) or None
    return (name, args, result)


async def _stream_open_responses(user_text: str, *, response_id: str, model: str, user_id: str | None,
                                 tools: list | None, instructions: str | None,
                                 previous_response_id: str | None) -> AsyncIterator[str]:
    """Translate NAT /chat/stream into Open Responses events Open WebUI renders specially.

    output[] order (rendered IN ORDER by Open WebUI): `reasoning` items -> collapsible "Thinking"
    boxes, `function_call` (+ `function_call_output`) items -> separate "Tool Executed" boxes,
    `message` item -> the clear final answer. The agent's interleaved flow becomes:
        Thinking -> Tool Executed -> Thinking -> … -> Answer
    A reasoning item is closed before a tool box (or the message) opens, so items never overlap.
    """
    em = OpenResponsesEmitter(response_id, model, tools=tools, instructions=instructions,
                              previous_response_id=previous_response_id,
                              step_payload_max=STEP_PAYLOAD_MAX, max_steps=MAX_STEPS,
                              reasoning_total_max=REASONING_TOTAL_MAX)
    msg_id = f"msg_{uuid.uuid4().hex}"
    extractor = _ThinkingExtractor()      # KIỂU 1: gom snapshot thinking thành block
    fa_stream = _FinalAnswerStreamer()    # KIỂU 3: stream câu trả lời cuối theo delta
    reasoning_open = msg_open = False
    n_blocks = 0
    got_answer = False           # đã stream được token trả lời (từ KIỂU 3) chưa?
    final_answer_fallback = ""   # văn bản "Final Answer:" đầy đủ, dùng nếu chưa stream gì
    data_fallback = ""           # nội dung gộp từ frame `data:` (chỉ dùng khi KIỂU 3 trống)
    tools_seen: dict[tuple[str, str], dict] = {}   # (name, args) -> {"call_id", "result": bool}
    pending_tool: tuple[str, str] | None = None    # tool đã gọi nhưng chưa có kết quả

    def _emit_block(block: str | None):
        nonlocal reasoning_open, n_blocks
        if not block or n_blocks >= MAX_STEPS:
            return []
        n_blocks += 1
        frames = []
        if not reasoning_open:
            frames.append(em.open_reasoning(f"rs_{uuid.uuid4().hex}"))   # a fresh reasoning item
            reasoning_open = True
        f = em.reasoning_delta(_format_block(block, n_blocks))
        if f:
            frames.append(f)
        return frames

    def _close_reasoning_frames():
        nonlocal reasoning_open
        frames = list(_emit_block(extractor.flush()))   # flush the last pending thinking block
        if reasoning_open:
            fr = em.close_reasoning()
            if fr:
                frames.append(fr)
            reasoning_open = False
        return frames

    def _open_message_frames():
        nonlocal msg_open
        frames = _close_reasoning_frames()
        frames.extend(em.open_message(msg_id))
        msg_open = True
        return frames

    def _emit_tool(name: str, args: str, result: str | None):
        """Emit the function_call once per (name,args); attach function_call_output when result known."""
        nonlocal pending_tool
        key = (name, args)
        frames: list[str] = []
        info = tools_seen.get(key)
        if info is None:                                       # lần đầu thấy tool -> "tool đang chạy"
            call_id = f"call_{uuid.uuid4().hex[:12]}"
            info = tools_seen[key] = {"call_id": call_id, "result": False}
            frames += em.tool_call(name, args, call_id)
            pending_tool = key
        if result and not info["result"]:                      # đã có kết quả -> gắn vào box tool
            frames += em.tool_output(info["call_id"], result)
            info["result"] = True
            if pending_tool == key:
                pending_tool = None
        return frames

    def _attach_observation(obj: dict):
        """If a tool is still awaiting its result, try to pull it from this LLM dump's observation."""
        nonlocal pending_tool
        if pending_tool is None:
            return []
        obs = _last_observation(obj, user_text)
        if not obs:
            return []
        return _emit_tool(pending_tool[0], pending_tool[1], obs)

    yield em.created()
    yield em.in_progress()

    try:
        async for kind, obj in _call_nat_chat_stream(user_text, user_id=user_id, conversation_id=response_id):
            if kind == "step":
                cls = _classify(obj)
                if cls == "skip":
                    continue
                if cls == "tool":                              # KIỂU 2: model gọi tool
                    name, args, result = _parse_tool(obj)
                    for frame in _close_reasoning_frames():    # đóng box Thinking trước box Tool
                        yield frame
                    for frame in _emit_tool(name, args, result):
                        yield frame
                    continue
                if cls == "final":                             # KIỂU 3: streaming câu trả lời cuối
                    for frame in _attach_observation(obj):     # vớt kết quả tool từ observation (nếu có)
                        yield frame
                    if not msg_open:
                        for frame in _open_message_frames():
                            yield frame
                    fa = _final_answer_text(obj)
                    if fa:
                        final_answer_fallback = fa
                    delta = fa_stream.feed(obj)
                    if delta:
                        got_answer = True
                        yield em.message_delta(delta)
                    continue
                # KIỂU 1: model đang thinking
                for frame in _attach_observation(obj):
                    yield frame
                if not msg_open:
                    for frame in _emit_block(extractor.feed(obj)):
                        yield frame
            elif kind == "token":
                # frame `data:` — KHÔNG dùng làm câu trả lời chính (KIỂU 3 lo việc đó); chỉ gom dự phòng.
                for choice in obj.get("choices", []):
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        data_fallback += content

        # Hết stream: mở message nếu cần; nếu KIỂU 3 chưa stream được gì thì dùng fallback.
        if not msg_open:
            for frame in _open_message_frames():
                yield frame
        if not got_answer:
            text = final_answer_fallback or data_fallback
            if text:
                yield em.message_delta(text)
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
                  tool_calls: list | None = None,  # giữ tham số nhưng KHÔNG dùng (xem _stream_openai)
                  role: str | None = None, finish_reason: str | None = None) -> str:
    delta: dict = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
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

    KIỂU 1 thinking + KIỂU 2 tool -> `delta.reasoning_content` (Open WebUI "Thinking" box; tools
    go here, NOT in `delta.tool_calls`, so Open WebUI doesn't try to re-execute them);
    KIỂU 3 final answer -> `delta.content` (the clear final answer, streamed as deltas).
    For a SEPARATE "Tool Executed" box, use /v1/responses instead.
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    extractor = _ThinkingExtractor()      # KIỂU 1
    fa_stream = _FinalAnswerStreamer()    # KIỂU 3
    n_blocks = 0
    answer_started = False
    got_answer = False
    final_answer_fallback = ""
    data_fallback = ""
    tools_seen: dict[tuple[str, str], dict] = {}
    pending_tool: tuple[str, str] | None = None
    yield _openai_chunk(chat_id, model, role="assistant")

    def _block_chunk(block: str | None):
        nonlocal n_blocks
        if not block or n_blocks >= MAX_STEPS:
            return None
        n_blocks += 1
        return _openai_chunk(chat_id, model, reasoning_content=_format_block(block, n_blocks))

    def _tool_chunk(name: str, args: str, result: str | None):
        # LƯU Ý: ở OpenAI chat completions, `delta.tool_calls` nghĩa là "model muốn CLIENT chạy tool"
        # -> Open WebUI sẽ cố THỰC THI current_datetime (vốn đã chạy trong NAT) -> lỗi/treo. Vì vậy ở
        # path OpenAI ta hiển thị tool TRONG hộp Thinking (reasoning_content). Muốn có "Tool Executed"
        # box RIÊNG (chỉ hiển thị, không thực thi) -> dùng /v1/responses.
        nonlocal n_blocks, pending_tool
        key = (name, args)
        info = tools_seen.get(key)
        if info is None:                                       # lần đầu -> "tool đang chạy"
            info = tools_seen[key] = {"result": False}
            pending_tool = key
            block = f"🔧 **Tool:** `{name}`\n- Input: `{args}`"
            if result:
                block += f"\n- Result: {result}"
                info["result"] = True
                pending_tool = None
            n_blocks += 1
            return _openai_chunk(chat_id, model, reasoning_content=_format_block(block, n_blocks))
        if result and not info["result"]:                      # đã chạy -> bổ sung kết quả
            info["result"] = True
            if pending_tool == key:
                pending_tool = None
            n_blocks += 1
            return _openai_chunk(chat_id, model,
                                 reasoning_content=_format_block(f"🔧 **Result** `{name}`: {result}", n_blocks))
        return None

    def _maybe_attach(obj: dict):
        nonlocal pending_tool
        if pending_tool is None:
            return None
        obs = _last_observation(obj, user_text)
        if not obs:
            return None
        return _tool_chunk(pending_tool[0], pending_tool[1], obs)

    try:
        async for kind, obj in _call_nat_chat_stream(user_text, user_id=user_id, conversation_id=conversation_id):
            if kind == "step":
                cls = _classify(obj)
                if cls == "skip":
                    continue

                if cls == "tool":                              # KIỂU 2: model gọi tool
                    name, args, result = _parse_tool(obj)
                    chunk = _tool_chunk(name, args, result)
                    if chunk:
                        yield chunk
                    continue

                if cls == "final":                             # KIỂU 3: streaming câu trả lời cuối
                    chunk = _maybe_attach(obj)                 # vớt kết quả tool từ observation
                    if chunk:
                        yield chunk
                    fa = _final_answer_text(obj)
                    if fa:
                        final_answer_fallback = fa
                    if not answer_started:
                        flush_chunk = _block_chunk(extractor.flush())   # dọn nốt hộp thinking
                        if flush_chunk:
                            yield flush_chunk
                        answer_started = True
                    delta = fa_stream.feed(obj)
                    if delta:
                        got_answer = True
                        yield _openai_chunk(chat_id, model, content=delta)
                    continue

                # KIỂU 1: model đang thinking
                chunk = _maybe_attach(obj)
                if chunk:
                    yield chunk
                if not answer_started:
                    block = _block_chunk(extractor.feed(obj))
                    if block:
                        yield block

            elif kind == "token":
                # frame `data:` — KHÔNG dùng làm câu trả lời chính (KIỂU 3 lo việc đó); chỉ gom dự phòng.
                for choice in obj.get("choices", []):
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        data_fallback += content

        if not answer_started:                          # đảm bảo đóng hộp thinking
            flush_chunk = _block_chunk(extractor.flush())
            if flush_chunk:
                yield flush_chunk
            answer_started = True
        if not got_answer:                              # KIỂU 3 trống -> dùng fallback
            text = final_answer_fallback or data_fallback
            if text:
                yield _openai_chunk(chat_id, model, content=text)
        yield _openai_chunk(chat_id, model, finish_reason="stop")
        yield "data: [DONE]\n\n"
        
    except Exception as exc:  # noqa: BLE001
        logger.exception("OpenAI stream failed")
        err = {"error": {"message": str(exc), "type": "server_error", "code": "server_error"}}
        yield f"data: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat endpoint (Open WebUI's default mode hits this)."""
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

    # Non-streaming: assemble a single chat.completion (answer text only).
    text = ""
    final_answer_fallback = ""
    async for kind, obj in _call_nat_chat_stream(user_text, user_id=user_id, conversation_id=conversation_id):
        if kind == "token":
            for choice in obj.get("choices", []):
                text += (choice.get("delta") or {}).get("content") or ""
        else:
            fa = _final_answer_text(obj)
            if fa:
                final_answer_fallback = fa
    if not text:
        text = final_answer_fallback
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
