<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Open Responses streaming API for NeMo Agent Toolkit output

A small server that exposes a NeMo Agent Toolkit workflow behind a **standard Open Responses
`POST /v1/responses` endpoint** and streams the agent's output as Open Responses **semantic
events** (SSE). It is the concrete realization of the bridge described in
`my_instruction/OPEN_RESPONSES_VS_NAT.md` §6.

- Spec / guide: `/home/minhth11/Projects/API_UI/openresponses/instruction/streaming-api-guide.md`
- Server: [`server.py`](server.py) · Workflow config: [`config.yml`](config.yml)

## What it does

## Two variants

| File | Mode | Use when |
|---|---|---|
| [`server.py`](server.py) | **In-process** — builds & runs the workflow itself (`load_workflow`) | You want one process; no separate `nat serve` |
| [`adapter.py`](adapter.py) | **Proxy** — calls a running `nat serve` `/chat/stream` over HTTP and re-encodes the stream | You already run `nat serve` and want an Open Responses front door without touching it |

Both expose the same `POST /v1/responses` and emit the same Open Responses event ladder.

> **Conformance:** `adapter.py` now emits the **full** Open Responses surface — verified field
> shapes against the spec's authoritative Zod schemas
> (`/home/minhth11/Projects/API_UI/openresponses/src/generated/kubb/zod/`):
> - the `response` object carries **all 31 required keys** (`responseResourceSchema`), not a
>   stub — for `created` / `in_progress` / `completed` / `failed`;
> - full **item + content** lifecycles (`message` item → `output_text` content part → deltas);
> - **error** as an `error` event + `response.failed` with the full `{message,type,param,code}`
>   payload;
> - request **`tools`** / `instructions` / `previous_response_id` are echoed into the response;
> - NAT's internally-executed steps are surfaced as **`nat:intermediate_step` items** in
>   `output[]` (framed by `output_item.added/done`) — the spec's pattern for internally-hosted
>   tools (cf. its own `openai:web_search_call` example). The shared, conformant building blocks
>   live in [`open_responses.py`](open_responses.py).
> The `adapter.py` also exposes OpenAI-compatible `GET /v1/models` + `POST /v1/chat/completions`
> for Open WebUI (which speaks OpenAI, not Open Responses).

```bash
# Proxy adapter (adapter on :8001, in front of nat serve on :8000)
uv run nat serve --config_file my_example/openresponses/config.yml --port 8000   # has /chat/stream
uv run --extra langchain python my_example/openresponses/adapter.py \
    --nat-url http://localhost:8000/chat/stream --port 8001
curl -N http://localhost:8001/v1/responses -H 'Content-Type: application/json' \
  -d '{"model":"nat","stream":true,"input":"What is the date today?"}'
```

## What it does (in-process `server.py`)

It runs a NAT workflow (`load_workflow` → `session.run` → `runner.result_stream`) and
translates the result into the Open Responses event ladder:

```
response.created → response.in_progress
  → response.output_item.added         (assistant `message` item)
    → response.content_part.added       (output_text part)
      → response.output_text.delta * N  (streamed answer chunks)
    → response.output_text.done
  → response.content_part.done
  → response.output_item.done
  [→ function_call item(s) for each tool the agent used]   (from the IntermediateStep stream)
→ response.completed
data: [DONE]
```

Mappings from NAT → Open Responses:

| NAT | Open Responses |
|---|---|
| Final answer text (`runner.result_stream`) | `message` item with `output_text` deltas |
| Tool calls (`TOOL_END` IntermediateSteps) | `function_call` items (`function_call_arguments.*`) |
| `x-user-id` header → `Context.user_id` | request identity (used for multi-tenant memory) |
| `previous_response_id` | reused as `conversation_id` so a memory middleware can chain turns |
| any workflow error | `error` event → `response.failed` → `[DONE]` |

## Run

```bash
# Needs a model for the workflow in config.yml (the default uses a `nim` LLM → set the key,
# or point --config at any NAT workflow YAML you already run).
uv run --extra langchain python my_example/openresponses/server.py \
  --config my_example/openresponses/config.yml --port 8000
```

## Call it (Open Responses streaming request)

```bash
curl -N http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -H 'x-user-id: alice' \
  -d '{
        "model": "nat",
        "stream": true,
        "input": [
          { "type": "message", "role": "user",
            "content": [ { "type": "input_text", "text": "What is the date today?" } ] }
        ]
      }'
```

`-N` disables curl buffering so you see events as they stream. You can also pass `input` as a
plain string. Non-streaming (`"stream": false`) returns a single `response` JSON object.

Multi-turn: take the `id` from the response and send it back as `previous_response_id` on the
next request (the server maps it to `conversation_id`; pair with a memory middleware such as
`redis_turn_memory` from `my_example/workflow/custom_middleware_redis/` to actually persist
history).

## Conformance

The server follows the contract in the guide and was verified against it:

- ✅ First event `response.created`, then `response.in_progress`.
- ✅ Each item framed by `output_item.added` / `output_item.done`; each text part framed by
  `content_part.added` → `output_text.delta`* → `output_text.done` → `content_part.done`.
- ✅ `sequence_number` strictly monotonic across the whole stream.
- ✅ Concatenation of all `output_text.delta` == the part's `output_text.done.text`
  (the server derives true deltas even if the workflow yields cumulative snapshots).
- ✅ Each SSE `event:` matches the JSON body `type`; terminal `data: [DONE]`.
- ✅ Mid-stream error → `error` then `response.failed`, then `[DONE]`.
- ✅ Tool calls surfaced as `function_call` items with `function_call_arguments.*`.

To run the repo's official compliance suite against it:

```bash
cd /home/minhth11/Projects/API_UI/openresponses
bun run test:compliance --base-url http://localhost:8000/v1 --api-key dummy \
  --filter basic-response,streaming-response
```

## Notes / limitations

- **Item ordering:** the assistant `message` is emitted first and tool `function_call` items
  follow it. NAT executes tools internally (internally-hosted), so they are surfaced as
  finalized items rather than yielding control to the client. `previous_response_id` /
  `function_call_output` round-trips for *externally*-hosted tools are not implemented.
- **Reasoning items** are not emitted (NAT does not expose a separate reasoning channel here).
- The default `config.yml` uses a `nim` LLM; set its key/endpoint or point `--config` at any
  workflow you already run (e.g. `my_example/workflow/custom_middleware_tool_code/.../config.yml`
  to also persist turns + tool steps to Redis while serving Open Responses).
