<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# filebridge — file-aware Open Responses adapter

Same Open Responses / OpenAI API surface as [`natbridge`](../natbridge/README.md) (it **reuses
natbridge's translators**, so the API standard is preserved), plus one extra step: when an
Open WebUI request carries **both a user message and a file upload**, the adapter

1. **extracts** every uploaded file from the request,
2. **saves** it to a local directory (`--upload-dir`), and
3. **appends the file's absolute path** to the user message,

before forwarding the turn to `nat serve`. The agent then receives a plain-text request naming the
files on disk (so a tool can open them by path).

## How Open WebUI actually sends files (verified against its source)

- Open WebUI uploads the browser file to **its own** backend `POST /api/v1/files/` and stores it on
  disk at `{DATA_DIR}/uploads/{id}_{name}`. The original binary is **never** sent to the provider.
- `metadata` is **stripped** before the request reaches the provider (`openai.py`) — so file ids do
  **not** arrive via `metadata.files`.
- In **native function-calling** mode, Open WebUI injects a file reference into the user message
  text (`add_file_context`):
  ```
  <attached_files>
  <file type="file" url="/api/v1/files/{id}/content" content_type="application/pdf" name="report.pdf"/>
  </attached_files>

  <your message>
  ```
- In default **RAG** mode it injects extracted-text chunks, each carrying the file id:
  `<source id="1" name="docker-compose.yml" resource-type="file" resource-id="{id}">…</source>`.
  filebridge reads that `resource-id` too, so **both modes work**.

## How filebridge gets the original file (primary path)

Because filebridge runs on the **same machine** as Open WebUI, it reads the file straight from
Open WebUI's storage:

1. parse the file id from the message text — either the `<file … url="…/files/{id}/content"…/>`
   tag (native mode) or the `<source … resource-id="{id}" …>` tag (RAG mode),
2. locate the file at `--owui-upload-dir/{id}_*` and **copy it** into `--upload-dir`,
3. **append the local absolute path** to the message before forwarding to `nat serve` (the
   unreachable `<attached_files>` block is stripped; RAG `<source>` context is left as-is).

> **Requirement:** point `--owui-upload-dir` at the running instance's `{DATA_DIR}/uploads`. Find
> it with the exact path your instance uses — e.g. for `open-webui serve` from a venv it is
> `<venv>/lib/pythonX.Y/site-packages/open_webui/data/uploads` (set a fixed `DATA_DIR` env var to
> keep it stable). Works in both RAG (default) and native function-calling modes.

## Other file paths (also supported)

- **Inline base64 / data-URI** parts (images, some clients) are decoded and saved directly.
- **`POST /api/v1/files/`** (multipart) — filebridge also exposes this endpoint; uploads routed
  here are saved and linked to the next chat turn by id or via a per-user pending queue. (Only
  reached if you proxy Open WebUI's uploads to filebridge.)

Files over `--max-file-mb` are skipped. File handling never breaks the turn — on any error it
forwards the text alone.

## What the agent receives

For request *"Analyze this"* + `report.pdf`:

```
Analyze this

Attached file(s):
- /tmp/nat_uploads/287c191a_report.pdf
```

## Modules

| File | Responsibility |
| --- | --- |
| `config.py` | `FileAdapterConfig` = natbridge's `AdapterConfig` + `upload_dir` / `max_file_bytes` / `attachment_label`. |
| `files.py` | Extract uploads from the body, decode + save to disk, append absolute paths. |
| `app.py` | `create_app(config)` (4 routes) + CLI `main()`; reuses natbridge translators. |

## Run

```bash
# nat serve (window 1)
uv run nat serve --config_file <your workflow>.yml          # :8000

# adapter (window 2)
python my_example/openresponses/run_filebridge.py \
    --nat-url http://localhost:8000/chat/stream --port 8001 \
    --upload-dir /tmp/nat_uploads \
    --owui-upload-dir /home/minhth11/Projects/API_UI/open-webui/backend/data/uploads
# or, from this directory:  python -m filebridge --nat-url ... --owui-upload-dir ...
```

In Open WebUI: Base URL `http://localhost:8001/v1`. Set `api_type: responses` for the separate
Thinking / Tool Executed boxes (same as natbridge).
