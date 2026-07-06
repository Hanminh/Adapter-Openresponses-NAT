# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config for the file-aware adapter: natbridge's AdapterConfig + upload settings."""

from __future__ import annotations

from dataclasses import dataclass

from natbridge.config import AdapterConfig


@dataclass(slots=True)
class FileAdapterConfig(AdapterConfig):
    """AdapterConfig plus where/how uploaded files are stored and referenced."""

    upload_dir: str = "/tmp/nat_uploads"                 # where files are copied/saved locally
    max_file_bytes: int = 25 * 1024 * 1024               # per-file size cap (0 = no limit)
    attachment_label: str = "Attached file(s)"           # heading prepended to the path list

    # Path to Open WebUI's own uploads dir ({DATA_DIR}/uploads). Set this (same machine) to read
    # the ORIGINAL uploaded file by id from the `<file .../api/v1/files/{id}/content .../>` tag
    # that Open WebUI injects in native function-calling mode. None disables this path.
    owui_upload_dir: str | None = None

    # Strip Open WebUI's RAG scaffolding ("### Task … <context><source>…</source></context>") so the
    # request forwarded to nat serve is just the real user question + the appended file path(s).
    strip_rag_context: bool = True
