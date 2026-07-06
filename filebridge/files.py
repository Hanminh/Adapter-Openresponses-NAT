# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract Open WebUI file uploads from a request, save them locally, and return absolute paths.

Open WebUI (and OpenAI/Open Responses clients) attach files in several shapes. This module scans
all of them and normalizes each to (filename, bytes):

  * Open Responses `input[]` message content parts:
      {"type": "input_file",  "filename": "d.pdf", "file_data": "data:application/pdf;base64,..."}
      {"type": "input_image", "image_url": "data:image/png;base64,..."}
  * OpenAI chat `messages[]` content parts:
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      {"type": "file",      "file": {"filename": "d.pdf", "file_data": "data:...;base64,..."}}
  * Open WebUI top-level `files[]`:
      {"type": "file", "file": {"filename": "d.pdf", "data": "data:...;base64,..."}}
      {"type": "image", "url": "data:image/png;base64,..."}

The data itself may be a `data:` URI, a bare base64 string, or an `http(s)://` URL (downloaded).
"""

from __future__ import annotations

import base64
import binascii
import glob
import logging
import mimetypes
import os
import re
import shutil
import uuid
from typing import Any

import httpx

logger = logging.getLogger("filebridge.files")

_DATA_URI = re.compile(r"^data:(?P<mime>[^;,]*)(?P<b64>;base64)?,(?P<payload>.*)$", re.DOTALL)
# Keys that may carry the file's bytes, in priority order.
_DATA_KEYS = ("file_data", "data", "b64_json", "url", "file_url", "image_url", "content")


def _decode_data(value: Any) -> tuple[bytes | None, str | None]:
    """Return (bytes, mime) from a data: URI, a bare base64 string, or an http(s) URL."""
    if isinstance(value, dict):  # e.g. {"url": "data:..."} from OpenAI image_url
        for k in ("url", "file_data", "data"):
            if isinstance(value.get(k), str):
                return _decode_data(value[k])
        return None, None
    if not isinstance(value, str) or not value:
        return None, None

    m = _DATA_URI.match(value)
    if m:
        mime = m.group("mime") or None
        payload = m.group("payload")
        if m.group("b64"):
            try:
                return base64.b64decode(payload), mime
            except (binascii.Error, ValueError):
                return None, mime
        return payload.encode("utf-8"), mime  # non-base64 data URI (rare)

    if value.startswith(("http://", "https://")):
        try:
            resp = httpx.get(value, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
            return resp.content, resp.headers.get("content-type", "").split(";")[0] or None
        except Exception:  # noqa: BLE001 - best-effort remote fetch
            logger.warning("Could not download attached file URL: %s", value[:80])
            return None, None

    # Heuristic: a bare base64 blob (long, base64 alphabet).
    if len(value) > 64 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", value):
        try:
            return base64.b64decode(value), None
        except (binascii.Error, ValueError):
            return None, None
    return None, None


def _safe_filename(filename: str | None, mime: str | None) -> str:
    """A collision-free, path-safe basename (keeps the original name when given)."""
    if filename:
        base = os.path.basename(filename).strip().replace("\x00", "")
        base = re.sub(r"[^\w.\- ]", "_", base) or "upload"
    else:
        ext = mimetypes.guess_extension(mime or "") or ".bin"
        base = f"upload{ext}"
    return f"{uuid.uuid4().hex[:8]}_{base}"


def _save(data: bytes, filename: str | None, mime: str | None, upload_dir: str,
          max_bytes: int) -> str | None:
    if not data:
        return None
    if max_bytes and len(data) > max_bytes:
        logger.warning("Skipping attached file (%d bytes > max %d)", len(data), max_bytes)
        return None
    os.makedirs(upload_dir, exist_ok=True)
    dest = os.path.join(upload_dir, _safe_filename(filename, mime))
    print(f"Saving file to {dest}")
    with open(dest, "wb") as fh:
        fh.write(data)
    abs_path = os.path.abspath(dest)
    logger.info("Saved attached file -> %s (%d bytes)", abs_path, len(data))
    return abs_path


def _filename_of(obj: dict) -> str | None:
    return obj.get("filename") or obj.get("name") or (obj.get("file") or {}).get("filename") \
        if isinstance(obj, dict) else None


def _save_from_obj(obj: Any, upload_dir: str, max_bytes: int) -> str | None:
    """Given a file-ish dict/part, find its data payload, save it, return the abs path."""
    if not isinstance(obj, dict):
        return None
    # A nested `file` / `image_url` dict carries the real payload.
    for nested_key in ("file", "image_url"):
        if isinstance(obj.get(nested_key), dict):
            saved = _save_from_obj({**obj[nested_key], "filename": _filename_of(obj)}, upload_dir, max_bytes)
            if saved:
                return saved
    filename = obj.get("filename") or obj.get("name")
    for key in _DATA_KEYS:
        if key in obj:
            data, mime = _decode_data(obj[key])
            if data is not None:
                return _save(data, filename, mime, upload_dir, max_bytes)
    return None


def _iter_content_parts(body: dict):
    """Yield every content part from Open Responses `input` and OpenAI `messages`."""
    inp = body.get("input")
    if isinstance(inp, list):
        for item in inp:
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, list):
                yield from content
    for msg in body.get("messages", []) or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            yield from content


_FILE_PART_TYPES = {"input_file", "input_image", "image_url", "file", "image"}


def save_uploaded_files(body: dict, upload_dir: str, max_bytes: int = 25 * 1024 * 1024) -> list[str]:
    """Find every uploaded file in the request body, save it, and return absolute paths."""
    paths: list[str] = []

    # 1) content parts inside input/messages
    for part in _iter_content_parts(body):
        if isinstance(part, dict) and part.get("type") in _FILE_PART_TYPES:
            saved = _save_from_obj(part, upload_dir, max_bytes)
            if saved:
                paths.append(saved)

    # 2) Open WebUI top-level `files`
    for f in body.get("files", []) or []:
        saved = _save_from_obj(f, upload_dir, max_bytes)
        if saved:
            paths.append(saved)

    return paths


def augment_user_text(user_text: str, paths: list[str], label: str = "Attached file(s)") -> str:
    """Append the saved files' absolute paths to the user message."""
    if not paths:
        return user_text
    listing = "\n".join(f"- {p}" for p in paths)
    block = f"{label}:\n{listing}"
    return f"{user_text}\n\n{block}" if user_text else block


# ---------------------------------------------------------------------------
# Multipart upload path — the ONLY way to get the ORIGINAL binary file.
#
# Open WebUI never sends the original document to the model provider: it uploads the binary to its
# own `POST /api/v1/files/` (multipart/form-data) and only forwards extracted <source> text (or
# inline base64 images) in the chat payload. To receive the real file, filebridge exposes the same
# `POST /api/v1/files/` endpoint; uploads routed there land here.
# ---------------------------------------------------------------------------

async def save_upload_stream(upload, upload_dir: str, *, file_id: str | None = None,
                             max_bytes: int = 0) -> tuple[str, str, int]:
    """Stream a Starlette/FastAPI ``UploadFile`` to disk as ``{file_id}_{name}``.

    Returns ``(file_id, absolute_path, size_bytes)``. Reads in 1 MB chunks so large files never
    load fully into memory (mirrors how Open WebUI streams into its storage).
    """
    file_id = file_id or uuid.uuid4().hex
    original = os.path.basename(upload.filename or "unnamed").strip().replace("\x00", "")
    safe = re.sub(r"[^\w.\- ]", "_", original) or "upload"
    os.makedirs(upload_dir, exist_ok=True)
    dest = os.path.join(upload_dir, f"{file_id}_{safe}")

    size = 0
    with open(dest, "wb") as fh:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if max_bytes and size > max_bytes:
                fh.close()
                os.remove(dest)
                raise ValueError(f"uploaded file exceeds max {max_bytes} bytes")
            fh.write(chunk)
    abs_path = os.path.abspath(dest)
    logger.info("Saved multipart upload -> %s (%d bytes)", abs_path, size)
    return file_id, abs_path, size


class UploadRegistry:
    """Remembers files uploaded via ``POST /api/v1/files/`` so the follow-up chat can reference them.

    Two lookup strategies:
      * by id      — ``metadata.files[].id`` in the chat request maps back to a saved path
                     (works when the same ids that filebridge returned are echoed back), and
      * pending    — a per-user queue consumed by the next chat turn (robust for the common
                     "upload a file, then ask about it" flow even if ids don't line up).
    """

    def __init__(self) -> None:
        self._by_id: dict[str, str] = {}
        self._pending: dict[str, list[str]] = {}

    def add(self, file_id: str, path: str, user_id: str | None) -> None:
        self._by_id[file_id] = path
        self._pending.setdefault(user_id or "*", []).append(path)

    def resolve_ids(self, ids: list[str]) -> list[str]:
        return [self._by_id[i] for i in ids if i in self._by_id]

    def take_pending(self, user_id: str | None) -> list[str]:
        """Return and clear the pending uploads for this user (fallback when no id matched).

        Also drains the global ``"*"`` bucket, because multipart uploads carry no user id and are
        registered there — so a chat request from any user still picks them up.
        """
        paths = self._pending.pop(user_id or "*", [])
        if user_id:  # merge the global bucket (uploads with no known user)
            paths += self._pending.pop("*", [])
        return paths


# ---------------------------------------------------------------------------
# Open WebUI same-machine path — read the ORIGINAL file straight from OWUI's storage.
#
# Verified against the Open WebUI source:
#   * `metadata` is stripped before the request reaches the provider (openai.py), so file ids do
#     NOT arrive in metadata.files.
#   * NATIVE FUNCTION-CALLING mode injects into the user message text (add_file_context):
#         <attached_files>
#         <file type="file" url="/api/v1/files/{id}/content" content_type="..." name="report.pdf"/>
#         </attached_files>
#   * DEFAULT RAG mode injects extracted-text chunks, each carrying the file id in resource-id:
#         <source id="1" name="docker-compose.yml" resource-type="file"
#                 resource-id="2d19b04c-...">…extracted text…</source>
#   * Uploaded files are stored on disk at  {DATA_DIR}/uploads/{id}_{name}.
# So we parse BOTH the `<file …/>` url and the `<source … resource-id=…>` tag for the id, then read
# {owui_upload_dir}/{id}_* directly. This works in both modes.
# ---------------------------------------------------------------------------

_FILE_TAG = re.compile(r"<file\b[^>]*/?>", re.IGNORECASE)
_SOURCE_TAG = re.compile(r"<source\b[^>]*>", re.IGNORECASE)
_ATTACHED_BLOCK = re.compile(r"<attached_files>.*?</attached_files>\s*", re.IGNORECASE | re.DOTALL)
_FILE_ID_IN_URL = re.compile(r"/(?:api/v1/)?files/([^/\"'\s]+)/content")
_CLOSE_CONTEXT = re.compile(r"</context>", re.IGNORECASE)


def strip_rag_wrapper(text: str) -> str:
    """Drop Open WebUI's RAG scaffolding, keeping only the real user query.

    The default RAG template is ``### Task … <context>{{CONTEXT}}</context>\\n\\n{{QUERY}}`` — the
    real question sits after the last ``</context>``. If there is no ``<context>`` block the text is
    returned unchanged (nothing to strip).
    """
    matches = list(_CLOSE_CONTEXT.finditer(text))
    if not matches:
        return text
    return text[matches[-1].end():].strip()


def _tag_attr(name: str, tag: str) -> str | None:
    m = re.search(rf'{name}\s*=\s*"([^"]*)"', tag, re.IGNORECASE)
    return m.group(1) if m else None


def _find_stored_file(owui_upload_dir: str, file_id: str, name: str | None) -> str | None:
    """Locate the OWUI-stored file for `file_id` (saved as `{id}_{name}`)."""
    matches = glob.glob(os.path.join(owui_upload_dir, f"{file_id}_*"))
    if matches:
        return matches[0]
    if name:
        cand = os.path.join(owui_upload_dir, f"{file_id}_{os.path.basename(name)}")
        if os.path.exists(cand):
            return cand
    return None


def resolve_owui_files(text: str, owui_upload_dir: str | None, dest_dir: str,
                       max_bytes: int = 0, strip_rag: bool = False) -> tuple[str, list[str]]:
    """Read Open WebUI file references out of the message text, copy the files locally.

    Returns (cleaned_text, [absolute_paths]). The `<attached_files>` block is stripped from the
    text (its `/api/v1/files/.../content` URL is unreachable by the agent); when ``strip_rag`` is
    set, the whole RAG `### Task … <context>…</context>` wrapper is stripped too, leaving only the
    real user query (which is then augmented with the local paths downstream).
    """
    if not text or not owui_upload_dir or not os.path.isdir(owui_upload_dir):
        return (strip_rag_wrapper(text) if (strip_rag and text) else text), []

    ids: dict[str, str | None] = {}
    # native function-calling mode: <file url=".../files/{id}/content" name="..."/>
    for tag in _FILE_TAG.findall(text):
        url = _tag_attr("url", tag)
        m = _FILE_ID_IN_URL.search(url or "")
        if m:
            ids.setdefault(m.group(1), _tag_attr("name", tag))
    for m in _FILE_ID_IN_URL.finditer(text):  # bare .../files/{id}/content urls
        ids.setdefault(m.group(1), None)
    # default RAG mode: <source resource-type="file" resource-id="{id}" name="...">
    for tag in _SOURCE_TAG.findall(text):
        rid = _tag_attr("resource-id", tag)
        rtype = _tag_attr("resource-type", tag)
        if rid and rtype in (None, "file"):
            ids.setdefault(rid, _tag_attr("name", tag))

    paths: list[str] = []
    for file_id, name in ids.items():
        src = _find_stored_file(owui_upload_dir, file_id, name)
        if not src:
            logger.warning("Open WebUI file id %s not found under %s", file_id, owui_upload_dir)
            continue
        if max_bytes and os.path.getsize(src) > max_bytes:
            logger.warning("Skipping Open WebUI file (too large): %s", src)
            continue
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(src))
        try:
            shutil.copyfile(src, dest)
        except OSError:
            logger.exception("Failed to copy Open WebUI file %s", src)
            continue
        abs_path = os.path.abspath(dest)
        paths.append(abs_path)
        logger.info("Resolved Open WebUI upload %s -> %s", file_id, abs_path)

    cleaned = _ATTACHED_BLOCK.sub("", text).strip()
    if strip_rag:
        cleaned = strip_rag_wrapper(cleaned)
    return cleaned, paths


def referenced_file_ids(body: dict) -> list[str]:
    """Collect file ids the chat request refers to (Open WebUI puts them in ``metadata.files``)."""
    ids: list[str] = []
    meta = body.get("metadata") or {}
    for f in (meta.get("files") or []):
        if isinstance(f, dict):
            fid = f.get("id") or (f.get("file") or {}).get("id")
            if fid:
                ids.append(str(fid))
    for f in (body.get("files") or []):  # some builds put files at top level
        if isinstance(f, dict) and f.get("id"):
            ids.append(str(f["id"]))
    return ids
