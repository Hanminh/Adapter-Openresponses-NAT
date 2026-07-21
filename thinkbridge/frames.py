# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phân loại frame kiểu "mọi thứ còn lại đều là thinking" — chạy cho CẢ HAI kiểu workflow.

Khác `natbridge.frames` ở điểm cốt lõi
--------------------------------------
`natbridge` nhận diện đáp án cuối bằng cách dò chuỗi `"Final Answer:"` bên trong `**Output:**`.
Đó là định dạng RIÊNG của `react_agent`. Khi workflow là `langgraph_wrapper` (agent LangGraph tự
viết), payload không hề có marker đó -> natbridge không bao giờ tìm thấy FINAL, mọi thứ rơi vào
thinking và người dùng không thấy câu trả lời.

`thinkbridge` bỏ hẳn việc dò marker. Nó phân loại theo NGUỒN của dữ liệu:

    Tool     : intermediate step có `name` bắt đầu bằng "Tool:"  (step_adaptor luôn gắn tiền tố này
               cho MỌI TOOL_END, bất kể agent là react_agent hay LangGraph — xem
               nat/front_ends/fastapi/step_adaptor.py:172)
    Final    : các token trên dòng `data:` (ChatResponseChunk).  KHÔNG phải việc của module này.
    Thinking : MỌI intermediate step còn lại.

Vì sao `data:` là đáp án cuối cho cả hai kiểu workflow:
  * `react_agent._stream_fn` buffer token cho tới khi thấy "Final Answer:" rồi CHỈ yield phần đáp án
    (nvidia_nat_langchain/.../react_agent/register.py:210-233).
  * `langgraph_wrapper` + `StreamSafeGraph` yield token của câu trả lời (và với `stream_tags` thì
    chỉ đúng lượt trả lời cuối).
"""

from __future__ import annotations

import html
import re
from typing import Literal

from thinkbridge.config import ThinkAdapterConfig

FrameKind = Literal["tool", "thinking", "skip"]   # KHÔNG có "final": đáp án đến từ `data:`


def _holdback_len(buf: str, marker: str) -> int:
    """Độ dài hậu tố dài nhất của `buf` là tiền tố của `marker` (để không phát vội thẻ cắt ngang)."""
    for k in range(min(len(buf), len(marker) - 1), 0, -1):
        if buf[-k:].lower() == marker[:k].lower():
            return k
    return 0


class ThinkStripper:
    """Bộ lọc STREAMING bỏ khối `<think>...</think>` inline khỏi luồng token đáp án.

    Qwen (bật thinking) trả `<think>...suy nghĩ...</think>đáp án` ngay trong content. Token bị
    stream từng mảnh nên thẻ có thể cắt ngang giữa hai chunk -> cần máy trạng thái + giữ đệm.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in = False

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        while self._buf:
            if not self._in:
                idx = self._buf.lower().find(self._OPEN)
                if idx >= 0:
                    out.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(self._OPEN):]
                    self._in = True
                    continue
                hold = _holdback_len(self._buf, self._OPEN)
                if hold < len(self._buf):
                    out.append(self._buf[:len(self._buf) - hold])
                    self._buf = self._buf[len(self._buf) - hold:]
                break
            idx = self._buf.lower().find(self._CLOSE)
            if idx >= 0:
                self._buf = self._buf[idx + len(self._CLOSE):]
                self._in = False
                continue
            self._buf = self._buf[len(self._buf) - _holdback_len(self._buf, self._CLOSE):]
            break
        return "".join(out)

    def flush(self) -> str:
        out = "" if self._in else self._buf
        self._buf = ""
        return out


class ThinkFrameParser:
    """Phân loại/parse intermediate step. Chỉ 3 loại; đáp án cuối không đi qua đây."""

    def __init__(self, config: ThinkAdapterConfig) -> None:
        self._cfg = config
        self._input_re = re.compile(re.escape(config.input_marker) + r"\s*```[a-zA-Z]*\n?(.*?)```", re.DOTALL)
        self._output_re = re.compile(re.escape(config.output_marker) + r"\s*```[a-zA-Z]*\n?(.*?)```", re.DOTALL)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _payload(step: dict) -> str:
        return html.unescape(str(step.get("payload") or ""))

    def is_tool(self, name: str) -> bool:
        return name.startswith(self._cfg.tool_name_prefix)

    def is_noise(self, name: str) -> bool:
        return any(m in name for m in self._cfg.noise_name_markers)

    def classify(self, step: dict) -> FrameKind:
        name = step.get("name") or ""
        if self.is_tool(name):
            return "tool"
        if self.is_noise(name):
            return "skip"
        return "thinking"          # <- MỌI THỨ CÒN LẠI

    # -- thinking -----------------------------------------------------------

    def thinking_text(self, step: dict) -> str | None:
        """Nội dung hiển thị trong hộp Thinking cho một step bất kỳ (không phải tool/noise).

        Ưu tiên phần `**Output:**` (đầu ra thật của model). Nếu step chỉ có `**Input:**` (dump
        system prompt) thì bỏ qua — đó là nhiễu, không phải suy nghĩ.
        """
        name = step.get("name") or ""
        if self.is_tool(name) or self.is_noise(name):
            return None

        payload = self._payload(step)
        marker = self._cfg.output_marker
        if marker not in payload:
            return None                      # input-only dump -> nhiễu

        text = payload.split(marker, 1)[1].strip()
        if not text:
            return None

        # react_agent nhét cả suy nghĩ lẫn đáp án vào cùng một **Output:**. Giữ "Thought:", bỏ đáp
        # án — đáp án đã được stream qua `data:`, để lại là hiện hai lần.
        fa = self._cfg.final_answer_marker
        if self._cfg.strip_final_answer_from_thinking and fa in text:
            text = text.split(fa, 1)[0].strip()

        return text or None

    # -- tool ---------------------------------------------------------------

    def parse_tool(self, step: dict) -> tuple[str, str, str | None]:
        """(tên tool, tham số, kết quả|None). Trả về ngay cả khi chưa có kết quả để hiện box
        "tool đang chạy"."""
        raw = step.get("name") or ""
        name = raw.split(":", 1)[1].strip() if ":" in raw else raw
        payload = self._payload(step)

        mi = self._input_re.search(payload)
        args = (mi.group(1).strip() if mi else "") or "{}"
        if "   (" in args:                       # bỏ phần văn xuôi đuôi sau JSON
            args = args.split("   (")[0].strip()

        result: str | None = None
        if self._cfg.output_marker in payload:
            mo = self._output_re.search(payload)
            result = (mo.group(1).strip() if mo
                      else payload.split(self._cfg.output_marker, 1)[1].strip()) or None
        return name, args, result
