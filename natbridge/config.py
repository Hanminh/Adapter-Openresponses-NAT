# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter configuration — every knob in one place.

`AdapterConfig` bundles BOTH the runtime settings (URLs, size limits) AND the NAT frame-format
markers used to classify `/chat/stream` steps. Keeping the markers here means adapting to a
different agent's wire format is a config change, not a code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


@dataclass(slots=True)
class AdapterConfig:
    """All adapter settings. Defaults match a verbose `react_agent` behind `nat serve`."""

    # -- runtime ------------------------------------------------------------
    nat_chat_stream_url: str = "http://localhost:8000/chat/stream"
    http_timeout: float = 120.0
    model_id: str = "nat"

    # -- size bounds (keep every SSE line < client limit; Open WebUI errors at 131072 B / 128KB)
    step_payload_max: int = 1000      # max chars per thinking block / tool arg / tool result
    max_steps: int = 50               # max thinking blocks surfaced
    reasoning_total_max: int = 8000   # max total chars in the accumulated Thinking box

    # -- NAT /chat/stream frame markers (see natbridge.frames) ---------------
    tool_name_prefix: str = "Tool:"                       # KIỂU 2: name starts with this
    noise_name_markers: tuple[str, ...] = field(          # internal wrappers -> skip
        default_factory=lambda: ("<workflow>", "Function Start:", "Function Complete:"))
    input_marker: str = "**Input:**"
    output_marker: str = "**Output:**"
    final_answer_marker: str = "Final Answer:"            # KIỂU 3: appears inside **Output:**
