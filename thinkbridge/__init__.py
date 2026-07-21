# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""thinkbridge — Open Responses adapter: mọi thứ không phải tool và không phải đáp án đều là thinking.

Chạy được cho CẢ HAI kiểu workflow của NAT (`react_agent` và `langgraph_wrapper`), vì đáp án cuối
được lấy từ luồng token `data:` chứ không dò chuỗi "Final Answer:".
"""
