<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# todobridge — adapter hiển thị box `write_todos` DỰA TRÊN LỜI GỌI TOOL

Giống `thinkbridge` (đứng trước `nat serve /chat/stream`, giữ chuẩn Open Responses, vẫn nhận
file upload Open WebUI), và **hiển thị dựa trên NAT intermediate step** đúng như thinkbridge —
nhưng chỉ quan tâm tool `write_todos`: mỗi todo (từ TodoListMiddleware) thành **một box**, phản
ánh status `pending → in_progress → completed`.

## Cơ chế (giống thinkbridge: đọc tool step)

`langgraph_wrapper._astream` gọi `graph.astream(...)` **không truyền callbacks**, nên tool
`write_todos` chạy bên trong deep agent **không tự sinh NAT intermediate step** → đó là lý do
trước đây không có box nào.

Cách xử lý: emitter phía workflow (`agenticskills/deep_agents/todo_event_stream.py`) chạy TRONG
Context của workflow; mỗi khi state `todos` đổi, nó **đẩy một cặp TOOL_START/TOOL_END** vào
`Context.intermediate_step_manager` với name `write_todos` và todos JSON. Front-end NAT
(`generate_streaming_response → pull_intermediate → StepAdaptor`) phát ra:

    intermediate_data: { name: "Tool: write_todos", payload: "...**Input:** ```json {todos}```..." }

todobridge đọc step này (`kind=="step"`), bóc todos từ khối `**Input:**`, rồi dựng box — **cùng
đường dữ liệu mà thinkbridge dùng cho box tool**. Token đáp án đi `data:` như thường.

## Map todos → box (theo cơ chế TodoListMiddleware)

Mỗi `write_todos` **thay nguyên** danh sách (kèm status mới). Bridge map theo `content`:
- `content` xuất hiện lần đầu → **mở box** (`function_call`, tên box = content).
- `content` → `completed` → **hoàn tất box** (`function_call_output` ✅).
- Cùng `content` ở lần write_todos sau → **không** tạo box trùng (dedup).

## Điều kiện (BẮT BUỘC)

1. Workflow là **deep agent v2** có `TodoListMiddleware` (`build_native_optimize_deep_agent_v2`).
2. Graph bọc bằng **`as_nat_graph_todo_events`** (đẩy write_todos thành NAT step). Nếu bọc bằng
   `as_nat_graph` thường → không có step → không có box.

Deployment + config sẵn: `deployments/telecom_todo_agent.py` + `conf/conf_telecom_todo.yaml`.

## Chạy

```bash
cd my_example/AgenticSkills
uv run nat serve --config_file conf/conf_telecom_todo.yaml           # :8000
python ../openresponses/run_todobridge.py --nat-url http://localhost:8000/chat/stream --port 8003
```
Open WebUI → connection `http://localhost:8003/v1`, **API Type = Responses**. `--hide-answer` nếu
chỉ muốn box.

## Chẩn đoán nhanh (nếu vẫn không box)

Kiểm tra nat serve có phát step write_todos không:
```bash
curl -N http://localhost:8000/chat/stream -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"vì sao cước 0385875019 tăng"}]}' | grep -a "write_todos"
```
- **Có dòng `intermediate_data: ... write_todos`** → nguồn OK; nếu bridge vẫn không box, gửi tôi output đó.
- **Không có** → workflow chưa bọc `as_nat_graph_todo_events`, hoặc model không gọi `write_todos`.

## Routes
`POST /v1/responses` · `POST /v1/chat/completions` · `POST /api/v1/files/` · `GET /v1/models` · `GET /health`
