<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# thinkbridge — "mọi thứ còn lại đều là thinking"

Adapter Open Responses đứng trước `nat serve`, giống `filebridge` (vẫn giữ chuẩn Open Responses,
vẫn nhận file upload từ Open WebUI rồi chèn **đường dẫn tuyệt đối** vào câu hỏi), nhưng đổi **cách
phân loại** những gì NAT phát ra.

## Quy tắc

| Nguồn dữ liệu từ `nat serve /chat/stream` | Hiển thị |
|---|---|
| intermediate step có `name` bắt đầu bằng `Tool:` | **box Tool** |
| các token trên dòng `data:` (`ChatResponseChunk`) | **ĐÁP ÁN CUỐI** (stream từng token) |
| **mọi intermediate step còn lại** | **THINKING** |

Nói cách khác: nếu model không gọi tool và chưa đưa ra kết quả cuối cùng, thì mặc định nó đang
**thinking** — cho tới khi token đáp án đầu tiên xuất hiện.

## Vì sao chạy được cho CẢ HAI kiểu workflow

`natbridge` / `filebridge` nhận diện đáp án bằng cách **dò chuỗi `"Final Answer:"`** trong
`**Output:**`. Đó là định dạng **riêng của `react_agent`**.

`thinkbridge` không dò marker nào cả — nó lấy đáp án từ **luồng token `data:`**, thứ mà **cả hai**
kiểu workflow đều phát ra:

* `workflow._type: react_agent` — `_stream_fn` buffer token cho tới khi thấy `"Final Answer:"` rồi
  **chỉ yield phần đáp án** (`nvidia_nat_langchain/.../react_agent/register.py:210-233`).
* `workflow._type: langgraph_wrapper` — graph được bọc `as_nat_graph` (`StreamSafeGraph`) stream
  token của câu trả lời; với `stream_tags` thì chỉ đúng lượt trả lời cuối.

Tool cũng nhận diện được ở cả hai kiểu: `step_adaptor` luôn gắn tiền tố `Tool: ` cho **mọi**
`TOOL_END`, bất kể agent là `react_agent` hay LangGraph (`fastapi/step_adaptor.py:172`).

## Hai chốt chặn để đáp án không hiện hai lần

| Cờ | Vấn đề nó giải quyết |
|---|---|
| `strip_final_answer_from_thinking` (mặc định **bật**) | `react_agent` nhét cả `Thought:` lẫn `Final Answer:` vào **cùng một** `**Output:**`. Cắt tại marker: giữ Thought, bỏ đáp án (đáp án đã đến qua `data:`). |
| `stop_thinking_on_first_token` (mặc định **bật**) | `langgraph_wrapper` **không có marker nào**: `**Output:**` của lượt LLM cuối **chính là** đáp án. Khi token đáp án đầu tiên tới, ta **vứt khối thinking đang dở** và ngừng phát thinking. |

Đo được (stream giả lập, workflow `langgraph_wrapper`):

```
hộp THINKING:
  filebridge  chứa đáp án? CÓ -> LẶP 2 LẦN ❌  (296 ký tự)
  thinkbridge chứa đáp án? không ✅            (233 ký tự)
```

## Chạy

```bash
python my_example/openresponses/run_thinkbridge.py \
    --nat-url http://localhost:8000/chat/stream --port 8002 \
    --upload-dir /home/minhth11/Projects/Data_Upload \
    --owui-upload-dir /home/minhth11/Projects/API_UI/open-webui/.venv/lib/python3.13/site-packages/open_webui/data/uploads
```

Trong Open WebUI, thêm connection tới `http://localhost:8002/v1`:

* **API Type = Responses** → tool hiện thành box **"Tool Executed"** riêng *(khuyến nghị)*
* **API Type = OpenAI** → tool hiện dưới dạng dòng 🔧 bên trong hộp Thinking

## Cờ CLI

| Cờ | Ý nghĩa |
|---|---|
| `--keep-final-in-thinking` | Giữ nguyên phần sau `Final Answer:` trong Thinking (mặc định: cắt) |
| `--keep-thinking-after-answer` | Vẫn phát khối thinking dở dang khi đáp án bắt đầu (mặc định: vứt) |
| `--hide-tools` | Không hiện box tool |
| `--keep-rag-context` | Giữ khối `<context>` RAG của Open WebUI (mặc định: cắt bỏ) |
| `--upload-dir`, `--owui-upload-dir`, `--max-file-mb` | Giống `filebridge` |

## Quan hệ với các bridge cũ

`natbridge` và `filebridge` **không bị sửa gì**. `thinkbridge` dùng lại của chúng: `nat_client`,
`open_responses` (emitter), `extractors`, `util`, và toàn bộ `filebridge.files` (xử lý upload).
Chỉ `frames.py` (phân loại) và `translator.py` (ánh xạ) là mới.
