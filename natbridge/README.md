# natbridge

Phiên bản **module hóa & tổng quát hóa** của `my_adapter.py` + `open_responses.py`. Cùng hành vi
(đã verify trên đúng 3 KIỂU frame thật của `nat serve /chat/stream`), nhưng tách thành các module
nhỏ dễ quản lý/chỉnh sửa. **Không thay thế** file cũ — chạy song song được.

## Cấu trúc module

| File | Trách nhiệm |
|---|---|
| `config.py` | `AdapterConfig` — mọi knob (URL, giới hạn kích thước, **các marker định dạng frame**) |
| `frames.py` | `NatFrameParser` — phân loại/parse 3 KIỂU (thinking / tool / final) |
| `extractors.py` | `ThinkingExtractor`, `FinalAnswerStreamer` — khử trùng lặp snapshot cumulative |
| `nat_client.py` | `call_chat_stream` — client async gọi nat serve |
| `open_responses.py` | `OpenResponsesEmitter` — SSE chuẩn + object response 31 khóa + item tool |
| `util.py` | `format_block`, `ToolTracker` (state dedup tool: gọi 1 lần, gắn kết quả sau) |
| `responses_translator.py` | path `/v1/responses` (box Thinking / Tool Executed / answer riêng) |
| `openai_translator.py` | path `/v1/chat/completions` (chế độ OpenAI mặc định) |
| `app.py` | `create_app(config)` + CLI `main()` |

Giải thích chi tiết logic: xem `my_instruction/MY_ADAPTER_EXPLAINED.md`.

## Chạy

```bash
# nat serve (cửa sổ 1)
uv run nat serve --config_file my_example/openresponses/config.yml          # :8000

# adapter (cửa sổ 2) — 1 trong 3 cách:
python my_example/openresponses/run_natbridge.py --nat-url http://localhost:8000/chat/stream --port 8001
python -m natbridge --nat-url http://localhost:8000/chat/stream --port 8001   # chạy trong thư mục này
```

Trong Open WebUI: Base URL `http://localhost:8001/v1`, đặt **`api_type: responses`** để có box
"Tool Executed" riêng (chế độ OpenAI mặc định sẽ gộp tool vào box Thinking).

## Tổng quát hóa cho agent khác

Định dạng frame của nat serve được khai báo trong `AdapterConfig` (`tool_name_prefix`,
`output_marker`, `final_answer_marker`, `noise_name_markers`, …). Đổi sang agent có wire format
khác = đổi config, không phải sửa code:

```python
from natbridge.config import AdapterConfig
from natbridge.app import create_app

cfg = AdapterConfig(tool_name_prefix="ToolCall:", final_answer_marker="ANSWER:")
app = create_app(cfg)
```
