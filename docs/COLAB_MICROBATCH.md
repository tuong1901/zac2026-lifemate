# Colab micro-batching

`scripts/colab_microbatch_server.py` thay thế FastAPI server một-request-một-lần cho mục đích phát triển. Server vẫn nhận API OpenAI-compatible tại `/v1/chat/completions`, nhưng gom các request LLM độc lập trong một cửa sổ rất ngắn rồi gọi GPU một lần.

## Chạy trên Colab GPU

Tải file `scripts/colab_microbatch_server.py` lên Colab hoặc clone repo. Trong runtime GPU mới:

```python
!pip install -q fastapi uvicorn pyngrok accelerate transformers bitsandbytes
```

```python
import os
os.environ["MODEL_ID"] = "Qwen/Qwen3-4B-Instruct-2507"
os.environ["BATCH_SIZE"] = "1"
os.environ["BATCH_WAIT_MS"] = "0"
os.environ["MAX_NEW_TOKENS"] = "512"
os.environ["MAX_INPUT_TOKENS"] = "4096"
os.environ["MAX_INPUT_HEAD_TOKENS"] = "3072"
# Ép checkpoint 4B tách layer thật sự giữa hai T4, đồng thời chừa bộ nhớ cho
# attention activation và KV cache.
os.environ["LOAD_IN_4BIT"] = "0"
```

Mở tunnel và khởi động server. Token ngrok được lưu trong runtime; chỉ hỏi
token khi file cấu hình chưa tồn tại. `ngrok.kill()` đóng tunnel cũ trước khi
tạo URL mới.

```python
from pathlib import Path
from getpass import getpass
from pyngrok import ngrok

if not Path("/root/.config/ngrok/ngrok.yml").exists():
    ngrok.set_auth_token(getpass("Ngrok token: "))

ngrok.kill()
public_url = ngrok.connect(8000).public_url
print(f"LLM endpoint: {public_url}/v1")

!uvicorn scripts.colab_microbatch_server:app --host 0.0.0.0 --port 8000
```

## Đo đúng hiệu quả batch

Batch chỉ xuất hiện khi nhiều request cùng chờ. Lệnh evaluation tuần tự sẽ không tạo batch. Khi tạo submission, runner cần chạy nhiều sample đồng thời; các sample vẫn phải dùng `ToolEnv` độc lập.

Trên T4 15 GB, thử lần lượt batch size 2 rồi 4. Ghi throughput, p95 latency và VRAM. Không tăng batch nếu p95 tiến gần 30 giây hoặc gây out-of-memory. Cấu hình đích A30 của BTC phải được benchmark riêng.

Greedy decoding (`temperature=0`, `do_sample=False`) được ép ở server. Sau khi đổi concurrency hoặc batch size, chạy lại một tập validation nhỏ hai lần để kiểm tra tool calls và state hash có lặp lại.
