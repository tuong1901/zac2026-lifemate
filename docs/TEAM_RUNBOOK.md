# LifeMate — Team Runbook

## Trạng thái thực tế

Hiện tại **agent chạy được với Gold Text**. Luồng audio-to-action chưa hoàn chỉnh vì ASR chưa được bàn giao và `src/asr/transcribe.py` mới là khung gọi Faster-Whisper; khi thiếu dependency nó còn có mock fallback. Không dùng mock này để benchmark hoặc tạo submission.

```text
Gold Text (đang chạy được) -> Agent -> ToolEnv -> tool_calls + state_hash
Audio -> ASR (đang chờ) -> transcript -> Agent -> ToolEnv -> tool_calls + state_hash
```

Hai resolver agent bật mặc định:

- Khóa cửa: đối chiếu câu hiện tại hoặc alert history với `door_lock.list_locks`.
- Task theo thứ tự: hiểu tham chiếu như “việc thứ hai con mới kể”.

`--structured-router` là nhánh thử nghiệm. Không bật khi tạo submission trước khi có A/B end-to-end trên holdout lớn.

Mốc benchmark sạch gần nhất trên 20 mẫu (chỉ để theo dõi phát triển, không phải điểm leaderboard): Gold Text 27.5/100, state match 8/20, schema penalty 0.

## Cài môi trường

Tại thư mục gốc repo, dùng Python 3.11:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn activate, gọi trực tiếp `.\venv\Scripts\python.exe` trong các lệnh bên dưới.

## Dữ liệu đầu vào

Bắt buộc cho development Gold:

```text
D:\Download\train+public_test\train+public_test\
  train\train.jsonl
  public_test\public_test.jsonl
```

Thư mục `C:\Users\Admin\Downloads\MultiASR_Results` (nếu có) chỉ là transcript thử nghiệm do một lần chạy ASR trước tạo ra — **không phải ASR model**. Chỉ dùng nó để so sánh agent với transcript lỗi; không xem đó là artifact để nộp bài.

## Chạy agent với Gold Text ngay bây giờ

Endpoint Colab/ngrok chỉ được phép dùng để phát triển. Submission cuối phải chạy LLM local trong Docker offline.

```powershell
.\venv\Scripts\python.exe scripts\run_with_asr.py `
  --base_url https://<server>/v1 `
  --limit 20 `
  --eval_source gold `
  --trace_file data\gold_trace.jsonl

.\venv\Scripts\python.exe scripts\analyze_tool_plans.py data\gold_trace.jsonl
```

Debug một mẫu:

```powershell
.\venv\Scripts\python.exe scripts\run_with_asr.py `
  --base_url https://<server>/v1 `
  --sample_ids train_0456 `
  --eval_source gold `
  --trace_file data\debug_trace.jsonl
```

## Hợp đồng bàn giao ASR

Người phụ trách ASR cần bàn giao một trong hai phương án sau:

1. Model/package chạy offline trong Docker, kèm lệnh gọi audio → text; hoặc
2. Transcript JSONL đầy đủ để agent tích hợp tạm thời (chỉ hợp lệ để đánh giá nội bộ nếu không có model trong image nộp bài).

Mỗi record transcript cần có:

```json
{"id":"<sample_id>","transcript":"<câu nói mới>","history_user_transcripts":[{"history_index":0,"transcript":"<lượt user trong history>"}]}
```

Kèm báo cáo: coverage/missing/duplicate ID bằng 0 cho split tương ứng, phiên bản model, kích thước model và thời gian suy luận. Khi có ASR, chạy `--eval_source both` trước; agent dừng nếu transcript bị thiếu, không âm thầm thay bằng Gold Text.

## Chunking và router

```powershell
.\venv\Scripts\python.exe scripts\build_structured_chunks.py
.\venv\Scripts\python.exe scripts\eval_structured_router.py
```

Artifact ở `data\structured_chunks\` và bị gitignore. Chỉ giữ router khi tăng cả workflow accuracy và điểm end-to-end, không chỉ retrieval accuracy.

## Submission

**Chưa tạo submission speech-based khi ASR chưa có model/pipeline chạy offline.** Khi ASR đã được tích hợp, smoke test tối đa 3 mẫu trước:

```powershell
.\venv\Scripts\python.exe scripts\run_with_asr.py `
  --make_submission `
  --mode hf `
  --model <duong_dan_model_local> `
  --asr_model <ten_model_asr_da_ban_giao> `
  --limit 3 `
  --output_file output\smoke_submission.jsonl
```

Sau smoke test, chạy đủ public test (script có resume theo JSONL output) và validate:

```powershell
.\venv\Scripts\python.exe scripts\validate_submission.py `
  output\submission.jsonl `
  --data_root "D:\Download\train+public_test\train+public_test"
```

## Phân công ngay lúc này

| Vai trò | Việc cần bàn giao |
|---|---|
| ASR | Model offline hoặc JSONL theo hợp đồng trên; báo coverage, runtime, size. |
| Agent | Cải thiện và đo Gold trước; chỉ đo ASR sau khi có transcript thật. |
| Data/chunking | Script tái tạo chunk, cross-validation retrieval; không tự bật router. |
| MLOps | Docker offline chứa cả LLM và ASR, smoke submission, validate schema. |

Mọi thay đổi agent phải giữ schema penalty bằng 0 và được A/B trên Gold trước. Khi ASR đến, lặp lại A/B trên transcript ASR để tách rõ lỗi ASR khỏi lỗi agent.
