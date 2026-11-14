# Bộ SFT giữ toàn bộ lịch sử

Chuẩn bị dữ liệu từ `data/train/train.jsonl`, chưa huấn luyện mô hình.

## Bản raw-history v2

`prepare_full_history_sft.py` hiện mặc định `--input-mode raw`. Hợp đồng input
nằm trong `src/agent/sft_input.py`, độc lập với hàm prompt legacy có sửa ASR.
Giữ nguyên `audio_text` nguồn, toàn bộ lịch sử và kết quả tool quá khứ. Không sửa
tên/số/ý định, không ghép nhiều ASR, không chạy gatekeeper hủy/OOS, không tự sinh
exemplar, không đưa đáp án dự đoán bằng if/else vào input. Tool vẫn được lọc theo
app đã cài và truyền đầy đủ schema. Ngữ cảnh ngày giờ gốc được giữ, chưa chèn
bảng tính ngày hay phân tích observation từ pipeline legacy.

Nguồn hiện tại là text có nhãn trong train, không phải dữ liệu ASR nhiễu mới.
Mẫu `label.tool_calls=[]` vẫn tạo một cặp input–output kết thúc, không bị bỏ do
câu nói chứa từ hủy. Ngược lại không tự gán nhãn rỗng chỉ vì có từ "thôi"/"hủy";
mọi target lấy nguyên từ nhãn nguồn. Các bước kết thúc sau tác vụ cũng được giữ.

```powershell
python scripts/prepare_full_history_sft.py --out out/sft_raw_history_v2 --input-mode raw --max-tokens 8192
```

Bản v1 cũ dùng prompt V1, có thể tái dựng với `--input-mode legacy`.
Các mô tả helper prompt V1 bên dưới chỉ áp dụng chế độ legacy. V2 cần dùng
`RAW_SYSTEM_PROMPT` + `build_raw_user_prompt` khi đánh giá inference, không phải
chỉ bật `full_history=True` trong prompt legacy. Pipeline agent hiện tại vẫn giữ
nguyên để làm baseline A; chưa tắt gatekeeper/sửa ASR trong inference hiện hành.
Đây là thay đổi hợp đồng dữ liệu, không phải kết quả A/B hay bằng chứng tăng điểm.

Mỗi dòng có `id`, `source_id`, `messages`: system, user và assistant mục tiêu.
User prompt chứa ngữ cảnh, tool của app đã cài, toàn bộ lịch sử theo thứ tự,
và kết quả các bước đã chạy trong yêu cầu hiện tại. Trong lịch sử giữ nguyên
`role`, `audio_text`, `text`, `tool_calls`, `tool_results` cùng các trường còn lại;
chỉ bỏ đường dẫn `audio`. Không cắt lịch sử theo ký tự, không giới hạn số đối tượng
trong kết quả tool lịch sử. Các helper khác của prompt V1 vẫn được giữ.

Mỗi nhãn tool tạo một bước học. Chỉ sau bước đó, kết quả tool tương ứng mới được
đưa vào input bước kế tiếp. Bước cuối dạy `actions: []`, `is_finished: true`.
`sys.ask_missing_info` là bước kết thúc. Không sinh rationale hay nhãn mới;
`thought` để trống. Không history mining, app stripping, dữ liệu tổng hợp hay public test.

Chia nguồn trước khi tách bước, seed 42 và validation mục tiêu khoảng 10%.
Các mẫu chung audio hiện tại/lịch sử, `conversation_id` nếu có, hoặc cùng input
được gom một nhóm. Đây là nhóm nhận diện từ dữ liệu, không đảm bảo phát hiện được
mọi hội thoại liên quan khi nguồn không cung cấp định danh/audio chung.

Chạy trong môi trường có `transformers` và tokenizer Qwen:

```powershell
python scripts/prepare_full_history_sft.py --out out/sft_full_history_v1 --max-tokens 8192
```

Thư mục output phải chưa tồn tại để tránh ghi đè bản dữ liệu cũ.
Có thể dùng `--tokenizer-dir` trỏ tới thư mục tokenizer offline, kèm thư viện
`tokenizers` và `jinja2`. Model mặc định: `Qwen/Qwen3-4B-Instruct-2507`.

| File | Nội dung |
|---|---|
| `train.jsonl`, `validation.jsonl` | Các mẫu có tổng token **nhỏ hơn** 8192 |
| `train_all_lengths.jsonl`, `validation_all_lengths.jsonl` | Toàn bộ mẫu, giữ nguyên split |
| `overlength.jsonl` | Mẫu từ 8192 token trở lên của cả hai split; tra split trong `lengths.csv` |
| `lengths.csv` | Token input, tổng token, split và cờ giữ cho mỗi bước |
| `split_manifest.json` | Split của từng câu gốc |
| `report.json` | Thống kê, cấu hình và SHA256 nguồn |

Tổng token gồm chat template, system, user và đáp án assistant. Ngưỡng loại
khớp `scripts/kaggle_train_sft.py`, vốn bỏ mẫu `len >= max_len`.
Khi train bản dưới 8K cần đặt `--max_len 8192`; mặc định 4096 sẽ loại thêm nhiều mẫu.
Validation dưới 8K chỉ phản ánh phần dữ liệu vừa context; đánh giá tác vụ sau này
nên dùng nguyên câu gốc thuộc validation và báo riêng nhóm dài.

Đây là hợp đồng prompt full-history, được bật bằng
`build_user_prompt(..., full_history=True)`. Mặc định inference V1/V2 hiện tại
chưa đổi. Trước khi đánh giá hoặc triển khai adapter được train từ bộ này,
cần đồng bộ cách dựng input inference; không dùng nguyên pipeline tóm/cắt lịch sử cũ
rồi coi đó là cùng cấu hình. Bộ dữ liệu giữ nhãn gốc, chưa chứng nhận mọi nhãn
đều đúng về ngữ nghĩa hay đã replay simulator.
