# Bàn giao triển khai LifeMate — 2026-11-11

Đã triển khai các phần mã nguồn và chuẩn bị dữ liệu theo CODEX_HANDOVER_INSTRUCTIONS.md. Chưa huấn luyện GPU, chưa có adapter thật hoặc điểm leaderboard mới; mục tiêu 55–65 không phải kết quả đã đo.

## Thay đổi

- PhoWhisper không còn prompt tên riêng cố định; decoding tất định, beam=5, no_repeat_ngram_size=3. Transformers dùng `condition_on_prev_tokens=False` (không phải `condition_on_previous_text`). Chưa đo chất lượng/tốc độ ASR trên GPU.
- Lexicon chỉ chuẩn hóa Unicode NFC, khoảng trắng và ZaloPay/MoMo/VTV. Chế độ aggressive cũ giữ API tương thích nhưng chuyển sang lexicon an toàn. `clean_asr_transcript` ở prompt cũng dùng cùng chuẩn hóa, tránh regex cũ sửa lại câu sau ASR. Package ASR tải module theo nhu cầu để xử lý văn bản không bắt buộc NumPy/PyTorch.
- `build_date_table` tính hôm qua/hôm nay/mai/kia/+3 ngày, thứ hai–chủ nhật tuần này/tuần sau, giữ timezone của sample và xử lý qua tháng/năm; bảng được đưa vào prompt đang chạy.
- GuardedToolEnv kiểm tra ID theo `x-ref` của schema. ID phải xuất hiện trong kết quả công cụ thành công đúng app, kể cả history. Khi có reader an toàn không yêu cầu tham số, tự gọi reader và hoãn WRITE để model chọn lại; không tự đoán hoặc thay ID. Khi reader cần query, trả phản hồi để model lập lệnh đọc theo yêu cầu thực tế. Các call đọc chèn thêm vẫn đi qua budget/validation và nằm trong log ToolEnv. Bảo vệ áp dụng cho các reference cấp đầu có x-ref; chưa bao phủ reference lồng nhau, mọi số tài khoản/điện thoại tự do hoặc mọi schema không có metadata.
- Synthetic: thích nghi 11 bộ template từ file tham khảo local `downloaded_minggu/train+public_test_solution_synth.py` của minggu151623. Chỉ mượn context/time từ train; toàn bộ state/history mới tự tạo. Không mở bất kỳ state.json hoặc public_test sample nào. Mỗi sample chứa initial states sau history, tool results và final hash; chạy lại bằng Simulator để đối chiếu cả results và hash. Có giới hạn số lần thử để không treo.
- SFT dùng cùng SYSTEM_PROMPT và build_user_prompt với inference, schema và context theo app cài đặt. Mỗi bước chỉ thấy quan sát trước đó, đầu ra JSON actions/is_finished khớp ReActAgent. Gỡ app thực sự khỏi context trước khi gán nhãn từ chối. Giữ history mining và trộn synthetic. `source_id` hỗ trợ chia train/validation theo nguồn để tránh rò rỉ giữa các bước của cùng tình huống.
- Trainer: rank 32/alpha 64, QLoRA NF4, batch 1, accumulation 16, 3 epochs; sửa số optimizer steps bằng ceil, chuẩn hóa nhóm tích lũy cuối, pad theo tokenizer, seed, mixed precision/gradient scaling và kiểm tra artifact adapter cuối. Một process dùng device_map=auto để phân bố model; không phải huấn luyện DDP hai GPU.

## Dữ liệu đã tạo

- `data/synth_11_apps.jsonl`: 1.100 samples, đúng 100/app, tất cả replay results/hash khớp.
- `data/sft_train_full.jsonl`: 14.354 quyết định ReAct, từ 1.545 mẫu train + 2.981 history-mined + 468 OOS + 1.100 synthetic = 6.094 tình huống. Số dòng khác ước tính ~6.000 trong brief vì tách quyết định theo quan hệ nhân quả.
- Dữ liệu được .gitignore loại khỏi Git; cần tải riêng lên Kaggle Dataset private.

Tái tạo từ thư mục gốc:

```powershell
python scripts/synth_unseen_apps.py --n 1100 --seed 42
python scripts/build_sft_dataset.py --train data/train/train.jsonl --synth data/synth_11_apps.jsonl --out data/sft_train_full.jsonl
```

## Kaggle

Mở `notebooks/kaggle_sft_handover.ipynb`. Đính kèm private Dataset chứa script `scripts/kaggle_train_sft.py` và `data/sft_train_full.jsonl`; chọn GPU 2x T4 hoặc P100 và bật Internet cho bước cài dependency/tải model khi training. Notebook tự tìm hai file và chạy cấu hình nêu trên; không cần tải môi trường test lên Kaggle.

Đầu ra cần tải về: `/kaggle/working/lifemate_final_adapter.zip`, chứa `adapter_config.json`, `adapter_model.safetensors` và tokenizer. Đường dẫn trước đóng gói: `/kaggle/working/out/lora_adapter/final_adapter`.

Đọc số mẫu bị loại vì dài hơn `max_len=4096` trong log. Nếu mất nhiều mẫu, cần đánh giá context/VRAM trước khi tăng max_len hoặc rút prompt. Không coi loss train là điểm thi; phải đánh giá checkpoint theo tập độc lập, nhóm theo source_id và thử leave-apps-out. Model/base tokenizer và dependency phải được chuẩn bị local cho inference offline; notebook này chỉ thực hiện training, chưa tích hợp adapter vào Docker submit.

## Kiểm chứng và giới hạn

- 86 kiểm thử chạy thành công: regression agent/execution/official rules và 8 test mới về NFC/tên người, ngày qua năm, provenance ID, hoãn write, không bịa query, replay 11 app và không rò rỉ quan sát tương lai vào SFT.
- Full discovery hiện không hoàn thành vì Python portable chưa có torch/openai; các kiểm thử backend Transformers/Colab và client SDK không nằm trong 86 test đã chạy. Không đổi test cũ để che thiếu dependency.
- Compile Python cho src/scripts và kiểm tra JSON/cell code của notebook. Chưa chạy huấn luyện, tokenizer/model thật hoặc benchmark âm thanh GPU.
- `check_compliance.py` còn báo hai mục có sẵn trong thư mục tham khảo `downloaded_minggu`: model 14B trong rationale và đọc train state trong synth gốc. Không thực thi hay đóng gói hai file đó vào runtime nộp bài; script synth mới không dùng các đoạn đó. Chưa tuyên bố toàn bộ repo qua compliance scanner.
- Không truy cập tài khoản Kaggle từ môi trường này; người dùng cần mở notebook và chạy GPU để sinh adapter. Cần đo score/latency bằng weights thật trước khi nộp bài.