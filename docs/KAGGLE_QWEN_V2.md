# Fine-tune Qwen3-4B trên raw-history data v2

Notebook: `notebooks/kaggle_qwen_v2.ipynb`.
Nguồn dữ liệu: `out/sft_raw_history_v2_all.zip` (4.397 train, 472 validation).
Đây là Qwen3-4B-Instruct-2507; không phải Qwen3.5, không phải model ASR.

## Chạy trên Kaggle

1. Tạo một **private Dataset** chứa `sft_raw_history_v2_all.zip`.
2. Tạo notebook, Import Notebook bằng file `kaggle_qwen_v2.ipynb` trong repo.
3. Add Input Dataset ở bước 1. Bật Internet và GPU **T4 x2**.
4. Chạy lần lượt các cell. Cài thư viện, clone public branch `test`, tìm data,
   xác nhận đủ 4.397/472 mẫu và hai GPU.
5. Cell smoke thực hiện 2 optimizer updates trên mẫu train dài nhất rồi evaluate
   mẫu validation dài nhất. Đây là phép kiểm tra kernel, gradient và VRAM thật;
   chưa được chạy trên GPU trong lúc chuẩn bị code. Nếu lỗi/OOM, giữ log và dừng.
6. Khi smoke thành công, cell tiếp theo mới chạy full training. Script kiểm tra
   receipt khớp model, data hash, script, GPU và cấu hình.
7. Chạy cell đóng gói; **Save Version**, tải cả adapter và recovery trước khi hết runtime.

Không cần tải bộ audio hoặc public test lên để chạy SFT này. Dữ liệu giữ nguyên
text nguồn, lịch sử, tool call/result; không sửa tên/số/ý định, không bỏ câu hủy.
Nhánh inference hiện tại không thay đổi. Không có API trả phí hoặc upload tự động.

## Cấu hình thử nghiệm đầu tiên

| Thành phần | Giá trị |
|---|---|
| Base | Qwen/Qwen3-4B-Instruct-2507 |
| Train/validation | 4.397 / 472 quyết định, không gộp validation vào train |
| Độ dài | 12.288 token; dữ liệu đã đo tối đa 10.251 |
| LoRA | rank 16, alpha 32, attention + MLP |
| Quantization | 4-bit NF4, double quant |
| GPU | DDP: mỗi T4 giữ một model replica, không cộng VRAM thành 32 GB |
| Compute | FP16 trên T4; BF16 nếu GPU hỗ trợ |
| Batch | 1/GPU × 2 GPU × accumulation 8 = 16 |
| Epoch / LR | 2 / 1e-4 |
| Bộ nhớ | Gradient checkpointing, SDPA, Liger fused loss cả train lẫn eval |
| Checkpoint | Mỗi 50 optimizer updates, giữ 3 checkpoint gần nhất |
| Evaluation | Loss trên toàn bộ validation mỗi epoch và cuối run |

Đây là cấu hình xuất phát, không phải cấu hình đã được đo đạt điểm tốt nhất.
T4 không dùng FlashAttention-2 trong recipe này. Smoke test dùng sequence dài
nhất, không giảm max length hay bỏ mẫu nếu hết VRAM. Tốc độ, thời lượng và peak
VRAM phải lấy từ runtime thực tế. Model nhỏ hơn hoặc phần cứng lớn hơn là lựa
chọn cần đánh giá nếu smoke không vừa bộ nhớ; không tuyên bố chắc chắn 12K vừa T4.

## Loss và tính đầy đủ dữ liệu

`src/agent/sft_encoding.py` dùng đúng chat template của tokenizer. Chỉ tính loss
trên JSON assistant và token kết thúc; system, user, lịch sử và kết quả tool
được mask -100. Không train mô hình chép lại prompt. Nếu vượt max length,
chat-template prefix sai, không còn token assistant hoặc trùng source giữa
train/validation, script báo lỗi thay vì tự cắt/bỏ dữ liệu.

Validation loss chỉ đo dự đoán token với input đã có kết quả tool đúng ở quá khứ.
Nó không phải điểm state hash/tool-calling, không đo được toàn bộ lỗi tích lũy
trong execution. Adapter cuối được xuất; chưa tự chọn adapter tốt nhất theo điểm thi.
Lượt đánh giá tác vụ sau train cần dùng đúng raw-history v2 input contract.

## Lưu và tiếp tục

- `/kaggle/working/qwen_v2_training/final_adapter`: adapter + tokenizer, cần base model để inference.
- `qwen_v2_final_adapter.zip`: bản nhỏ để tải về dùng inference.
- `qwen_v2_training_recovery.zip`: có checkpoint, optimizer, scheduler, RNG và Trainer state.
- `metrics.json`, `run_config.json`, `trainer_state.json`: cấu hình và kết quả, không tự upload ra ngoài.

Để resume, giải nén recovery về OUT, giữ cùng data/model/cấu hình và smoke
receipt tương ứng, đặt `RESUME=True`. Adapter riêng không đủ để resume đúng
optimizer. Nếu smoke lỗi sau khi ghi file, chọn SMOKE_OUT mới khi chạy lại;
không xóa toàn bộ working directory. Checkpoint trong `/kaggle/working` không
phải bản sao lưu ngoài runtime; tải recovery hoặc Save Version kịp thời.

Dependencies của notebook được pin: transformers 4.57.1, peft 0.17.1,
accelerate 1.10.1, bitsandbytes 0.48.1, liger-kernel 0.6.2. Giữ PyTorch CUDA
của Kaggle. Pin version giúp tái lập nhưng chưa thay thế phép kiểm tra GPU thật.
Code tham khảo API chính thức: Hugging Face Trainer/PEFT và Liger Qwen3 forward
(`skip_logits=True` để validation không cấp phát tensor logits toàn vocabulary).
