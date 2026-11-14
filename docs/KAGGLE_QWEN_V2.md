# Fine-tune Qwen3-4B trên raw-history data v2

## Sau khi smoke chunked-math thành công nhưng mất 81 phút

Lượt thực tế: 2 updates, accumulation 8, mẫu dài nhất; train_runtime khoảng
4.870 giây và peak rank0 khoảng 10,71 GiB. Kết quả xác nhận đủ bộ nhớ nhưng quá
chậm để khuyến nghị full train. Đây không phải hoàn tất fine-tune bộ dữ liệu.

Dùng `notebooks/kaggle_qwen_v2_fast_t4.ipynb` cho lượt thử tiếp theo:

- Tạo môi trường riêng `/tmp/lifemate-qwen-cu128`, Python 3.11, torch 2.8.0 CUDA
  12.8 theo wheel chính thức PyTorch; giữ nguyên torch/kernel Kaggle hiện tại.
- Thử `--attention-mode fused`, explicit KV heads và không cho fallback math.
  Đây là ứng viên cần probe trên T4 thực tế, chưa bảo đảm kernel/VRAM/tốc độ.
- Smoke mới chỉ 2 updates, **accumulation=1**, vẫn sequence dài nhất và có optimizer
  state. Không lặp 16 microbatches/GPU chỉ để kiểm tra bộ nhớ. Full vẫn accum 8.
- Receipt ghi giây/microbatch và `ready_for_full_training`. Quá 60 giây/microbatch
  thì chặn full run. Ngưỡng chỉ loại cấu hình rõ ràng quá chậm, không cam kết cả
  2 epoch vừa một runtime. Dùng trace/progress và checkpoint để đo ETA thật.
- OUT mới `qwen_v2_fast_training`, SMOKE_OUT mới `qwen_v2_fast_smoke`; không xóa
  kết quả smoke cũ. Cache HF/data vẫn dùng được trong cùng runtime.
- Chạy notebook mới từ đầu trong runtime hiện tại, attach cùng data. Không cần
  thêm API trả phí. Python/torch wheels cần tải qua Internet và tốn thêm dung lượng
  tạm; môi trường trong /tmp không phải artifact cần tải về hoặc đưa vào Output.

Trainer mới kiểm tra receipt có cờ tốc độ, nên receipt 81 phút cũ không dùng để
bật full run. Full train vẫn khởi tạo LoRA mới; không tiếp tục từ trọng số smoke.

Lỗi được xác định từ log fast T4: probe nhận BF16 trong khi kernel memory-efficient
trên sm75 chỉ nhận Half/Float. `torch.cuda.is_bf16_supported()` mặc định tính cả
BF16 emulation nên không đủ để chọn compute dtype. Recipe mới kiểm tra compute
capability >= 8 và `including_emulation=False`; T4 (7.5) luôn dùng FP16 cho cả
probe, model/BitsAndBytes và Trainer autocast. Dtype/capability được ghi vào
fingerprint và log. Sau pull chỉ cần chạy lại cell smoke trong cùng ENV_PY,
không cài lại môi trường hoặc chuyển về chunked-math chậm.

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
| Bộ nhớ | Query-chunked SDPA + gradient checkpointing, Liger fused loss cả train lẫn eval |
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

## Sửa OOM attention trên T4

Log smoke đầu tiên trên Kaggle cho thấy SDPA cấp thêm 12,45 GiB cho attention
và OOM trước optimizer step đầu tiên. Transformers 4.57.1 bật native GQA khi
mask=None; trên T4 đường này có thể dùng math backend thay vì fused attention.
Liger fused loss không xử lý được phần cấp phát attention này.

Recipe mới dùng `src/agent/training_attention.py`: tắt native GQA trong hook
Transformers để repeat KV heads tường minh, chỉ cho phép flash/memory-efficient
SDPA, cấm fallback math và cuDNN. Trước khi load weights, mỗi rank chạy thử
attention forward/backward trên sequence 256 token. Nếu build PyTorch không có
kernel phù hợp thì dừng với thông báo rõ ràng; không fallback sang math.
Smoke dài nhất vẫn cần chạy thành công mới có thể xác nhận VRAM toàn model.
Dataset, max-length, LoRA rank và số mẫu không thay đổi.

Sau lỗi cũ, pull nhánh `test` và dùng SMOKE_OUT mới
`/kaggle/working/qwen_v2_smoke_t4_fused_v1` vì thư mục smoke cũ có run_config
dở dang. Không cần tải lại weights/data và không xóa checkpoint cũ.
Không tái sử dụng receipt trước khi đổi backend/script/PyTorch.

### Build Kaggle trả Invalid backend (bản sửa tiếp theo)

Smoke thực tế tiếp theo thất bại ở probe 256 token với `Invalid backend`, trước
khi tải weights. Do đó recipe mặc định hiện là `--attention-mode chunked-math
--query-chunk-size 128`. Không yêu cầu fused SDPA kernel trên GPU/build đó.

Attention chia **query** thành tối đa 128 vị trí mỗi lần; mỗi query vẫn được
attend tới toàn bộ key quá khứ hợp lệ. Causal/padding/additive mask và GQA được
giữ, không phải sliding window, không tách input thành các mẫu độc lập.
Checkpoint từng khối giúp không giữ score matrices của tất cả khối đến backward.
Math SDPA chỉ nhận từng khối, thay vì ma trận score 10K×10K của cả sequence.
Đổi lại chậm hơn fused attention; thời gian cả epoch phải đo trong runtime.

Notebook mới dùng `/kaggle/working/qwen_v2_smoke_t4_chunked_v2` và thêm hai cờ
trên cho cả smoke lẫn full train. Nếu dùng notebook cũ, phải cập nhật `common`
cho cả hai lượt để fingerprint khớp. Không chỉ đổi cờ lúc smoke rồi dùng recipe
cũ khi full train. Không cần cài lại torch hay xóa weights/dataset đã cache.
Mode `fused` vẫn còn để dùng ở GPU/build có kernel đã qua probe, không phải
lựa chọn phù hợp với build vừa báo lỗi. Smoke model đầy đủ vẫn phải qua trước.

Kiểm chứng bản query-chunked: 16 test đã qua trên PyTorch 2.8.0 CPU, gồm so sánh
output và gradient với dense SDPA ở float64 (atol/rtol 1e-10), GQA, causal,
padding, additive mask, noncausal và checkpoint lồng nhau. Test cũng kiểm tra
mỗi phép attention không vượt query chunk-size và khối cuối vẫn thấy toàn bộ
key hợp lệ. Đây là bằng chứng tính đúng của phép chia khối, chưa phải đo VRAM
hoặc kiểm chứng toàn bộ trainer/Liger/DDP trên GPU T4 của Kaggle.
