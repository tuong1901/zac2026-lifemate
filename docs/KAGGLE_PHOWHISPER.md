# Fine-tune PhoWhisper-large trên Kaggle với src/asr

Đây là luồng ASR riêng với model large theo yêu cầu; không dùng kaggle_train_sft.py hoặc sft_train_full.jsonl. Model large vượt giới hạn ASR <1B ghi trong brief, nên bản này phục vụ thử nghiệm cho tới khi quy chế được xác nhận lại.

## Thao tác trên Kaggle

Notebook clone trực tiếp repo public `https://github.com/tuong1901/zac2026-lifemate`, nhánh `feat-eval-LLM`. Không cần Dataset code hoặc ZIP nguồn.

1. Import `notebooks/kaggle_phowhisper_large.ipynb` vào Kaggle, bật Internet và GPU T4 x2.
2. Chuẩn bị Dataset train private (hoặc gắn Dataset train bạn đã có), giữ đúng cấu trúc:

   ```text
   train/
     train.jsonl
     audio/
       train_0001/train_0001.wav
       ...
   ```

   Cần đủ các file audio được tham chiếu ở lượt hiện tại và user turns trong history. Không cần tải public/private test hay thư mục environment. Đường dẫn audio trong train.jsonl tính từ thư mục chứa train.jsonl; nếu Dataset có cấu trúc khác thì cấu hình AUDIO_ROOT trong notebook.
3. Import `notebooks/kaggle_phowhisper_large.ipynb` bằng tính năng Import Notebook của Kaggle. Add Input Dataset train. Bật GPU (T4 hoặc P100) và Internet cho cài dependency/tải weights. Notebook hiện yêu cầu 2xT4, dùng torchrun --nproc_per_node=2 và DDP. Mỗi GPU giữ một bản model, không gộp VRAM thành một GPU.
4. Chạy cell cài dependency. Nếu trước đó đã import Transformers/PEFT khác version, restart session trước khi chạy tiếp. Chạy cell clone/cập nhật Git và kiểm tra GPU/dữ liệu. Nếu tìm được nhiều train.jsonl, điền TRAIN_JSON tường minh.
5. Cell train mặc định: vinai/PhoWhisper-large, LoRA rank=32/alpha=64, FP16, batch=1, accumulation=16, LR=1e-4, 3 epochs, SpecAugment=0.05. Có gradient checkpointing. Thời gian và VRAM chưa được đo trên GPU thật; không bảo đảm thời lượng phiên Kaggle. Nếu OOM ở batch=1, cần giảm mô hình hoặc cấu hình kỹ thuật, tăng accumulation không giảm bộ nhớ của một forward pass.
6. So sánh baseline_metrics.json và validation_metrics.json. WER là tỷ lệ: 0.10 = 10%. `wer` dùng NFC, lowercase, whitespace của src/asr/evaluate.py; `raw_wer` giữ case/dấu câu. Cùng validation, beam=1, không postprocess. Nếu fine-tune kém baseline thì giữ model gốc, đừng dùng adapter chỉ vì loss giảm. Trainer chọn checkpoint WER thấp nhất trong các epoch đã train, chưa tự chọn model gốc.
7. Chạy cell merge và kiểm tra một clip validation bằng chính UnifiedASRTranscriber; sau đó tải `phowhisper_large_finetuned.zip` trong Output. Clip kiểm tra chỉ chứng minh đường nạp model/chạy inference; không thay thế benchmark.

## Đầu ra và dùng lại

```text
/kaggle/working/asr_training/
  split_manifest.json       # nguồn clip, split, clip bị loại
  baseline_metrics.json
  validation_metrics.json
  training_summary.json
  checkpoint-N/             # adapter + optimizer/RNG để resume
  best_adapter/             # adapter được chọn theo validation WER
/kaggle/working/phowhisper-large-finetuned/  # full weights đã merge + processor
/kaggle/working/phowhisper_large_finetuned.zip
```

Sau khi giải nén model về local, có thể gọi trực tiếp:

```python
from src.asr.config import ASRConfig
from src.asr.transcribe import UnifiedASRTranscriber

asr = UnifiedASRTranscriber(ASRConfig(
    asr_engine="phowhisper",
    model_name_or_path="/duong/dan/phowhisper-large-finetuned",
    batch_size=1,
))
print(asr.transcribe("/duong/dan/audio.wav"))
```

Hoặc đặt `ASR_ENGINE=phowhisper` và `ASR_MODEL_NAME_OR_PATH` thành thư mục model đã merge trong .env. Không trỏ vào best_adapter bằng pipeline hiện tại; phải merge trước. Thư mục local model phải tồn tại để không bị diễn giải thành tên Hugging Face.

Resume: giữ toàn bộ checkpoint-N và split_manifest.json, đặt RESUME trong notebook thành đường dẫn checkpoint; giữ cùng model, seed, data paths và tham số training. Adapter best_adapter chỉ đủ merge/inference, không chứa optimizer để tiếp tục đúng trạng thái. Lưu version notebook với outputs hoặc tải checkpoint trước khi phiên Kaggle kết thúc. Sau khi train xong, chuẩn bị đủ model/dependency local để inference offline.

## Tách dữ liệu và giới hạn

Chỉ đọc train JSONL và audio được liệt kê. Những hội thoại dùng chung đường dẫn audio được nối cùng nhóm trước khi split. Trùng đường dẫn audio được loại trong mỗi split; nhãn khác nhau cho cùng đường dẫn bị báo lỗi. Các bản sao audio có tên file khác nhau hoặc cùng người nói chưa được phát hiện tự động; nếu dữ liệu có metadata người nói, cần split theo người nói để đánh giá tổng quát hóa mạnh hơn.

Audio dài hơn 30 giây, audio rỗng hoặc target hơn 448 token bị loại và ghi vào manifest, không bị cắt âm thầm. Audio nhiều kênh chuyển mono và resample về 16kHz. Nhãn giữ nguyên văn, không qua regex sửa câu. Validation khoảng 10% theo nhóm, không nhất thiết đúng 10% số clip.

Trainer/notebook mới đã kiểm tra cú pháp và unit tests của phần chuẩn bị dữ liệu. Chưa chạy fine-tune, merge hoặc inference với weights large trên GPU trong phiên phát triển này. Cơ chế training dùng API Seq2SeqTrainer và LoRA; tham khảo chính thức: https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/trainer và https://github.com/huggingface/peft.

## Cập nhật code từ public

Sau khi tôi push lên nhánh feat-eval-LLM, chạy lại cell setup để `git pull --ff-only`. Nếu có sửa local trên Kaggle, notebook giữ nguyên và không tự pull; bạn cần commit hoặc stash trước. Không dùng reset --hard. Nếu module đã import trong kernel, restart session để nạp bản mới. Train/merge chạy trong subprocess nên đọc mã hiện tại trên ổ đĩa.

Có thể sửa trực tiếp `/kaggle/working/lifemate-asr/src/asr/finetune.py` để thử nghiệm. Những sửa đổi trên Kaggle không tự đồng bộ về GitHub/máy local; cần commit/push riêng hoặc tải patch về. Đầu ra weights vẫn đóng ZIP để tải về, chỉ phần lấy mã nguồn đã chuyển hoàn toàn sang Git.
## Model Drive và hai GPU T4

Notebook có cell gdown tải model của người dùng từ URL bạn nhập riêng khi chạy cell, sau đó in config.json/adapter_config.json. Chưa xác minh được nội dung folder từ môi trường phát triển; không tự khẳng định đó là model medium hay large. Model này được lưu riêng, chưa tự dùng làm base student. Cần checkpoint thật của đội tham khảo, không chỉ script train, trước khi dùng model thứ hai.

Train dùng DDP hai process: batch/GPU=1, accumulation=16, global effective batch=32 clips/optimizer step. Chọn T4 x2 và restart session nếu torch đã import khi chỉ GPU0 được hiển thị. DDP phân chia dữ liệu/gradient; file metrics, manifest và processor chỉ được ghi ở rank 0. Chưa đo utilization/throughput thực tế; không bảo đảm luôn 100% GPU.

Không thể lấy lại dataset gốc từ weights. Với audio có nhãn chuẩn, giữ gold audio_text; không nhân đôi cùng audio bằng nhãn model để giả tăng dữ liệu. Với audio bổ sung chưa nhãn, có thể dùng hai model tạo transcript rồi lọc chất lượng, bảo đảm không trùng holdout. Chưa tạo pseudo-label hoặc thêm bất kỳ public/private test sample nào vào train. Chờ người dùng xác định nguồn audio bổ sung/dataset và checkpoint thứ hai để nối bước này.