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
## Khôi phục khi DDP timeout sau khi đã chạy hết các bước

Nếu log đã tới 423/423 rồi NCCL timeout cuối quá trình, không xóa thư mục asr_training hoặc chạy lại từ đầu. Sau git pull, dùng một process để chọn checkpoint có eval_wer thấp nhất trong các checkpoint còn đủ adapter + trainer_state.json:

```python
subprocess.run([sys.executable, '-m', 'src.asr.finetune', 'recover',
    '--run-dir', '/kaggle/working/asr_training',
    '--out', '/kaggle/working/phowhisper-large-finetuned'], check=True)
```

Lệnh này không train lại và không đo WER mới; nó đọc WER đã lưu, merge checkpoint tốt nhất còn trên đĩa và ghi recovery_summary.json. Không khẳng định khôi phục được khi chưa thấy checkpoint thực tế. Sau đó chạy cell nhận dạng thử và đóng ZIP; bỏ qua cell metrics yêu cầu validation_metrics.json/best_adapter nếu lần chạy cũ chưa xuất các file đó.

Trainer mới tắt load_best_model_at_end tự động, chờ mọi rank hoàn tất rồi chọn checkpoint ở rank 0 và broadcast cùng đường dẫn cho mọi rank trước khi load. Đây là biện pháp tránh nhánh load-best khác nhau giữa các rank trong lúc checkpoint còn đang được ghi; chưa tái hiện/xác minh lỗi này bằng DDP GPU thật tại môi trường phát triển. Log timeout chỉ chứng minh collective bị lệch/chờ quá hạn, chưa đủ kết luận mọi nguyên nhân NCCL đã được xử lý.
## Thử giảm WER sau lượt chạy 13,91%

Mở `notebooks/kaggle_phowhisper_wer_experiments.ipynb`, dùng cùng phiên Kaggle còn model đã merge và thư mục asr_training. Nếu phiên đã hết, gắn Output cũ làm Input và sửa BASE_RUN/BASE_MODEL; các đường dẫn audio trong split_manifest.json phải còn đúng. Giữ bản model/checkpoint cũ trước khi thử nghiệm.

1. Setup clone/pull public và bật T4 x2. Không cần model Drive.
2. Đánh giá model 13,91% trên chính danh sách validation cũ với ba cấu hình: beam1/ngram3 (mốc cũ), beam1/ngram0 (chỉ bỏ chặn lặp), beam5/ngram0 (chỉ đổi beam so với cấu hình thứ hai). Hai GPU chia danh sách audio, không chạy huấn luyện. Có thể mất đáng kể thời gian cho cả ba lượt, chưa đo trên GPU thật. Beam5 không bảo đảm tốt hơn.
3. Đọc report.json và 15 câu lỗi nhiều nhất. Có transcript/reference từng clip để kiểm tra lỗi tên riêng, số và dấu câu. WER corpus, NFC/lowercase/whitespace giống trainer; raw_wer giữ case/dấu câu. Đây là tuning trên validation, không phải kết quả test độc lập.
4. Nếu cần train thử, đặt RUN_TRAIN=True và chọn PROFILE:
   - speed_only: LR=1e-4, accumulation=16, thêm speed +/-10%; global batch 32, giữ những cấu hình khác như cũ.
   - reference_like: LR=5e-4, accumulation=8, speed +/-10%; global batch 16. Đây là lịch gần mặc định script minggu, không khẳng định là cấu hình sinh ra 8,4%. Chỉ thử sau khi xem speed_only hoặc khi có đủ ngân sách GPU.
5. Mỗi profile khởi tạo từ base Hugging Face, 3 epochs, cùng split cũ qua --split-manifest. Không resume optimizer từ lượt cũ với LR mới. Output riêng trong asr_wer_experiments, không ghi đè model 13,91%. run_config.json lưu cấu hình; resume bị chặn nếu cấu hình thay đổi. Speed chỉ thay đổi audio train; validation giữ nguyên. Nếu làm chậm khiến audio vượt 30s thì bỏ augmentation của clip đó, không cắt lời nói.
6. Sau train, notebook merge sang thư mục riêng và benchmark lại cùng ba cấu hình. Chỉ báo cải thiện khi so cùng manifest và decoding. Dữ liệu synthetic tool-call và nhãn do hai ASR đoán không được thêm vào audio_text chuẩn.

Code mới gồm khôi phục checkpoint/DDP cuối run đang chuẩn bị từ đợt sửa trước. Đã kiểm tra cú pháp và 11 unit test CPU/stdlib; chưa chạy benchmark/augmentation/huấn luyện trên CUDA tại máy phát triển, chưa có WER mới. T4 x2 không bảo đảm utilization 100%; DDP không cộng VRAM thành một GPU lớn. Nếu WER kém hơn, giữ model cũ. Không tự động thay cấu hình pipeline inference theo config thắng cuộc; cần áp dụng đúng beam/ngram đã đo khi dùng model sau này.
### Tận dụng runtime/model cũ

Không restart session và không xóa Output cũ. Có thể đánh giá decoding ngay trên model đã merge mà không train. Notebook thí nghiệm nay mặc định profile warm_start khi người dùng bật RUN_TRAIN=True: chọn adapter cũ có WER thấp nhất còn lưu, tải đúng base + adapter với is_trainable=True, khởi tạo optimizer/schedule mới và học thêm 1 epoch, LR=2e-5, speed +/-10%. Không dùng --resume với cấu hình LR mới để giả tiếp tục cùng lịch học. --init-adapter kiểm tra base khớp; kiến trúc adapter được giữ nguyên theo checkpoint.

Chỉ chạy train nếu decoding chưa đủ tốt và còn thời gian GPU. So WER trước/sau trên cùng split và decoding. Nếu warm-start tệ hơn, dùng lại model 13,91%. Nó không chắc đạt 8%. Các profile speed_only/reference_like vẫn có sẵn để thử từ base; không chạy tất cả một lượt. Đường dẫn run mới asr_wer_experiments/warm_start, không ghi đè asr_training. Cần giữ cả checkpoint adapter (để warm-start) và full model đã merge (để inference).
### Kết quả decoding do người dùng chạy trên 511 clip

Cùng manifest: beam1/ngram3 WER=13,8398% (1011 lỗi/7305 từ), beam1/ngram0=12,3477% (902 lỗi), beam5/ngram0=11,8138% (863 lỗi). Cải thiện 2,026 điểm phần trăm giữa hai đầu của phép so cùng lượt; không có cập nhật weights. Không dùng đây làm kết quả test độc lập. Pipeline PhoWhisper nay mặc định ngram=0, beam=5; có thể đặt ASR_NO_REPEAT_NGRAM_SIZE để thử lại. Merge mới lưu beam5/ngram0 vào generation_config. Notebook WER warm-start đánh giá trực tiếp bằng --eval-beams 5 --eval-ngram 0 để so với mốc 11,81%. Đánh giá beam5 có thể chậm hơn greedy, chưa đo thời gian lượt train mới.