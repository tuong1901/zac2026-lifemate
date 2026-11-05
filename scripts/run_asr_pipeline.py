"""
run_asr_pipeline.py — Script dòng lệnh (CLI) chạy trọn gói Pipeline ASR & Post-Processing.
Tối ưu hóa chạy trên Kaggle GPU (T4 / P100 / A100) hoặc Local máy cá nhân.

Các chế độ chạy chính:
1. Đánh giá nhanh trên tập Benchmark (200 mẫu):
   python scripts/run_asr_pipeline.py --mode benchmark

2. Đánh giá FULL toàn bộ 1.545 mẫu tập Train:
   python scripts/run_asr_pipeline.py --mode train

3. Gỡ băng FULL toàn bộ 634 mẫu tập nộp bài Public Test:
   python scripts/run_asr_pipeline.py --mode public_test

4. Tùy chọn ASR Engine & Bật/Tắt các tầng Post-Processing:
   python scripts/run_asr_pipeline.py --asr_engine phowhisper --model_name vinai/phowhisper-medium --batch_size 16 --enable_t1
"""

import os
import sys
import argparse
from pathlib import Path

# Đảm bảo in tiếng Việt chuẩn trên Windows / Linux
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.asr.config import ASRConfig, get_asr_config
from src.asr.batch_runner import ASRBatchRunner


def resolve_data_root(preferred_path: str = None) -> Path:
    candidates = [
        preferred_path,
        os.getenv("DATA_ROOT"),
        "/kaggle/input/datasets/n107hoangtuong/zalo-competion/train+public_test",
        "/kaggle/input/zalo-ai-challenge-2026/train+public_test",
        "/kaggle/input/train+public_test",
        "D:/Download/train+public_test/train+public_test",
        "C:/Users/Admin/Downloads/train+public_test",
        str(PROJECT_ROOT / "data"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError(
        "Không tìm thấy thư mục DATA_ROOT! Hãy chỉ định --data_root hoặc cấu hình biến DATA_ROOT trong file .env."
    )


def main():
    parser = argparse.ArgumentParser(description="Chạy Pipeline ASR & Post-Processing cho ZAC 2026 LifeMate")
    parser.add_argument("--mode", choices=["benchmark", "train", "public_test", "custom"], default="benchmark",
                        help="Chế độ: 'benchmark' (200 mẫu), 'train' (FULL 1.545 mẫu), 'public_test' (FULL 634 mẫu nộp bài), hoặc 'custom'")
    parser.add_argument("--data_root", type=str, default=None, help="Đường dẫn thư mục chứa train và public_test")
    parser.add_argument("--asr_engine", choices=["phowhisper", "phoasr", "faster_whisper", "whisper"], default=None,
                        help="Lựa chọn ASR Engine (ghi đè .env)")
    parser.add_argument("--model_name", type=str, default=None, help="Tên model HuggingFace hoặc path checkpoint")
    parser.add_argument("--batch_size", type=int, default=None, help="Kích thước batch xử lý trên GPU (VD: 8, 16, 32)")
    parser.add_argument("--device", type=str, default=None, help="Thiết bị: 'cuda', 'cpu', 'auto'")
    parser.add_argument("--enable_t1", action="store_true", default=None, help="Bật Tầng 1 (Lexicon/Rule)")
    parser.add_argument("--enable_t2", action="store_true", default=None, help="Bật Tầng 2 (Contextual ViT5)")
    parser.add_argument("--enable_t3", action="store_true", default=None, help="Bật Tầng 3 (ViBERT-capu Punctuation)")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số mẫu chạy thử (VD: 10, 50, bỏ trống để chạy FULL)")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Ghi đè file kết quả cũ nếu đã tồn tại")
    parser.add_argument("--resume", dest="overwrite", action="store_false", help="Bỏ qua các mẫu đã xử lý trước đó (resume)")
    parser.add_argument("--output_file", type=str, default=None, help="Đường dẫn file jsonl đầu ra nếu chạy custom")
    parser.add_argument("--input_file", type=str, default=None, help="Đường dẫn file jsonl đầu vào nếu chạy custom")
    args = parser.parse_args()

    # 1. Khởi tạo Config và ghi đè từ CLI nếu có
    cfg = get_asr_config()
    if args.asr_engine:
        cfg.asr_engine = args.asr_engine
    if args.model_name:
        cfg.model_name_or_path = args.model_name
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.device:
        cfg.device = args.device
    if args.enable_t1 is not None:
        cfg.enable_lexicon_correction = args.enable_t1
    if args.enable_t2 is not None:
        cfg.enable_contextual_correction = args.enable_t2
    if args.enable_t3 is not None:
        cfg.enable_punctuation_restoration = args.enable_t3

    # 2. Xác định đường dẫn Data Root
    try:
        data_root = resolve_data_root(args.data_root)
        cfg.data_root = str(data_root)
    except FileNotFoundError:
        data_root = PROJECT_ROOT / "data"
        cfg.data_root = str(data_root)

    out_dir = Path(cfg.output_dir) if Path(cfg.output_dir).is_absolute() else PROJECT_ROOT / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Phân chia kịch bản chạy
    runner = ASRBatchRunner(cfg)

    if args.mode == "benchmark":
        bench_candidates = [
            PROJECT_ROOT / "data" / "val_benchmark_200.jsonl",
            data_root / "val_benchmark_200.jsonl",
            data_root / "train" / "train.jsonl"
        ]
        bench_file = next((f for f in bench_candidates if f.exists()), None)
        if not bench_file:
            print("[Lỗi] Không tìm thấy file benchmark val_benchmark_200.jsonl hoặc train.jsonl!")
            return

        out_name = f"benchmark_{cfg.asr_engine}_{Path(cfg.model_name_or_path).name.replace('/', '_')}.jsonl"
        out_path = out_dir / out_name
        audio_base = data_root / "train" if (data_root / "train").exists() else data_root

        print(f"\n🚀 BẮT ĐẦU CHẠY ĐÁNH GIÁ BENCHMARK (200 MẪU):")
        print(f"  - Input:      {bench_file}")
        print(f"  - Audio dir:  {audio_base}")
        print(f"  - Output:     {out_path}")
        runner.run_on_dataset(
            input_jsonl_path=bench_file,
            output_jsonl_path=out_path,
            audio_base_dir=audio_base,
            limit=args.limit,
            transcribe_history=True,
            overwrite=args.overwrite
        )

    elif args.mode == "train":
        train_file = data_root / "train" / "train.jsonl"
        if not train_file.exists():
            train_file = data_root / "train.jsonl"
        if not train_file.exists():
            print(f"[Lỗi] Không tìm thấy file train.jsonl tại {data_root}!")
            return

        out_name = f"train_{cfg.asr_engine}_{Path(cfg.model_name_or_path).name.replace('/', '_')}_for_agent.jsonl"
        out_path = out_dir / out_name
        audio_base = data_root / "train" if (data_root / "train").exists() else data_root

        print(f"\n🚀 BẮT ĐẦU CHẠY ĐÁNH GIÁ VÀ GỠ BĂNG FULL TOÀN BỘ TẬP TRAIN (1.545 MẪU):")
        print(f"  - Input:      {train_file}")
        print(f"  - Audio dir:  {audio_base}")
        print(f"  - Output:     {out_path}")
        runner.run_on_dataset(
            input_jsonl_path=train_file,
            output_jsonl_path=out_path,
            audio_base_dir=audio_base,
            limit=args.limit,
            transcribe_history=True,
            overwrite=args.overwrite
        )

    elif args.mode == "public_test":
        test_file = data_root / "public_test" / "public_test.jsonl"
        if not test_file.exists():
            test_file = data_root / "public_test.jsonl"
        if not test_file.exists():
            print(f"[Lỗi] Không tìm thấy file public_test.jsonl tại {data_root}!")
            return

        out_name = f"public_test_{cfg.asr_engine}_for_agent.jsonl"
        out_path = out_dir / out_name
        audio_base = data_root / "public_test" if (data_root / "public_test").exists() else data_root

        print(f"\n🚀 BẮT ĐẦU GỠ BĂNG FULL TOÀN BỘ PUBLIC_TEST (634 MẪU) ĐỂ NỘP BÀI:")
        print(f"  - Input:      {test_file}")
        print(f"  - Audio dir:  {audio_base}")
        print(f"  - Output:     {out_path}")
        runner.run_on_dataset(
            input_jsonl_path=test_file,
            output_jsonl_path=out_path,
            audio_base_dir=audio_base,
            limit=args.limit,
            transcribe_history=True,
            overwrite=args.overwrite
        )

    elif args.mode == "custom":
        if not args.input_file or not args.output_file:
            print("[Lỗi] Chế độ 'custom' yêu cầu truyền cả --input_file và --output_file!")
            return
        inp_path = Path(args.input_file)
        out_path = Path(args.output_file)
        audio_base = data_root

        runner.run_on_dataset(
            input_jsonl_path=inp_path,
            output_jsonl_path=out_path,
            audio_base_dir=audio_base,
            limit=args.limit,
            transcribe_history=True,
            overwrite=args.overwrite
        )


if __name__ == "__main__":
    main()
