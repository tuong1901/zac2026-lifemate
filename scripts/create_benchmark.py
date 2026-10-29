"""
create_benchmark.py — Trích xuất 200 mẫu cố định từ tập train làm bài thi thử nội bộ.
Đảm bảo phân bố chuẩn:
- ~78% mẫu EXECUTE_TOOLS
- ~14% mẫu ASK_MISSING_INFO
- ~8% mẫu NO_TOOL ([])
"""

import os
import sys
import json
import random
from pathlib import Path

# Đảm bảo UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SEED = 42
random.seed(SEED)

def main():
    root = Path("D:/Download/train+public_test/train+public_test")
    train_file = root / "train" / "train.jsonl"
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "val_benchmark_200.jsonl"

    if not train_file.exists():
        print(f"Lỗi: Không tìm thấy file {train_file}")
        return

    execute_samples = []
    ask_missing_samples = []
    no_tool_samples = []

    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            calls = item.get("label", {}).get("tool_calls", [])
            if len(calls) == 0:
                no_tool_samples.append(item)
            elif any(c["name"] == "sys.ask_missing_info" for c in calls):
                ask_missing_samples.append(item)
            else:
                execute_samples.append(item)

    print(f"Tổng hợp tập train: {len(execute_samples)} Execute, {len(ask_missing_samples)} AskMissing, {len(no_tool_samples)} NoTool")

    # Lấy phân bố chuẩn: 156 Execute + 28 AskMissing + 16 NoTool = đúng 200 mẫu
    selected = (
        random.sample(execute_samples, min(156, len(execute_samples))) +
        random.sample(ask_missing_samples, min(28, len(ask_missing_samples))) +
        random.sample(no_tool_samples, min(16, len(no_tool_samples)))
    )

    random.shuffle(selected)

    with open(out_file, "w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[THÀNH CÔNG] Đã tạo tập thi thử cố định gồm {len(selected)} mẫu tại: {out_file}")

if __name__ == "__main__":
    main()
