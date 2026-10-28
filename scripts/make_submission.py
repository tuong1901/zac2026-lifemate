import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Tạo file nộp bài submission.jsonl cho ZAC 2026 LifeMate")
    parser.add_argument("--data_root", type=str, default="D:/Download/train+public_test/train+public_test", help="Thư mục chứa train và public_test")
    parser.add_argument("--output", type=str, default="submission.jsonl", help="Tên file kết quả đầu ra")
    args = parser.parse_args()

    root = Path(args.data_root)
    split = "public_test"
    split_dir = root / split
    test_file = split_dir / f"{split}.jsonl"

    if not test_file.exists():
        print(f"Lỗi: Không tìm thấy file {test_file}")
        return

    # Thêm zaic_tools của split public_test vào sys.path
    sys.path.insert(0, str(split_dir))
    try:
        from zaic_tools import ToolEnv
    except ImportError as e:
        print(f"Lỗi import zaic_tools: {e}")
        return

    print(f"--- Bắt đầu chạy suy luận trên {split} và xuất {args.output} ---")
    count = 0

    with open(test_file, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in tqdm(fin):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sample_id = item["id"]

            # Mỗi mẫu nạp một ToolEnv độc lập
            env = ToolEnv(split, sample_id, root=root)

            # --- TẠI ĐÂY: GỌI PIPELINE CỦA NHÓM ---
            # 1. user_text = transcriber.transcribe(root / split / item["audio"])
            # 2. agent.run(env, item, user_text)

            # Lấy chuỗi call đã gọi và hash của state cuối cùng
            tool_calls = env.get_tool_calls()
            state_hash = env.get_state_hash()

            record = {
                "id": sample_id,
                "tool_calls": tool_calls,
                "state_hash": state_hash
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"\n[Thành công] Đã xuất {count} dòng kết quả vào '{args.output}'!")
    print("Để kiểm tra tính hợp lệ của file nộp bài theo chuẩn BTC, hãy chạy:")
    print(f"python -m zaic_tools --split public_test --root \"{root}\" --validate \"{args.output}\"")

if __name__ == "__main__":
    main()
