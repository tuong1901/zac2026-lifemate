"""
validate_submission.py — Tiện ích kiểm tra nhanh file submission.jsonl
Tự động nạp zaic_tools của BTC mà không cần cấu hình PYTHONPATH thủ công.
Cách dùng:
    python scripts/validate_submission.py submission.jsonl
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Validate submission file with zaic_tools")
    parser.add_argument("submission_file", type=str, default="submission.jsonl", nargs="?", help="Đường dẫn file submission.jsonl")
    parser.add_argument("--data_root", type=str, default="D:/Download/train+public_test/train+public_test", help="Thư mục data gốc")
    parser.add_argument("--split", type=str, default="public_test", help="Split cần validate (mặc định: public_test)")
    args = parser.parse_args()

    root = Path(args.data_root)
    split_dir = root / args.split
    sub_path = Path(args.submission_file)

    if not sub_path.exists():
        print(f"[LỖI] Không tìm thấy file: {sub_path}")
        return

    # Tự động nạp zaic_tools từ thư mục split
    sys.path.insert(0, str(split_dir))
    try:
        from zaic_tools.validate import schema_validate
        from zaic_tools.executor import Executor
        import json
    except ImportError as e:
        print(f"[LỖI] Không nạp được zaic_tools từ {split_dir}: {e}")
        return

    print(f"--- Đang kiểm tra tính hợp lệ của file: {sub_path.name} ---")
    
    # Nạp catalog của split
    catalog_path = split_dir / f"{args.split}_tools_catalog.json"
    env_dir = split_dir / f"{args.split}_environment"
    
    executor = Executor(str(env_dir), str(catalog_path) if catalog_path.exists() else None)
    
    lines_checked = 0
    errors = 0

    with open(sub_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            lines_checked += 1
            try:
                item = json.loads(line)
            except Exception as e:
                print(f"Dòng {line_no}: Lỗi JSON format: {e}")
                errors += 1
                continue

            sample_id = item.get("id")
            tool_calls = item.get("tool_calls", [])
            
            if not sample_id:
                print(f"Dòng {line_no}: Thiếu trường 'id'")
                errors += 1
                continue

            try:
                session = executor.open(sample_id)
                for c in tool_calls:
                    cname = c.get("name", "")
                    cargs = c.get("arguments", {})
                    v = session.validate(cname, cargs)
                    if not v.get("ok"):
                        print(f"Dòng {line_no} ({sample_id}): Lỗi schema tại tool '{cname}': {v.get('error')}")
                        errors += 1
            except Exception as e:
                print(f"Dòng {line_no} ({sample_id}): Lỗi kiểm tra: {e}")
                errors += 1

    print("\n" + "=" * 50)
    print(f"Kết quả: {lines_checked} lines checked, {errors} schema errors")
    if errors == 0:
        print("[THÀNH CÔNG] File submission hoàn toàn hợp lệ 100% để nộp lên Leaderboard!")
    else:
        print(f"[CẢNH BÁO] Phát hiện {errors} lỗi schema cần sửa trước khi nộp bài!")
    print("=" * 50)

if __name__ == "__main__":
    main()
