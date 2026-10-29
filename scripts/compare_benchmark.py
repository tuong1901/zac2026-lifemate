"""
compare_benchmark.py — Bộ so sánh và kiểm định tự động điểm số của nhóm.
Kiểm tra xem commit mới làm TĂNG hay GIẢM điểm so với kỷ lục cũ.
Cách dùng:
    python scripts/compare_benchmark.py --author "Khoa" --note "Tích hợp Parakeet CTC"
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Đảm bảo UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_FILE = ROOT_DIR / "data" / "val_benchmark_200.jsonl"
HISTORY_FILE = ROOT_DIR / "benchmark_history.json"

# Import hàm đánh giá từ local_eval
sys.path.insert(0, str(ROOT_DIR / "scripts"))
from local_eval import load_fuzzy_map, evaluate_sample

def get_git_info():
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT_DIR).decode().strip()
        return branch, commit
    except Exception:
        return "main", "unknown"

def load_history():
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "best_score": 5.9900,
        "state_match_rate": 8.0,
        "author": "Lục Tiến Đạt (Baseline)",
        "timestamp": "2026-10-29 12:56:00",
        "branch": "main",
        "commit": "root",
        "note": "Initial dummy submission (Out-of-scope baseline)"
    }

def save_history(history_data):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)

def run_evaluation(data_root: Path):
    split_dir = data_root / "train"
    catalog_file = split_dir / "train_tools_catalog.json"
    
    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv

    fuzzy_map = load_fuzzy_map(catalog_file)
    
    scores = []
    state_matches = 0
    total = 0

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            sample_id = item["id"]
            gold_calls = item.get("label", {}).get("tool_calls", [])

            # 1. Gold environment replay
            gold_env = ToolEnv("train", sample_id, root=data_root)
            for c in gold_calls:
                gold_env.call(c["name"], c.get("arguments") or {})
            hash_gold = gold_env.get_state_hash()

            # 2. Agent system environment replay
            sys_env = ToolEnv("train", sample_id, root=data_root)

            # ==============================================================
            # ĐIỂM NỐI MÔ HÌNH:
            # Tại đây hệ thống sẽ gọi pipeline dự đoán của nhóm.
            # Nếu chưa có ASR/LLM thật, mặc định chạy theo logic hiện tại:
            # ==============================================================
            # Để kiểm thử so sánh, nếu có module agent thật sẽ gọi:
            # pred_calls, hash_sys = agent.run(sys_env, item, text)
            # Hiện tại mô phỏng dự đoán hiện có trong repo:
            pred_calls = sys_env.get_tool_calls()
            hash_sys = sys_env.get_state_hash()

            sample_score = evaluate_sample(pred_calls, gold_calls, hash_sys, hash_gold, fuzzy_map)
            scores.append(sample_score)

            if hash_sys == hash_gold:
                state_matches += 1
            total += 1

    final_score = (100.0 / total) * sum(scores) if total > 0 else 0.0
    state_rate = (state_matches / total) * 100 if total > 0 else 0.0
    return final_score, state_rate, total

def main():
    parser = argparse.ArgumentParser(description="So sánh điểm số trước và sau khi commit")
    parser.add_argument("--author", type=str, default="Thành viên nhóm", help="Tên thành viên commit")
    parser.add_argument("--note", type=str, default="Cải tiến mô hình", help="Mô tả ngắn cải tiến")
    parser.add_argument("--data_root", type=str, default="D:/Download/train+public_test/train+public_test", help="Thư mục data gốc")
    args = parser.parse_args()

    if not BENCHMARK_FILE.exists():
        print(f"[LỖI] Chưa có file benchmark. Chạy scripts/create_benchmark.py trước!")
        return

    print("=" * 65)
    print("      ZAC 2026 — BỘ ĐO LƯỜNG & SO SÁNH ĐIỂM SỐ NỘI BỘ")
    print("=" * 65)
    print(f"Người thực hiện: {args.author} | Ghi chú: {args.note}")
    print(f"Đang chạy kiểm thử trên 200 mẫu cố định ({BENCHMARK_FILE.name})...\n")

    current_history = load_history()
    old_score = current_history["best_score"]
    old_state_rate = current_history["state_match_rate"]

    new_score, new_state_rate, total_samples = run_evaluation(Path(args.data_root))

    diff_score = new_score - old_score
    diff_state = new_state_rate - old_state_rate
    branch, commit = get_git_info()

    print("\n" + "=" * 65)
    print("                    BẢNG ĐỐI CHIẾU KẾT QUẢ")
    print("=" * 65)
    print(f"Chỉ số đánh giá          | Kỷ lục cũ      | Phiên bản mới  | Chênh lệch")
    print("-" * 65)
    print(f"Tỷ lệ khớp State Hash    | {old_state_rate:6.2f}%       | {new_state_rate:6.2f}%       | {diff_state:+6.2f}%")
    print(f"ĐIỂM SỐ CHÍNH THỨC (F1)  | {old_score:6.4f}       | {new_score:6.4f}       | {diff_score:+6.4f}")
    print("=" * 65)

    if new_score > old_score:
        print("\n [CHẤP THUẬN - CODE CẢI TIẾN THÀNH CÔNG!]")
        print(f"Điểm số tăng vọt +{diff_score:.4f} điểm! Đạt điều kiện MERGE và SUBMIT lên web!")
        
        # Cập nhật kỷ lục mới vào history
        new_record = {
            "best_score": round(new_score, 4),
            "state_match_rate": round(new_state_rate, 2),
            "author": args.author,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "branch": branch,
            "commit": commit,
            "note": args.note
        }
        save_history(new_record)
        print(f"Đã cập nhật kỷ lục mới vào {HISTORY_FILE.name}")
        sys.exit(0)
    elif new_score == old_score:
        print("\n [CẢNH BÁO - ĐIỂM SỐ KHÔNG ĐỔI]")
        print(f"Phiên bản này bằng điểm kỷ lục cũ ({new_score:.4f}). Cần tối ưu thêm trước khi submit!")
        sys.exit(0)
    else:
        print("\n❌ [TỪ CHỐI MERGE - CODE BỊ LÙI BƯỚC (REGRESSION)!]")
        print(f"Điểm số bị tụt {diff_score:.4f} điểm so với kỷ lục của {current_history['author']}.")
        print("Hãy kiểm tra lại prompt hoặc logic gọi tool để không làm mất điểm các câu cũ!")
        sys.exit(1)

if __name__ == "__main__":
    main()
