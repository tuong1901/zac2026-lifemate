import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def calculate_tool_f1(pred_calls, true_calls):
    """
    Tính F1-score trên danh sách tool calls (so khớp tên tool và arguments)
    """
    if not pred_calls and not true_calls:
        return 1.0
    if not pred_calls or not true_calls:
        return 0.0

    def call_to_str(c):
        return f"{c.get('name')}::{json.dumps(c.get('arguments', {}), sort_keys=True)}"

    pred_set = set(call_to_str(c) for c in pred_calls)
    true_set = set(call_to_str(c) for c in true_calls)

    common = pred_set.intersection(true_set)
    precision = len(common) / len(pred_set) if pred_set else 0.0
    recall = len(common) / len(true_set) if true_set else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def main():
    parser = argparse.ArgumentParser(description="Chấm điểm offline cho ZAC 2026 LifeMate")
    parser.add_argument("--data_root", type=str, default="D:/Download/train+public_test/train+public_test", help="Thư mục chứa train và public_test")
    parser.add_argument("--limit", type=int, default=50, help="Số lượng mẫu test (để test nhanh)")
    args = parser.parse_args()

    root = Path(args.data_root)
    split_dir = root / "train"
    train_file = split_dir / "train.jsonl"

    if not train_file.exists():
        print(f"Lỗi: Không tìm thấy file {train_file}")
        return

    # Thêm zaic_tools của split train vào sys.path
    sys.path.insert(0, str(split_dir))
    try:
        from zaic_tools import ToolEnv
        from zaic_tools.utils import state_hash as calc_hash
    except ImportError as e:
        print(f"Lỗi import zaic_tools: {e}")
        return

    print(f"--- Bắt đầu đánh giá nội bộ trên {args.limit} mẫu train ---")
    f1_scores = []
    state_matches = 0
    total = 0

    with open(train_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, total=args.limit)):
            if idx >= args.limit:
                break
            item = json.loads(line)
            sample_id = item["id"]
            true_calls = item.get("label", {}).get("tool_calls", [])
            true_final_state = item.get("label", {}).get("final_state")

            # Tạo môi trường độc lập cho mẫu này
            env = ToolEnv("train", sample_id, root=root)

            # --- MÔ PHỎNG DỰ ĐOÁN (Baseline Mock) ---
            # Sau này thay phần này bằng agent thực tế của nhóm:
            # pred_calls, current_hash = agent.run(env, item, user_text)
            # Ở bản mock baseline này, ta gọi thử chính label để kiểm tra tính đúng đắn của scorer:
            for call in true_calls:
                env.call(call["name"], call.get("arguments", {}))
            
            pred_calls = env.get_tool_calls()
            pred_hash = env.get_state_hash()

            # 1. Đo F1 trên tool calls
            f1 = calculate_tool_f1(pred_calls, true_calls)
            f1_scores.append(f1)

            # 2. Đo State khớp với final_state
            if true_final_state is not None:
                current_states = env.get_state()
                # Kiểm tra tất cả các app trong true_final_state có khớp không
                match = True
                for app_name, expected_app_state in true_final_state.items():
                    if current_states.get(app_name) != expected_app_state:
                        match = False
                        break
                if match:
                    state_matches += 1
            else:
                # Nếu không có final_state thay đổi (NO_TOOL hoặc chỉ đọc)
                state_matches += 1

            total += 1

    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    state_acc = (state_matches / total) * 100 if total > 0 else 0.0

    print("\n================ KẾT QUẢ ĐÁNH GIÁ NỘI BỘ ================")
    print(f"Tổng số mẫu đánh giá: {total}")
    print(f"Điểm F1 Tool Calls trung bình: {avg_f1:.4f}")
    print(f"Độ chính xác Final State (State Accuracy): {state_acc:.2f}%")
    print("=========================================================")

if __name__ == "__main__":
    main()
