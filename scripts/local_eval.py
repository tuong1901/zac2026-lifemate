"""
local_eval.py — Hệ thống chấm điểm nội bộ Zalo AI Challenge 2026 (LifeMate).
Cài đặt chính xác 100% theo công thức toán học và thể lệ kỹ thuật chính thức của BTC:

Score = {
  0,  if hash_sys != hash_gold or M == 0
  (2 * P * R) / (P + R),  otherwise
}

P = M / (M + U)
R = M / |G|
Final score = (100 / N) * sum(Score_i)

- Nếu |G| == 0 (Out-of-scope): Score = 1 nếu submitted calls rỗng ([]), ngược lại = 0.
- M: Số lượng tool call khớp (Tên tool trùng khớp, tham số non-fuzzy giống hệt, tham số fuzzy có token-F1 > 0.6).
- U: Số lượng call không khớp (bao gồm cả call bị từ chối do lỗi schema).
- hash_sys != hash_gold: Điểm bằng 0 ngay lập tức (Gatekeeper).
"""

import os
import sys
import json
import random
import unicodedata
import argparse
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# Cố định random seed theo quy định Reproducibility của BTC
SEED = 42
random.seed(SEED)
try:
    import numpy as np
    np.random.seed(SEED)
except ImportError:
    pass
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass


def compute_token_f1(s1, s2) -> float:
    """
    Tính Token-F1 sau khi lowercase và chuẩn hóa Unicode NFC
    """
    s1_norm = unicodedata.normalize("NFC", str(s1).lower()).strip()
    s2_norm = unicodedata.normalize("NFC", str(s2).lower()).strip()
    tokens1 = s1_norm.split()
    tokens2 = s2_norm.split()

    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    c1 = Counter(tokens1)
    c2 = Counter(tokens2)
    common = sum((c1 & c2).values())

    p = common / len(tokens1)
    r = common / len(tokens2)
    if p + r == 0:
        return 0.0
    return (2 * p * r) / (p + r)


def load_fuzzy_map(catalog_path: Path) -> dict:
    """
    Trích xuất danh sách các tham số fuzzy (x-match: fuzzy) từ catalog
    """
    fuzzy_map = {}
    if not catalog_path.exists():
        return fuzzy_map

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    for app in catalog.get("apps", []):
        for tool in app.get("tools", []):
            tname = tool["name"]
            props = tool.get("parameters", {}).get("properties", {})
            for pname, pinfo in props.items():
                if pinfo.get("x-match") == "fuzzy":
                    fuzzy_map.setdefault(tname, set()).add(pname)
    return fuzzy_map


def match_single_call(pred_call: dict, gold_call: dict, fuzzy_params: set) -> bool:
    """
    Kiểm tra 1 pred call có khớp với 1 gold call không:
    - Tên tool trùng khớp
    - Tham số non-fuzzy giống hệt
    - Tham số fuzzy có token-F1 > 0.6
    """
    if pred_call.get("name") != gold_call.get("name"):
        return False

    pred_args = pred_call.get("arguments") or {}
    gold_args = gold_call.get("arguments") or {}

    # Tất cả các key trong gold bắt buộc phải có trong pred và ngược lại
    all_keys = set(pred_args.keys()) | set(gold_args.keys())

    for k in all_keys:
        val_pred = pred_args.get(k)
        val_gold = gold_args.get(k)

        if k in fuzzy_params:
            # Tham số tự do: tính token-F1 > 0.6
            f1 = compute_token_f1(val_pred, val_gold)
            if f1 <= 0.6:
                return False
        else:
            # Tham số strict: phải giống hệt
            if val_pred != val_gold:
                return False

    return True


def evaluate_sample(pred_calls: list, gold_calls: list, hash_sys: str, hash_gold: str, fuzzy_map: dict) -> float:
    """
    Tính điểm cho một mẫu (Score_i) theo đúng công thức BTC
    """
    # 1. Trường hợp Out-of-scope (Gold list rỗng)
    if len(gold_calls) == 0:
        return 1.0 if len(pred_calls) == 0 else 0.0

    # 2. Điều kiện tiên quyết: State Hash Match
    if hash_sys != hash_gold:
        return 0.0

    # 3. Tính số lượng call trùng khớp M (Bipartite matching)
    matched_gold_indices = set()
    M = 0

    for p_call in pred_calls:
        tname = p_call.get("name", "")
        fuzzy_p = fuzzy_map.get(tname, set())

        for g_idx, g_call in enumerate(gold_calls):
            if g_idx not in matched_gold_indices:
                if match_single_call(p_call, g_call, fuzzy_p):
                    matched_gold_indices.add(g_idx)
                    M += 1
                    break

    if M == 0:
        return 0.0

    U = len(pred_calls) - M
    P = M / (M + U) if (M + U) > 0 else 0.0
    R = M / len(gold_calls) if len(gold_calls) > 0 else 0.0

    if P + R == 0:
        return 0.0

    return (2 * P * R) / (P + R)


def main():
    parser = argparse.ArgumentParser(description="ZAC 2026 LifeMate Official Scorer")
    parser.add_argument("--data_root", type=str, default="D:/Download/train+public_test/train+public_test", help="Thư mục data gốc")
    parser.add_argument("--limit", type=int, default=100, help="Số lượng mẫu test offline (mặc định: 100 mẫu)")
    args = parser.parse_args()

    root = Path(args.data_root)
    split_dir = root / "train"
    train_file = split_dir / "train.jsonl"
    catalog_file = split_dir / "train_tools_catalog.json"

    if not train_file.exists():
        print(f"Lỗi: Không tìm thấy file {train_file}")
        return

    sys.path.insert(0, str(split_dir))
    try:
        from zaic_tools import ToolEnv
    except ImportError as e:
        print(f"Lỗi import zaic_tools: {e}")
        return

    fuzzy_map = load_fuzzy_map(catalog_file)
    print(f"Đã nạp schema: có {len(fuzzy_map)} tools chứa tham số dạng fuzzy (x-match: fuzzy).")
    print(f"--- Bắt đầu đánh giá trên {args.limit} mẫu train theo chuẩn BTC ---")

    scores = []
    state_matches = 0
    empty_matches = 0
    total = 0

    with open(train_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, total=args.limit)):
            if idx >= args.limit:
                break
            item = json.loads(line)
            sample_id = item["id"]
            gold_calls = item.get("label", {}).get("tool_calls", [])

            # 1. Tính hash_gold chuẩn bằng cách chạy replay gold_calls trên 1 ToolEnv sạch
            gold_env = ToolEnv("train", sample_id, root=root)
            for c in gold_calls:
                gold_env.call(c["name"], c.get("arguments") or {})
            hash_gold = gold_env.get_state_hash()

            # 2. Chạy môi trường Agent dự đoán (System Env)
            sys_env = ToolEnv("train", sample_id, root=root)

            # --- MÔ PHỎNG DỰ ĐOÁN CỦA PIPELINE NHÓM ---
            # (Thay thế đoạn này bằng agent thực tế của nhóm: pred_calls, hash_sys = agent.run(...))
            # Ở bản baseline test này, ta gọi đúng gold_calls để xác minh độ chuẩn của bộ Scorer:
            for c in gold_calls:
                sys_env.call(c["name"], c.get("arguments") or {})

            pred_calls = sys_env.get_tool_calls()
            hash_sys = sys_env.get_state_hash()

            # 3. Tính điểm theo công thức chính thức
            sample_score = evaluate_sample(pred_calls, gold_calls, hash_sys, hash_gold, fuzzy_map)
            scores.append(sample_score)

            if hash_sys == hash_gold:
                state_matches += 1
            if len(gold_calls) == 0 and len(pred_calls) == 0:
                empty_matches += 1

            total += 1

    final_score = (100.0 / total) * sum(scores) if total > 0 else 0.0
    state_match_rate = (state_matches / total) * 100 if total > 0 else 0.0

    print("\n" + "=" * 60)
    print("       KẾT QUẢ ĐÁNH GIÁ CHUẨN THỂ LỆ ZALO AI CHALLENGE 2026")
    print("=" * 60)
    print(f"• Tổng số mẫu đánh giá (N):           {total}")
    print(f"• Tỷ lệ khớp State Hash (hash_sys == hash_gold): {state_match_rate:.2f}%")
    print(f"• ĐIỂM SỐ CHÍNH THỨC (FINAL SCORE / 100):  {final_score:.4f} / 100.0")
    print("=" * 60)
    print("Lưu ý: Mọi cải tiến của nhóm phải làm tăng chỉ số FINAL SCORE này trước khi submit!")

if __name__ == "__main__":
    main()
