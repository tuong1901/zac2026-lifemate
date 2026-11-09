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

# Đảm bảo in tiếng Việt mượt mà trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rules.catalog_filter import CatalogFilter
from src.rules.schema_guardrail import SafeSchemaGuardrail
from src.agent.rag.exemplar_retriever import BM25ExemplarRetriever
from src.agent.rag.history_resolver import HistoryResolver
from src.agent.react_loop import ReActAgent
from src.agent.llm_client import get_llm_client


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


def load_fuzzy_map(catalog_path: Optional[Any] = None) -> dict:
    """
    Trích xuất danh sách các tham số fuzzy (x-match: fuzzy) từ catalog
    """
    fuzzy_map = {}
    if not catalog_path:
        for cand in [
            Path("data/train/train_tools_catalog.json"),
            Path("data/public_test/public_test_tools_catalog.json"),
            Path("/kaggle/input/datasets/n107hoangtuong/zalo-competion/train+public_test/train/train_tools_catalog.json"),
        ]:
            if cand.exists():
                catalog_path = cand
                break
    if not catalog_path or not Path(catalog_path).exists():
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

    args_pred = pred_call.get("arguments") or {}
    args_gold = gold_call.get("arguments") or {}

    keys_gold = set(args_gold.keys())
    keys_pred = set(args_pred.keys())

    # Tất cả các keys trong gold phải có mặt trong pred
    if not keys_gold.issubset(keys_pred):
        return False

    for k in keys_gold:
        val_pred = args_pred.get(k)
        val_gold = args_gold.get(k)

        if k in fuzzy_params:
            if compute_token_f1(val_pred, val_gold) <= 0.6:
                return False
        else:
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

    # 2. Điều kiện tiên quyết: State Hash Match (Cổng sinh tử)
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
    parser.add_argument("--eval_file", type=str, default="data/val_benchmark_200.jsonl", help="File dữ liệu đánh giá")
    parser.add_argument("--limit", type=int, default=20, help="Số lượng mẫu test offline (mặc định: 20 mẫu)")
    parser.add_argument("--mode", choices=["hf", "server"], default="hf", help="Chế độ LLM: 'hf' hoặc 'server'")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Tên model")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="Base URL nếu dùng server vLLM/Ollama")
    parser.add_argument("--checkpoint_file", type=str, default=None,
                        help="JSONL lưu kết quả từng mẫu để khôi phục khi endpoint ngắt")
    parser.add_argument("--resume", action="store_true",
                        help="Bỏ qua các sample đã có trong checkpoint")
    args = parser.parse_args()

    root = Path(args.data_root)
    split_dir = root / "train"
    catalog_file = split_dir / "train_tools_catalog.json"

    eval_path = Path(args.eval_file)
    if not eval_path.is_absolute():
        eval_path = PROJECT_ROOT / eval_path
    if not eval_path.exists():
        eval_path = split_dir / "train.jsonl"

    sys.path.insert(0, str(split_dir))
    try:
        from zaic_tools import ToolEnv
    except ImportError as e:
        print(f"Lỗi import zaic_tools: {e}")
        return

    fuzzy_map = load_fuzzy_map(catalog_file)
    print("=" * 65)
    print("       HỆ THỐNG ĐÁNH GIÁ CHUẨN THỂ LỆ ZALO AI CHALLENGE 2026")
    print("=" * 65)
    print(f"• Dataset đánh giá: {eval_path.name}")
    print(f"• Số lượng mẫu:     {args.limit}")
    print(f"• LLM Mode:         {args.mode} ({args.model})")
    print(f"• Schema:           {len(fuzzy_map)} tools chứa tham số fuzzy (x-match: fuzzy)")

    # Khởi tạo Pipeline của nhóm
    print("\n[+] Đang khởi tạo Agent Pipeline...")
    if args.mode == "hf":
        llm_client = get_llm_client(client_type="hf", model_name_or_path=args.model)
    else:
        llm_client = get_llm_client(client_type="openai", base_url=args.base_url, model=args.model)

    catalog_filter = CatalogFilter(str(catalog_file))
    retriever = BM25ExemplarRetriever()
    with open(eval_path, "r", encoding="utf-8") as f:
        excluded_ids = {json.loads(line)["id"] for line in f if line.strip()}
    retriever.build_from_train_file(str(split_dir / "train.jsonl"), exclude_ids=excluded_ids)
    if any(rec["id"] in excluded_ids for rec in retriever.corpus):
        raise RuntimeError("Validation ID vẫn nằm trong chỉ mục BM25")
    print(f"[Clean holdout] Loại {len(excluded_ids)} validation ID khỏi BM25; còn {len(retriever.corpus)} ví dụ train.")

    history_resolver = HistoryResolver()
    guardrail = SafeSchemaGuardrail()

    agent = ReActAgent(
        llm_client=llm_client,
        catalog_filter=catalog_filter,
        exemplar_retriever=retriever,
        history_resolver=history_resolver,
        guardrail=guardrail,
        max_steps=7
    )

    print(f"\n--- Bắt đầu đánh giá trên {args.limit} mẫu theo chuẩn BTC ---")

    checkpoint_path = Path(args.checkpoint_file) if args.checkpoint_file else None
    completed = {}
    if checkpoint_path:
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as checkpoint_in:
                for checkpoint_line in checkpoint_in:
                    if checkpoint_line.strip():
                        record = json.loads(checkpoint_line)
                        completed[int(record["index"])] = record
            print(f"[Resume] Đã khôi phục {len(completed)} sample từ {checkpoint_path}")
        else:
            checkpoint_path.write_text("", encoding="utf-8")

    checkpoint_out = (
        open(checkpoint_path, "a", encoding="utf-8", buffering=1)
        if checkpoint_path else None
    )
    stopped_for_endpoint = False
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(tqdm(f, total=args.limit)):
                if idx >= args.limit:
                    break
                if idx in completed:
                    continue
                item = json.loads(line)
                sample_id = item["id"]
                user_text = item.get("audio_text", "")
                gold_calls = item.get("label", {}).get("tool_calls", [])

                # 1. Tính hash_gold chuẩn bằng cách chạy replay gold_calls trên 1 ToolEnv sạch
                gold_env = ToolEnv("train", sample_id, root=root)
                for c in gold_calls:
                    gold_env.call(c["name"], c.get("arguments") or {})
                hash_gold = gold_env.get_state_hash()

                # 2. Chạy Agent thật của nhóm trên môi trường ToolEnv sạch
                sys_env = ToolEnv("train", sample_id, root=root)
                pred_calls, hash_sys = agent.run(sys_env, item, user_text)
                if getattr(llm_client, "last_error_kind", None) == "endpoint_unavailable":
                    stopped_for_endpoint = True
                    print(
                        f"\n[Checkpoint] Endpoint ngắt tại mẫu {idx + 1} "
                        f"({sample_id}); mẫu này chưa được ghi."
                    )
                    break

                # 3. Tính điểm theo công thức chính thức
                has_rejected = any("error" in c.get("result", {}) for c in sys_env.log)
                sample_score = evaluate_sample(pred_calls, gold_calls, hash_sys, hash_gold, fuzzy_map)
                record = {
                    "index": idx,
                    "sample_id": sample_id,
                    "score": sample_score,
                    "state_match": hash_sys == hash_gold,
                    "schema_penalty": has_rejected,
                    "pred_calls": pred_calls,
                    "gold_calls": gold_calls,
                    "hash_sys": hash_sys,
                    "hash_gold": hash_gold,
                }
                completed[idx] = record
                if checkpoint_out:
                    checkpoint_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if checkpoint_out:
            checkpoint_out.close()

    records = [completed[index] for index in sorted(completed) if index < args.limit]
    scores = [record["score"] for record in records]
    total = len(records)
    state_matches = sum(bool(record["state_match"]) for record in records)
    schema_penalties = sum(bool(record["schema_penalty"]) for record in records)

    final_score = (100.0 / total) * sum(scores) if total > 0 else 0.0
    state_match_rate = (state_matches / total) * 100 if total > 0 else 0.0

    print("\n" + "=" * 65)
    print("                 KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC")
    print("=" * 65)
    print(f"• Tổng số mẫu đánh giá (N):           {total}")
    print(f"• Tỷ lệ khớp State Hash (Cổng sinh tử): {state_match_rate:.2f}% ({state_matches}/{total})")
    schema_rate = (schema_penalties / total * 100) if total else 0.0
    print(f"• Số mẫu bị phạt U (Lỗi schema):      {schema_penalties}/{total} ({schema_rate:.1f}%)")
    print(f"• ĐIỂM SỐ CHÍNH THỨC (FINAL SCORE):   {final_score:.4f} / 100.0")
    if stopped_for_endpoint:
        print("• Trạng thái:                         DỪNG SỚM do endpoint offline")
    if checkpoint_path:
        print(f"• Checkpoint:                         {checkpoint_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
