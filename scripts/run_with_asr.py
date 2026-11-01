"""
run_with_asr.py — Đánh giá và so sánh trực tiếp điểm số giữa:
1. Benchmark với DB chuẩn của BTC (Gold audio_text trong val_benchmark_200.jsonl)
2. Benchmark với DB thực tế đã tối ưu ASR của Khoa (MultiASR_Results: PhoWhisper-medium / Whisper-turbo / PhoWhisper-small)
3. Xuất file submission.jsonl trên 634 mẫu public_test (nếu bật --make_submission)
"""

import os
import sys
import json
import copy
import argparse
from pathlib import Path
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.rules.catalog_filter import CatalogFilter
from src.rules.schema_guardrail import SafeSchemaGuardrail
from src.agent.rag.exemplar_retriever import BM25ExemplarRetriever
from src.agent.rag.history_resolver import HistoryResolver
from src.agent.react_loop import ReActAgent
from src.agent.llm_client import get_llm_client
from local_eval import load_fuzzy_map, evaluate_sample


def load_asr_map(asr_jsonl_path: Path) -> dict:
    asr_map = {}
    if not asr_jsonl_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file ASR: {asr_jsonl_path}")
    with open(asr_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            asr_map[rec["id"]] = rec
    return asr_map


def inject_asr_into_item(item: dict, asr_rec: dict) -> tuple[dict, str]:
    """
    Thay thế câu khẩu lệnh chính và các lượt user trong history bằng kết quả ASR của Khoa.
    """
    new_item = copy.deepcopy(item)
    user_text = asr_rec.get("transcript", item.get("audio_text", ""))
    new_item["audio_text"] = user_text

    hist_Map = {}
    for h_tr in asr_rec.get("history_user_transcripts", []):
        if "history_index" in h_tr and "transcript" in h_tr:
            hist_Map[h_tr["history_index"]] = h_tr["transcript"]

    if "history" in new_item and isinstance(new_item["history"], list):
        for idx, turn in enumerate(new_item["history"]):
            if idx in hist_Map:
                turn["audio_text"] = hist_Map[idx]

    return new_item, user_text


def eval_dataset(agent, eval_items, root: Path, fuzzy_map: dict, asr_map: dict = None, desc: str = "Eval"):
    from zaic_tools import ToolEnv

    scores = []
    state_matches = 0
    schema_penalties = 0
    total = len(eval_items)

    for idx, item in enumerate(tqdm(eval_items, desc=desc), 1):
        sample_id = item["id"]
        gold_calls = item.get("label", {}).get("tool_calls", [])

        if asr_map and sample_id in asr_map:
            run_item, user_text = inject_asr_into_item(item, asr_map[sample_id])
        else:
            run_item = item
            user_text = item.get("audio_text", "")

        gold_env = ToolEnv("train", sample_id, root=root)
        for c in gold_calls:
            gold_env.call(c["name"], c.get("arguments") or {})
        hash_gold = gold_env.get_state_hash()

        sys_env = ToolEnv("train", sample_id, root=root)
        pred_calls, hash_sys = agent.run(sys_env, run_item, user_text)

        has_rejected = any("error" in c.get("result", {}) for c in sys_env.log)
        if has_rejected:
            schema_penalties += 1

        sc = evaluate_sample(pred_calls, gold_calls, hash_sys, hash_gold, fuzzy_map)
        scores.append(sc)
        matched = (hash_sys == hash_gold)
        if matched:
            state_matches += 1

        if total <= 20:
            status_icon = "✅" if matched and sc == 1.0 else ("⚠️" if matched else "❌")
            print(f"\n   {status_icon} [{idx:02d}/{total}] {sample_id} | Điểm mẫu: {sc:.2f} (Hash match={matched})", flush=True)
            if sc < 1.0:
                print(f"      - PRED: {json.dumps(pred_calls, ensure_ascii=False)}", flush=True)
                print(f"      - GOLD: {json.dumps(gold_calls, ensure_ascii=False)}", flush=True)
            else:
                print(f"      - CALLS: {[c['name'] for c in pred_calls]}", flush=True)

        if idx % 25 == 0 or idx == total:
            cur_score = (100.0 / idx) * sum(scores)
            cur_state = (state_matches / idx) * 100.0
            print(f"\n   -> [{desc}] Tiến độ {idx}/{total} | State Hash: {cur_state:.2f}% ({state_matches}/{idx}) | Phạt U: {schema_penalties} | Điểm: {cur_score:.4f}/100", flush=True)

    final_score = (100.0 / total) * sum(scores) if total > 0 else 0.0
    state_rate = (state_matches / total) * 100.0 if total > 0 else 0.0
    res = {
        "total": total,
        "final_score": final_score,
        "state_match_rate": state_rate,
        "state_matches": state_matches,
        "schema_penalties": schema_penalties,
    }
    print(f"\n✅ HOÀN TẤT [{desc}]: Khớp State Hash = {state_rate:.2f}% ({state_matches}/{total}) | Phạt U = {schema_penalties} | ĐIỂM CHUNG CUỘC = {final_score:.4f} / 100\n", flush=True)
    return res


def main():
    parser = argparse.ArgumentParser(description="Chạy so sánh Benchmark BTC vs DB ASR của Khoa")
    parser.add_argument("--data_root", type=str, default="D:/TOOL_TH/ZAC_2026/train+public_test/train+public_test")
    parser.add_argument("--asr_dir", type=str, default="D:/TOOL_TH/ZAC_2026/MultiASR_Results")
    parser.add_argument("--eval_file", type=str, default="data/val_benchmark_200.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--mode", choices=["hf", "server"], default="server")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--asr_model", choices=["phowhisper_medium", "whisper_turbo", "phowhisper_small", "all"], default="phowhisper_medium")
    args = parser.parse_args()

    root = Path(args.data_root)
    split_dir = root / "train"
    catalog_file = split_dir / "train_tools_catalog.json"
    cache_path = PROJECT_ROOT / "data" / "bm25_train_index.pkl"

    sys.path.insert(0, str(split_dir))
    import zaic_tools  # noqa

    eval_path = PROJECT_ROOT / args.eval_file if not Path(args.eval_file).is_absolute() else Path(args.eval_file)
    eval_items = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= args.limit:
                break
            if line.strip():
                eval_items.append(json.loads(line))

    fuzzy_map = load_fuzzy_map(catalog_file)

    if args.mode == "hf":
        llm_client = get_llm_client(client_type="hf", model_name_or_path=args.model)
    else:
        llm_client = get_llm_client(client_type="openai", base_url=args.base_url, model=args.model)

    catalog_filter = CatalogFilter(str(catalog_file))
    retriever = BM25ExemplarRetriever()
    if cache_path.exists():
        retriever.load_cache(str(cache_path))
    else:
        retriever.build_from_train_file(str(split_dir / "train.jsonl"))
        retriever.save_cache(str(cache_path))

    agent = ReActAgent(
        llm_client=llm_client,
        catalog_filter=catalog_filter,
        exemplar_retriever=retriever,
        history_resolver=HistoryResolver(),
        guardrail=SafeSchemaGuardrail(),
        max_steps=7
    )

    print("\n" + "=" * 72)
    print(f" BẮT ĐẦU CHẠY ĐỐI CHIẾU TRÊN {len(eval_items)} MẪU BENCHMARK")
    print("=" * 72)

    # 1. Chạy trên DB chuẩn của BTC (Gold Text)
    res_gold = eval_dataset(agent, eval_items, root, fuzzy_map, asr_map=None, desc="1. DB Chuẩn BTC (Gold Text)")

    # 2. Chạy trên DB ASR của Khoa
    asr_dir = Path(args.asr_dir)
    models_to_run = ["phowhisper_medium", "whisper_turbo", "phowhisper_small"] if args.asr_model == "all" else [args.asr_model]
    asr_results = {}
    for m_name in models_to_run:
        asr_file = asr_dir / f"train_{m_name}_for_agent.jsonl"
        asr_map = load_asr_map(asr_file)
        asr_results[m_name] = eval_dataset(agent, eval_items, root, fuzzy_map, asr_map=asr_map, desc=f"2. DB Khoa ({m_name})")

    print("\n" + "=" * 72)
    print(" BẢNG TỔNG HỢP ĐIỂM SỐ (TRÊN THANG 100.0 CỦA BTC)")
    print("=" * 72)
    print(f"{'Bộ dữ liệu (Dataset)':<35} | {'Khớp State Hash':<16} | {'Phạt U':<8} | {'ĐIỂM SỐ (/100)'}")
    print("-" * 72)
    print(f"{'1. DB Ban Tổ Chức (Gold Text)':<35} | {res_gold['state_match_rate']:>6.2f}% ({res_gold['state_matches']}/{res_gold['total']}) | {res_gold['schema_penalties']:>6} | {res_gold['final_score']:>8.4f} / 100")
    for m_name, r in asr_results.items():
        label = f"2. DB Khoa ({m_name})"
        print(f"{label:<35} | {r['state_match_rate']:>6.2f}% ({r['state_matches']}/{r['total']}) | {r['schema_penalties']:>6} | {r['final_score']:>8.4f} / 100")
    print("=" * 72)


if __name__ == "__main__":
    main()

