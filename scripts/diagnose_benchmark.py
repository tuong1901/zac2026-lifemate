"""
diagnose_benchmark.py — Phân tích chi tiết các mẫu sai trong 20 mẫu benchmark.
"""
import os
import sys
import json
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Đảm bảo in tiếng Việt mượt mà
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.rules.catalog_filter import CatalogFilter
from src.rules.schema_guardrail import SafeSchemaGuardrail
from src.agent.rag.exemplar_retriever import BM25ExemplarRetriever
from src.agent.rag.history_resolver import HistoryResolver
from src.agent.react_loop import ReActAgent
from src.agent.llm_client import get_llm_client
from scripts.local_eval import evaluate_sample, load_fuzzy_map

def main():
    data_root = Path("D:/Download/train+public_test/train+public_test")
    split_dir = data_root / "train"
    catalog_file = split_dir / "train_tools_catalog.json"
    cache_path = PROJECT_ROOT / "data" / "bm25_train_index.pkl"
    eval_path = PROJECT_ROOT / "data" / "val_benchmark_200.jsonl"
    base_url = "https://aadf-34-16-190-83.ngrok-free.app/v1"

    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv

    fuzzy_map = load_fuzzy_map(catalog_file)
    llm_client = get_llm_client(client_type="openai", base_url=base_url, model="Qwen/Qwen2.5-3B-Instruct")
    catalog_filter = CatalogFilter(str(catalog_file))
    retriever = BM25ExemplarRetriever()
    retriever.load_cache(str(cache_path))
    history_resolver = HistoryResolver()
    guardrail = SafeSchemaGuardrail()

    agent = ReActAgent(
        llm_client=llm_client,
        catalog_filter=catalog_filter,
        exemplar_retriever=retriever,
        history_resolver=history_resolver,
        guardrail=guardrail,
        max_steps=5
    )

    limit = 10
    print(f"=== CHẠY DIAGNOSTIC TRÊN {limit} MẪU ===")

    with open(eval_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            item = json.loads(line)
            sample_id = item["id"]
            user_text = item.get("audio_text", "")
            gold_calls = item.get("label", {}).get("tool_calls", [])

            gold_env = ToolEnv("train", sample_id, root=data_root)
            for c in gold_calls:
                gold_env.call(c["name"], c.get("arguments") or {})
            hash_gold = gold_env.get_state_hash()

            sys_env = ToolEnv("train", sample_id, root=data_root)
            print(f"\n[{idx+1}/{limit}] Sample ID: {sample_id}")
            print(f"  Utterance: \"{user_text}\"")
            pred_calls, hash_sys = agent.run(sys_env, item, user_text)

            score = evaluate_sample(pred_calls, gold_calls, hash_sys, hash_gold, fuzzy_map)
            hash_match = (hash_sys == hash_gold)

            print(f"  --> Score: {score:.4f} | Hash match: {hash_match}")
            print(f"      Gold calls ({len(gold_calls)}): {json.dumps(gold_calls, ensure_ascii=False)}")
            print(f"      Pred calls ({len(pred_calls)}): {json.dumps(pred_calls, ensure_ascii=False)}")
            if not hash_match:
                print(f"      [MISMATCH] Hash Gold: {hash_gold} vs Hash Sys: {hash_sys}")

if __name__ == "__main__":
    main()
