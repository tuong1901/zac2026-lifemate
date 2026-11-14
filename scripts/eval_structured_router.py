"""Tune a workflow-chunk voting router on dev, then score clean Gold holdout."""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.rag.exemplar_retriever import BM25ExemplarRetriever
from src.agent.rag.structured_chunks import build_structured_chunks, domain_sequence, load_jsonl


def signature(calls: Iterable[Dict[str, Any]]) -> Tuple[str, ...]:
    return tuple(call.get("name", "") for call in calls)


def chunk_text(chunk: Dict[str, Any], mode: str) -> str:
    current = chunk["text"]
    history = chunk.get("history_signals", {})
    if mode == "current":
        return current
    if mode == "tools":
        return " ".join([current, *history.get("prior_tools", [])])
    return " ".join([
        current,
        current,
        *history.get("user_utterances", []),
        *history.get("prior_tools", []),
    ])


def make_retriever(chunks: List[Dict[str, Any]], mode: str) -> BM25ExemplarRetriever:
    records = []
    for chunk in chunks:
        sequence = tuple(chunk["tool_sequence"])
        records.append({
            "id": chunk["sample_id"],
            "audio_text": chunk_text(chunk, mode),
            "installed_apps": chunk["installed_apps"],
            "called_apps": set(domain_sequence(sequence)),
            "tool_calls": [{"name": name, "arguments": {}} for name in sequence],
        })
    retriever = BM25ExemplarRetriever()
    retriever.index_records(records)
    return retriever


def route(
    retriever: BM25ExemplarRetriever,
    query: str,
    installed_apps: Dict[str, Any],
    top_k: int,
    beta: float,
) -> Tuple[str, ...]:
    candidates = retriever.retrieve(query, installed_apps, top_k=top_k)
    if not candidates:
        return ()
    grouped: Dict[Tuple[str, ...], List[float]] = defaultdict(list)
    for candidate in candidates:
        grouped[signature(candidate["tool_calls"])].append(candidate["bm25_score"])
    scores = {
        workflow: values[0] + beta * sum(values[1:])
        for workflow, values in grouped.items()
    }
    return max(scores, key=scores.get)


def evaluate(
    index_chunks: List[Dict[str, Any]],
    eval_chunks: List[Dict[str, Any]],
    mode: str,
    top_k: int,
    beta: float,
) -> Dict[str, float]:
    retriever = make_retriever(index_chunks, mode)
    exact = 0
    domain_exact = 0
    for chunk in eval_chunks:
        predicted = route(
            retriever,
            chunk_text(chunk, mode),
            {app: {} for app in chunk["installed_apps"]},
            top_k,
            beta,
        )
        gold = tuple(chunk["tool_sequence"])
        exact += predicted == gold
        domain_exact += set(domain_sequence(predicted)) == set(chunk["domains"])
    total = len(eval_chunks)
    return {
        "total": total,
        "workflow_accuracy": exact / total if total else 0.0,
        "domain_accuracy": domain_exact / total if total else 0.0,
    }


def stable_bucket(sample_id: str) -> int:
    return int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16) % 7


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=Path, default=Path("D:/Download/train+public_test/train+public_test/train/train.jsonl"))
    parser.add_argument("--validation_file", type=Path, default=PROJECT_ROOT / "data" / "val_benchmark_200.jsonl")
    args = parser.parse_args()

    train = load_jsonl(args.train_file)
    validation = load_jsonl(args.validation_file)
    validation_ids = {item["id"] for item in validation}
    train_chunks = build_structured_chunks(train, exclude_ids=validation_ids)["intent"]
    validation_chunks = build_structured_chunks(validation)["intent"]

    dev_chunks = [chunk for chunk in train_chunks if stable_bucket(chunk["sample_id"]) == 0]
    build_chunks = [chunk for chunk in train_chunks if stable_bucket(chunk["sample_id"]) != 0]

    configs = []
    for mode in ("current", "tools", "all_history"):
        for top_k in (1, 3, 5, 8, 12):
            for beta in ((0.0,) if top_k == 1 else (0.15, 0.3, 0.5, 0.75, 1.0)):
                metrics = evaluate(build_chunks, dev_chunks, mode, top_k, beta)
                configs.append((metrics["workflow_accuracy"], metrics["domain_accuracy"], mode, top_k, beta))
    configs.sort(reverse=True)
    best = configs[0]
    _, _, mode, top_k, beta = best
    holdout = evaluate(train_chunks, validation_chunks, mode, top_k, beta)
    baseline = evaluate(train_chunks, validation_chunks, "current", 1, 0.0)

    print(json.dumps({
        "build_size": len(build_chunks),
        "dev_size": len(dev_chunks),
        "best_dev": {
            "workflow_accuracy": best[0],
            "domain_accuracy": best[1],
            "mode": mode,
            "top_k": top_k,
            "beta": beta,
        },
        "holdout_baseline": baseline,
        "holdout_structured_router": holdout,
        "top_dev_configs": [
            {"workflow_accuracy": row[0], "domain_accuracy": row[1], "mode": row[2], "top_k": row[3], "beta": row[4]}
            for row in configs[:8]
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
