"""Report tool planning and argument errors from clean validation traces."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"total": 0}
    counts = Counter()
    examples = defaultdict(list)
    for row in rows:
        gold = row["gold_calls"]
        pred = row["pred_calls"]
        gold_names = [call["name"] for call in gold]
        pred_names = [call["name"] for call in pred]
        same_set = Counter(gold_names) == Counter(pred_names)
        same_order = gold_names == pred_names
        same_args = same_order and all(
            (g.get("arguments") or {}) == (p.get("arguments") or {})
            for g, p in zip(gold, pred)
        )
        for key, ok in (("tool_multiset", same_set), ("tool_sequence", same_order),
                        ("arguments", same_args), ("state", row["state_match"])):
            counts[key] += int(ok)
            if not ok and len(examples[key]) < 5:
                examples[key].append(row["id"])
        if not same_set:
            counts["wrong_tool_selection"] += 1
        elif not same_order:
            counts["wrong_order"] += 1
        elif not same_args:
            counts["wrong_arguments"] += 1
        elif not row["state_match"]:
            counts["state_or_runtime_mismatch"] += 1
    return {"total": n, "counts": dict(counts), "sample_ids": dict(examples)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_file", type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    with args.trace_file.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                groups[row["source"]].append(row)
    for source, rows in groups.items():
        print(source, json.dumps(summarize(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
