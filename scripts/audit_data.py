from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

from lifemate.config import discover_original_data
from lifemate.data.loader import iter_jsonl
from lifemate.data.parser import parse_target, step_index
from lifemate.data.trajectory import group_trajectories
from lifemate.data.validator import validate_example
from lifemate.utils.hardware import inspect_hardware


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--sft-dir", default=os.environ.get("SFT_DATA_DIR", str(repo.parent/"sft_full_history_v1")))
    p.add_argument("--original-dir", default=os.environ.get("ORIGINAL_DATA_DIR"))
    p.add_argument("--output-dir", default=str(repo/"artifacts/audit"))
    args = p.parse_args()
    candidates = [args.original_dir] if args.original_dir else [repo.parent/"train+public_test", repo.parent/"zalo-ai-challenge/train+public_test", "/kaggle/input/zalo-original-data"]
    data = discover_original_data(candidates)
    sft = Path(args.sft_dir)
    train = list(iter_jsonl(data/"train/train.jsonl")); public = list(iter_jsonl(data/"public_test/public_test.jsonl"))
    steps = list(iter_jsonl(sft/"train_all_lengths.jsonl"))+list(iter_jsonl(sft/"validation_all_lengths.jsonl"))
    split = json.loads((sft/"split_manifest.json").read_text(encoding="utf-8"))
    ids = {r["id"] for r in train}
    invalid = [{"id": r.get("id"), "errors": validate_example(r, ids)} for r in steps if validate_example(r, ids)]
    trajectories = group_trajectories(steps)
    tools, action_counts, finished = collections.Counter(), collections.Counter(), collections.Counter()
    for row in steps:
        target = parse_target(row["messages"][-1]["content"])
        action_counts[len(target["actions"])] += 1
        finished[str(target["is_finished"])] += 1
        tools.update(a["name"] for a in target["actions"])
    split_sources = {name: {r["source_id"] for r in steps if split[r["source_id"]] == name} for name in ("train", "validation")}
    trajectory_stats = []
    for source, group in sorted(trajectories.items()):
        idx = [step_index(r["id"]) for r in group]
        trajectory_stats.append({"source_id": source, "steps": len(group), "step_indices": json.dumps(idx)})
    result = {"data_dir": str(data), "sft_dir": str(sft), "sft_source_file_sha256": sha256(data/"train/train.jsonl"),
              "split_manifest_sha256": sha256(sft/"split_manifest.json"), "train_samples": len(train), "public_test_samples": len(public),
              "sft_steps": len(steps), "source_trajectories": len(trajectories), "source_match_count": len(set(trajectories)&ids),
              "source_split": {"train": len(split_sources["train"]), "validation": len(split_sources["validation"]),
                               "overlap": sorted(split_sources["train"] & split_sources["validation"])},
              "role_counts": dict(collections.Counter(m.get("role") for r in steps for m in r.get("messages", []))),
              "action_counts": dict(action_counts), "finished_counts": dict(finished), "distinct_tools": len(tools),
              "invalid_examples": invalid, "hardware": inspect_hardware([repo, data, sft])}
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out/"data_quality_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out/"tool_frequency.csv").write_text("tool_name,frequency\n"+"".join(f"{name},{count}\n" for name,count in tools.most_common()), encoding="utf-8-sig")
    (out/"trajectory_statistics.csv").write_text("source_id,steps,step_indices\n"+"".join(f"{r['source_id']},{r['steps']},\"{r['step_indices']}\"\n" for r in trajectory_stats), encoding="utf-8-sig")
    inventory = {"sft_jsonl_files": [{"path": str(p.relative_to(sft)), "bytes": p.stat().st_size, "rows": sum(1 for _ in iter_jsonl(p))}
                                      for p in sorted(sft.rglob("*.jsonl"))],
                 "official_train_count": len(train), "public_test_count": len(public),
                 "catalogs": {name: {"apps": len(json.loads((data/name/f"{name}_tools_catalog.json").read_text(encoding="utf-8")).get("apps", [])),
                                     "tools": sum(len(a.get("tools", [])) for a in json.loads((data/name/f"{name}_tools_catalog.json").read_text(encoding="utf-8")).get("apps", []))}
                              for name in ("train", "public_test")},
                 "environment_state_counts": {name: len(list((data/name/f"{name}_environment").glob("*/state.json"))) for name in ("train", "public_test")},
                 "hardware": result["hardware"]}
    (out/"data_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="hardware"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
