from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

from lifemate.config import discover_original_data
from lifemate.data.loader import iter_jsonl


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--original-dir", default=os.environ.get("ORIGINAL_DATA_DIR"))
    p.add_argument("--sft-dir", default=os.environ.get("SFT_DATA_DIR", str(repo.parent / "sft_full_history_v1")))
    p.add_argument("--output", default=str(repo / "artifacts/audit/toolenv_verification.json"))
    args = p.parse_args()
    candidates = [args.original_dir] if args.original_dir else [repo.parent / "train+public_test", repo.parent / "zalo-ai-challenge/train+public_test", "/kaggle/input/zalo-original-data"]
    data = discover_original_data(candidates)
    sys.path.insert(0, str(data / "train"))
    from zaic_tools import ToolEnv
    train = {r["id"]: r for r in iter_jsonl(data / "train/train.jsonl")}
    sft_dir = Path(args.sft_dir)
    sft = list(iter_jsonl(sft_dir / "train_all_lengths.jsonl")) + list(iter_jsonl(sft_dir / "validation_all_lengths.jsonl"))
    by_source = collections.defaultdict(list)
    for row in sft:
        by_source[row["source_id"]].append(row)
    errors, sequence_errors = [], []
    total_calls = 0
    for sample_id, sample in train.items():
        label = sample["label"]
        env = ToolEnv("train", sample_id, root=data)
        actual = [env.call(c["name"], c.get("arguments") or {}) for c in label.get("tool_calls", [])]
        total_calls += len(actual)
        if actual != label.get("tool_results", []):
            errors.append({"id": sample_id, "kind": "tool_results"})
        state = env.get_state()
        if any(state.get(k) != v for k, v in label.get("final_state", {}).items()):
            errors.append({"id": sample_id, "kind": "final_state"})
        if any(not env.validate(c["name"], c.get("arguments") or {}).get("ok") for c in label.get("tool_calls", [])):
            errors.append({"id": sample_id, "kind": "schema"})
        seq = []
        try:
            for row in sorted(by_source[sample_id], key=lambda r: int(r["id"].rsplit("#step", 1)[1])):
                seq.extend(json.loads(row["messages"][-1]["content"])["actions"])
        except Exception as exc:
            sequence_errors.append({"id": sample_id, "kind": "parse", "error": str(exc)})
        else:
            if seq != label.get("tool_calls", []):
                sequence_errors.append({"id": sample_id, "kind": "action_sequence"})
    result = {"scope": "train only; public_test was not read or replayed", "toolenv": str(data / "train/zaic_tools"),
              "samples": len(train), "sft_steps": len(sft), "tool_calls": total_calls,
              "replay_discrepancies": errors, "sft_sequence_discrepancies": sequence_errors,
              "pass": not errors and not sequence_errors}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"replay_discrepancies", "sft_sequence_discrepancies"}}, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
