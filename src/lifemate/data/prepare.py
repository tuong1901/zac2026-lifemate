from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from .loader import iter_jsonl, write_jsonl
from .parser import parse_target, step_index
from .splitter import load_group_split
from .trajectory import group_trajectories
from .validator import catalog_action_errors, validate_example


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(sft_dir: Path, original_dir: Path, output_dir: Path) -> dict:
    # Import only the official schema code after locating the supplied package.
    import sys
    train_pkg = original_dir / "train"
    sys.path.insert(0, str(train_pkg))
    from zaic_tools import schema_validate

    original = {row["id"]: row for row in iter_jsonl(train_pkg / "train.jsonl")}
    catalog = json.loads((train_pkg / "train_tools_catalog.json").read_text(encoding="utf-8"))
    canonical = []
    for filename in ("train_all_lengths.jsonl", "validation_all_lengths.jsonl"):
        canonical.extend(iter_jsonl(sft_dir / filename))
    groups = group_trajectories(canonical)
    sources = set(groups)
    unknown, absent = sources - set(original), set(original) - sources
    if unknown or absent:
        raise ValueError(f"source mapping mismatch: unknown={len(unknown)}, absent={len(absent)}")
    assignment = load_group_split(sft_dir / "split_manifest.json", sources)
    output = {"train": [], "validation": []}
    bad = []
    for source, trajectory in groups.items():
        sample = original[source]
        allowed = {t for app in sample["context"]["installed_apps"].values() for t in app.get("tools", [])} | {"sys.ask_missing_info"}
        split = assignment[source]
        for row in trajectory:
            errors = validate_example(row, set(original))
            if not errors:
                target = parse_target(row["messages"][-1]["content"])
                errors.extend(catalog_action_errors(target["actions"], catalog, allowed))
            if errors:
                bad.append({"id": row.get("id"), "source_id": source, "errors": errors})
            else:
                output[split].append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    if bad:
        (output_dir / "data_exclusions.json").write_text(json.dumps(bad, ensure_ascii=False, indent=2), encoding="utf-8")
        raise ValueError(f"Found {len(bad)} invalid records; no label was repaired; see data_exclusions.json")
    for split in output:
        output[split].sort(key=lambda r: (r["source_id"], step_index(r["id"])))
        write_jsonl(output_dir / f"{split}.jsonl", output[split])
    lengths_path = sft_dir / "lengths.csv"
    lengths = {}
    with lengths_path.open(encoding="utf-8-sig", newline="") as f:
        for item in csv.DictReader(f):
            lengths[item["id"]] = item
    totals = [int(lengths[r["id"]]["total_tokens"]) for rows in output.values() for r in rows]
    sorted_totals = sorted(totals)
    train_sources = {r["source_id"] for r in output["train"]}
    validation_sources = {r["source_id"] for r in output["validation"]}
    manifest = {
        "source_split": assignment,
        "train_sources": len(train_sources), "validation_sources": len(validation_sources),
        "train_steps": len(output["train"]), "validation_steps": len(output["validation"]),
        "source_overlap": sorted(train_sources & validation_sources), "excluded_count": 0,
        "token_lengths": {"source": "existing Qwen tokenizer lengths.csv; not recomputed", "median": sorted_totals[len(sorted_totals)//2],
                          "p90": sorted_totals[int(.9*len(sorted_totals))], "max": max(sorted_totals)},
        "original_train_sha256": sha256(train_pkg / "train.jsonl"),
        "split_manifest_sha256": sha256(sft_dir / "split_manifest.json"),
        "public_test_used_for_training": False,
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
