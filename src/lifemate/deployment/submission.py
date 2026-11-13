from __future__ import annotations

import json
import re
from pathlib import Path

HASH_RE = re.compile(r"^[0-9a-f]{64}$")
FIELDS = {"id", "tool_calls", "state_hash"}


def validate_submission(path: str | Path, expected_ids: list[str]) -> dict:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {n}")
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != FIELDS:
                raise ValueError(f"line {n}: required keys are exactly {sorted(FIELDS)}")
            if not isinstance(row["id"], str) or not isinstance(row["tool_calls"], list) or not isinstance(row["state_hash"], str) or not HASH_RE.fullmatch(row["state_hash"]):
                raise ValueError(f"line {n}: invalid field type or state_hash format")
            for i, call in enumerate(row["tool_calls"]):
                if not isinstance(call, dict) or set(call) != {"name", "arguments"} or not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
                    raise ValueError(f"line {n} call {i}: malformed tool call")
            rows.append(row)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("submission contains duplicate IDs")
    if set(ids) != set(expected_ids):
        raise ValueError(f"sample ID mismatch: missing={len(set(expected_ids)-set(ids))}, extra={len(set(ids)-set(expected_ids))}")
    return {"records": len(rows), "unique_ids": len(set(ids)), "id_set_matches": True}
