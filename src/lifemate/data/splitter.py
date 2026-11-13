from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_group_split(path: str | Path, sources: set[str]) -> dict[str, str]:
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or set(mapping) != sources:
        raise ValueError("split manifest must assign every and only source_id")
    if any(v not in {"train", "validation"} for v in mapping.values()):
        raise ValueError("unknown split value")
    return mapping


def split_hash(mapping: dict[str, str]) -> str:
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
