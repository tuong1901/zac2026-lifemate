from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        result = _expand(yaml.safe_load(f) or {})
    if not isinstance(result, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return result


def resolve_path(value: str | Path, repo_root: str | Path) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    return (expanded if expanded.is_absolute() else Path(repo_root) / expanded).resolve()


def discover_original_data(candidates: list[str | Path]) -> Path:
    required = ("train/train.jsonl", "public_test/public_test.jsonl", "train/zaic_tools")
    for candidate in candidates:
        p = Path(candidate).expanduser()
        if p.exists() and all((p / rel).exists() for rel in required):
            return p.resolve()
    raise FileNotFoundError("Could not discover official data directory; set ORIGINAL_DATA_DIR to a root containing train/ and public_test/.")
