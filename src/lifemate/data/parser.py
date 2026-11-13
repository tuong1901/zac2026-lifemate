from __future__ import annotations

import json
import re
from typing import Any

TARGET_FIELDS = {"thought", "actions", "is_finished"}
STEP_PATTERN = re.compile(r"^(.+)#step(\d+)$")


def parse_target(content: str) -> dict[str, Any]:
    if not isinstance(content, str):
        raise ValueError("assistant target must be text")
    try:
        target = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"assistant target is not strict JSON: {exc}") from exc
    if not isinstance(target, dict) or set(target) != TARGET_FIELDS:
        raise ValueError(f"target keys must be exactly {sorted(TARGET_FIELDS)}")
    if not isinstance(target["thought"], str) or not isinstance(target["actions"], list) or not isinstance(target["is_finished"], bool):
        raise ValueError("target fields have invalid types")
    for i, action in enumerate(target["actions"]):
        if not isinstance(action, dict) or set(action) != {"name", "arguments"}:
            raise ValueError(f"actions[{i}] must contain name and arguments only")
        if not isinstance(action["name"], str) or not isinstance(action["arguments"], dict):
            raise ValueError(f"actions[{i}] has invalid name or arguments")
    return target


def step_index(sample_id: str) -> int:
    match = STEP_PATTERN.fullmatch(sample_id or "")
    if not match:
        raise ValueError(f"invalid SFT step id {sample_id!r}")
    return int(match.group(2))
