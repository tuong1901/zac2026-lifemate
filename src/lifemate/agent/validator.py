from __future__ import annotations

from typing import Any


def validate_action(action: dict[str, Any], *, allowed: set[str], catalog: dict, env) -> None:
    name = action["name"]
    if name not in allowed:
        raise ValueError(f"tool is not authorized for this sample: {name}")
    if name not in catalog:
        raise ValueError(f"tool is absent from official catalog: {name}")
    result = env.validate(name, action["arguments"])
    if not result.get("ok"):
        raise ValueError(f"official ToolEnv rejected {name}: {result.get('error')}")
