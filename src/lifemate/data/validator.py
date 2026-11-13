from __future__ import annotations

from typing import Any

from .parser import parse_target, step_index


def validate_example(row: dict[str, Any], source_ids: set[str] | None = None) -> list[str]:
    errors = []
    sample_id, source_id, messages = row.get("id"), row.get("source_id"), row.get("messages")
    try:
        step_index(sample_id)
    except (TypeError, ValueError):
        errors.append("invalid id/step index")
    if not isinstance(source_id, str):
        errors.append("missing source_id")
    elif source_ids is not None and source_id not in source_ids:
        errors.append("source_id not in original train data")
    if not isinstance(messages, list) or len(messages) != 3:
        errors.append("messages must contain system/user/assistant")
        return errors
    if [m.get("role") if isinstance(m, dict) else None for m in messages] != ["system", "user", "assistant"]:
        errors.append("message role/order mismatch")
    if any(not isinstance(m, dict) or not isinstance(m.get("content"), str) for m in messages):
        errors.append("message content must be text")
    else:
        try:
            parse_target(messages[-1]["content"])
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def catalog_action_errors(actions: list[dict], catalog: dict, allowed: set[str]) -> list[str]:
    from zaic_tools import schema_validate
    return [f"{r.get('name')}: {r.get('error')}" for r in schema_validate(actions, catalog, allowed) if not r.get("ok")]
