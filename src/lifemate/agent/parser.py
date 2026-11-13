from __future__ import annotations

import json


class AgentOutputError(ValueError):
    pass


def parse_agent_output(raw: str) -> dict:
    try:
        result = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise AgentOutputError(f"model output is not strict JSON: {exc}") from exc
    if not isinstance(result, dict) or set(result) != {"thought", "actions", "is_finished"}:
        raise AgentOutputError("expected exactly thought, actions, is_finished")
    if not isinstance(result["thought"], str) or not isinstance(result["actions"], list) or not isinstance(result["is_finished"], bool):
        raise AgentOutputError("agent output has invalid field types")
    if len(result["actions"]) > 1:
        raise AgentOutputError("one action per decision is required so the model observes each tool result before choosing again")
    if not result["actions"] and not result["is_finished"]:
        raise AgentOutputError("an unfinished decision must execute one tool or ask for missing information")
    for i, action in enumerate(result["actions"]):
        if not isinstance(action, dict) or set(action) != {"name", "arguments"}:
            raise AgentOutputError(f"actions[{i}] must contain exactly name and arguments")
        if not isinstance(action["name"], str) or not isinstance(action["arguments"], dict):
            raise AgentOutputError(f"actions[{i}] has invalid types")
    return result
