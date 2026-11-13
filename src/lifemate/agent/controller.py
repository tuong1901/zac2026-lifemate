from __future__ import annotations

import json
from typing import Any

from .parser import parse_agent_output
from .state import WorkflowState
from .validator import validate_action


class ToolController:
    def __init__(self, env, allowed_tools: set[str], catalog: dict[str, dict], state: WorkflowState | None = None):
        self.env = env
        self.allowed = set(allowed_tools) | {"sys.ask_missing_info"}
        self.catalog = catalog
        self.state = state or WorkflowState(task="current_request")
        self.write_tools = {name for name, spec in catalog.items() if spec.get("kind") == "write"}

    def parse(self, raw: str) -> dict:
        return parse_agent_output(raw)

    def execute(self, output: dict) -> list[dict]:
        if output["is_finished"] and self.state.pending:
            raise ValueError("refusing completion while controller has unresolved pending items")
        actions = output["actions"]
        for action in actions:
            validate_action(action, allowed=self.allowed, catalog=self.catalog, env=self.env)
            key = (action["name"], json.dumps(action["arguments"], sort_keys=True, ensure_ascii=False))
            if key in self.state.completed_write_keys(self.write_tools):
                raise ValueError(f"refusing repeated successful side effect: {action['name']}")
        observations = []
        for action in actions:
            result = self.env.call(action["name"], action["arguments"])
            if not isinstance(result, dict):
                raise TypeError("official tool result must be an object")
            self.state.record_result(action, result)
            observations.append(result)
            if action["name"] == "sys.ask_missing_info" and result.get("ok") is True:
                self.state.waiting_for_user = True
                self.state.status = "waiting_for_user"
        if output["is_finished"]:
            self.state.status = "finished"
        return observations
