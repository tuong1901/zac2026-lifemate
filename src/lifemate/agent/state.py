from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class WorkflowState:
    task: str
    status: str = "in_progress"
    completed: list[dict[str, Any]] = field(default_factory=list)
    current_step: str | None = None
    pending: list[str] = field(default_factory=list)
    entities: dict[str, Any] = field(default_factory=dict)
    waiting_for_user: bool = False
    observations: list[dict[str, Any]] = field(default_factory=list)

    def record_result(self, action: dict[str, Any], result: dict[str, Any]) -> None:
        event = {"action": action, "result": result}
        self.observations.append(event)
        if result.get("ok") is True and result.get("done") is not False:
            self.completed.append(event)

    def completed_write_keys(self, write_tools: set[str]) -> set[tuple[str, str]]:
        import json
        return {(event["action"]["name"], json.dumps(event["action"].get("arguments", {}), sort_keys=True, ensure_ascii=False))
                for event in self.completed if event["action"]["name"] in write_tools}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
