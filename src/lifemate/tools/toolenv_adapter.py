from __future__ import annotations

import json
import sys
from pathlib import Path


class OfficialToolEnvAdapter:
    """Thin adapter. One official ToolEnv is constructed once per original sample."""
    def __init__(self, data_root: str | Path, split: str, sample_id: str):
        self.data_root = Path(data_root).resolve()
        package_dir = self.data_root / split
        if str(package_dir) not in sys.path:
            sys.path.insert(0, str(package_dir))
        from zaic_tools import ToolEnv
        self.split, self.sample_id = split, sample_id
        self.env = ToolEnv(split, sample_id, root=self.data_root)

    def validate(self, name: str, arguments: dict) -> dict:
        return self.env.validate(name, arguments)

    def call(self, name: str, arguments: dict) -> dict:
        return self.env.call(name, arguments)

    def get_tool_calls(self) -> list[dict]:
        return self.env.get_tool_calls()

    def get_state_hash(self) -> str:
        return self.env.get_state_hash()

    def get_state_for_evaluation(self) -> dict:
        """Explicitly opt into state inspection only after a labeled evaluation episode."""
        if self.split != "train":
            raise RuntimeError("raw state inspection is disabled outside labeled train evaluation")
        return self.env.get_state()


def load_catalog(data_root: str | Path, split: str) -> tuple[dict, dict[str, dict]]:
    path = Path(data_root) / split / f"{split}_tools_catalog.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    tools = {tool["name"]: tool for app in raw.get("apps", []) for tool in app.get("tools", [])}
    return raw, tools
