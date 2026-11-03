"""Build semantic chunks for intent routing, workflows, and argument provenance."""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.agent.prompt import READ_TOOLS


def tool_sequence(item: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(call.get("name", "") for call in item.get("label", {}).get("tool_calls", []))


def domain_sequence(sequence: Iterable[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(name.split(".", 1)[0] for name in sequence if name))


def workflow_id(sequence: Iterable[str]) -> str:
    raw = "|".join(sequence)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def history_signals(item: Dict[str, Any]) -> Dict[str, Any]:
    user_turns: List[str] = []
    tools: List[str] = []
    for turn in item.get("history", []):
        if turn.get("role") == "user" and turn.get("audio_text"):
            user_turns.append(turn["audio_text"])
        elif turn.get("role") == "assistant":
            tools.extend(call.get("name", "") for call in turn.get("tool_calls", []) if call.get("name"))
    return {
        "last_user_utterance": user_turns[-1] if user_turns else "",
        "user_utterances": user_turns,
        "prior_tools": tools,
    }


def _text_values(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for nested in value.values():
            yield from _text_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _text_values(nested)
    elif isinstance(value, (str, int, float)):
        yield str(value)


def infer_argument_source(key: str, value: Any, item: Dict[str, Any]) -> str:
    current = item.get("audio_text", "").lower()
    history_blob = " ".join(_text_values(item.get("history", []))).lower()
    value_text = str(value).lower()
    if key.endswith("_id") or key in {"chat_id", "contact_id", "task_id", "event_id", "lock_id"}:
        if value_text and value_text in history_blob:
            return "history_observation"
        return "tool_observation"
    if key in {"time", "start", "end", "due", "date", "date_from", "date_to", "schedule_time"}:
        return "current_time_and_utterance"
    if isinstance(value, bool):
        return "intent_constant"
    if value_text and value_text in current:
        return "current_utterance"
    if value_text and value_text in history_blob:
        return "history"
    return "utterance_or_derived"


def build_structured_chunks(
    items: Iterable[Dict[str, Any]],
    exclude_ids: Optional[Set[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    exclude_ids = exclude_ids or set()
    intent_chunks: List[Dict[str, Any]] = []
    argument_chunks: List[Dict[str, Any]] = []
    workflow_groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)

    for item in items:
        sample_id = item.get("id", "")
        if sample_id in exclude_ids:
            continue
        sequence = tool_sequence(item)
        domains = domain_sequence(sequence)
        signals = history_signals(item)
        intent_chunks.append({
            "chunk_id": f"intent:{sample_id}",
            "kind": "intent",
            "sample_id": sample_id,
            "text": item.get("audio_text", ""),
            "history_signals": signals,
            "installed_apps": sorted(item.get("context", {}).get("installed_apps", {}).keys()),
            "domains": list(domains),
            "workflow_id": workflow_id(sequence),
            "tool_sequence": list(sequence),
        })
        workflow_groups[sequence].append(item)

        for index, call in enumerate(item.get("label", {}).get("tool_calls", [])):
            arguments = call.get("arguments") or {}
            argument_chunks.append({
                "chunk_id": f"argument:{sample_id}:{index}",
                "kind": "argument",
                "sample_id": sample_id,
                "tool": call.get("name", ""),
                "argument_keys": list(arguments.keys()),
                "bindings": {
                    key: infer_argument_source(key, value, item)
                    for key, value in arguments.items()
                },
            })

    workflow_chunks: List[Dict[str, Any]] = []
    for sequence, examples in workflow_groups.items():
        reads = [name for name in sequence if name in READ_TOOLS]
        actions = [name for name in sequence if name not in READ_TOOLS]
        workflow_chunks.append({
            "chunk_id": f"workflow:{workflow_id(sequence)}",
            "kind": "workflow",
            "workflow_id": workflow_id(sequence),
            "domains": list(domain_sequence(sequence)),
            "tool_sequence": list(sequence),
            "read_tools": reads,
            "action_tools": actions,
            "example_count": len(examples),
            "example_ids": [item.get("id", "") for item in examples],
            "example_utterances": [item.get("audio_text", "") for item in examples[:8]],
        })

    return {
        "intent": intent_chunks,
        "workflow": workflow_chunks,
        "argument": argument_chunks,
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_chunks(chunks: Dict[str, List[Dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for kind, records in chunks.items():
        path = output_dir / f"{kind}_chunks.jsonl"
        with path.open("w", encoding="utf-8") as target:
            for record in records:
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
