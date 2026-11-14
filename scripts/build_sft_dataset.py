"""Build causal ReAct JSON training examples using the inference prompt contract.

Includes history mining, app stripping and simulator-verified synthetic scenarios.
Never opens split environments. Tool results must already be present in labels.
"""
import argparse
import copy
import json
import inspect
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.agent.prompt import SYSTEM_PROMPT, build_user_prompt, format_tools_with_enums
from src.rules.catalog_filter import CatalogFilter
from src.agent.sft_input import RAW_SYSTEM_PROMPT, build_raw_user_prompt


def get_installed_apps(item):
    return set(item.get("context", {}).get("installed_apps", {}))


def get_gold_apps(calls):
    names = [(c.get("arguments", {}).get("tool_name", "") if c.get("name") == "sys.ask_missing_info"
              else c.get("name", "")) for c in calls]
    return {name.split(".")[0] for name in names if "." in name} - {"sys"}


def build_rows(item, catalog, *, full_history=False, raw_input=False):
    """One supervised decision per observed result; no future result/ID leakage."""
    context = item.get("context", {})
    installed = context.get("installed_apps", {})
    if isinstance(installed, list):
        installed = {app: {} for app in installed}
    allowed = catalog.get_allowed_tools(installed)
    allowed_names = {tool["name"] for tool in allowed}
    calls = item.get("label", {}).get("tool_calls", [])
    results = item.get("label", {}).get("tool_results", [])
    if any(call["name"] not in allowed_names for call in calls):
        raise ValueError(f"{item['id']}: gold call not in installed catalog")
    if len(results) != len(calls):
        raise ValueError(f"{item['id']}: missing aligned tool results; cannot teach causal execution")
    rows, steps = [], []
    history = item.get("history", [])
    allowed_tools = allowed  # Legacy upstream prompt inspects caller locals.
    explicit_history = "history" in inspect.signature(build_user_prompt).parameters
    for index in range(len(calls) + 1):
        options = ({"history": history, "allowed_tools": allowed, "exemplars": [],
                    "full_history": full_history} if explicit_history else {})
        if raw_input:
            prompt = build_raw_user_prompt(item.get("audio_text", ""), context,
                                           history, allowed, steps)
        else:
            prompt = build_user_prompt(
                item.get("audio_text", ""), context, "",
                format_tools_with_enums(allowed, ""), "", steps, **options)
        if full_history and not explicit_history and not raw_input:
            # Keep the incoming agent prompt untouched; adapt only the SFT exporter.
            memory = json.dumps([{k: v for k, v in turn.items() if k != "audio"}
                                 for turn in history], ensure_ascii=False) if history else "Không có lịch sử hội thoại trước đó."
            marker = "[ACTIVE MEMORY FROM PREVIOUS DIALOGUE]\n"
            before, rest = prompt.split(marker, 1)
            boundary = re.search(r"\n\n\[", rest)
            if boundary is None:
                raise ValueError("Upstream prompt has no history section boundary")
            prompt = before + marker + memory + rest[boundary.start():]
        actions = [calls[index]] if index < len(calls) else []
        finished = not actions or actions[0]["name"] == "sys.ask_missing_info"
        target = {"thought": "", "actions": actions, "is_finished": finished}
        rows.append({"id": f"{item['id']}#step{index}", "source_id": item["id"].split("#")[0],
                     "messages": [{"role": "system", "content": RAW_SYSTEM_PROMPT if raw_input else SYSTEM_PROMPT},
                                  {"role": "user", "content": prompt},
                                  {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}]})
        if finished:
            break
        result = results[index]
        steps.append({"step": index + 1, "tool_called": calls[index]["name"],
                      "arguments": calls[index].get("arguments", {}),
                      "observation": json.dumps(result, ensure_ascii=False), "raw_observation": result})
    return rows


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--catalog", default="data/train/train_tools_catalog.json")
    ap.add_argument("--synth-catalog", default="data/public_test/public_test_tools_catalog.json")
    ap.add_argument("--synth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    catalog = CatalogFilter(args.catalog)
    if args.synth:
        public_catalog = CatalogFilter(args.synth_catalog)
        for name, app in public_catalog.apps_by_id.items():
            catalog.apps_by_id.setdefault(name, app)
    rows = []
    counts = {"original": 0, "history_mined": 0, "oos_stripped": 0, "synth": 0}

    def add(item, kind):
        rows.extend(build_rows(item, catalog))
        counts[kind] += 1

    for item in read_jsonl(args.train):
        add(item, "original")
        history = item.get("history", [])
        for k in range(len(history) - 1):
            user, assistant = history[k:k + 2]
            text = user.get("audio_text") or user.get("content") or user.get("text")
            if user.get("role") == "user" and assistant.get("role") == "assistant" and text and assistant.get("tool_calls"):
                mined = copy.deepcopy(item)
                mined.update(id=f"{item['id']}#h{k}", audio_text=text, history=history[:k],
                             label={"tool_calls": assistant["tool_calls"],
                                    "tool_results": assistant.get("tool_results", [])})
                add(mined, "history_mined")
        calls = item.get("label", {}).get("tool_calls", [])
        used = get_gold_apps(calls)
        if used and not history:
            stripped = copy.deepcopy(item)
            installed = stripped["context"].get("installed_apps", {})
            stripped["context"]["installed_apps"] = ({app: spec for app, spec in installed.items() if app not in used}
                                                     if isinstance(installed, dict) else [app for app in installed if app not in used])
            stripped.update(id=f"{item['id']}#oos", label={"tool_calls": [], "tool_results": []})
            add(stripped, "oos_stripped")
    if args.synth:
        for item in read_jsonl(args.synth):
            add(item, "synth")
    random.Random(args.seed).shuffle(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"source_samples": counts, "decision_samples": len(rows), "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
