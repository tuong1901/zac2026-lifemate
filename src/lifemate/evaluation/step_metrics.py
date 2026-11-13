from __future__ import annotations

import json

from lifemate.agent.parser import parse_agent_output


def normalized(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def score_prediction(raw: str, gold: dict, catalog: dict, allowed: set[str]) -> dict:
    row = {"json_valid": False, "exact_match": False, "action_name_correct": False,
           "arguments_correct": False, "is_finished_correct": False, "decision_correct": False,
           "invalid_tool": False, "prediction": None, "error": None}
    try:
        pred = parse_agent_output(raw)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    row["json_valid"] = True
    row["prediction"] = pred
    pred_actions, gold_actions = pred["actions"], gold["actions"]
    row["action_name_correct"] = bool(pred_actions and gold_actions and pred_actions[0]["name"] == gold_actions[0]["name"]) or (not pred_actions and not gold_actions)
    row["arguments_correct"] = bool(pred_actions and gold_actions and pred_actions[0]["name"] == gold_actions[0]["name"] and normalized(pred_actions[0]["arguments"]) == normalized(gold_actions[0]["arguments"])) or (not pred_actions and not gold_actions)
    row["is_finished_correct"] = pred["is_finished"] == gold["is_finished"]
    row["decision_correct"] = row["action_name_correct"] and row["arguments_correct"] and row["is_finished_correct"]
    bad = [a for a in pred_actions if a["name"] not in catalog or a["name"] not in allowed]
    row["invalid_tool"] = bool(bad)
    row["exact_match"] = normalized({k: pred[k] for k in gold}) == normalized(gold)
    return row


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    keys = ("json_valid", "exact_match", "action_name_correct", "arguments_correct", "is_finished_correct", "decision_correct", "invalid_tool")
    return {"evaluated": n, **{k: sum(bool(r[k]) for r in rows)/n if n else None for k in keys}}
