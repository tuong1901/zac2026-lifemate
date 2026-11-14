from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from lifemate.agent.controller import ToolController
from lifemate.agent.loop import run_episode
from lifemate.agent.state import WorkflowState
from lifemate.config import discover_original_data
from lifemate.data.loader import iter_jsonl
from lifemate.data.parser import parse_target
from lifemate.deployment.prompt import build_messages
from lifemate.evaluation.failure_analysis import categorize
from lifemate.models.loader import load_local_qwen
from lifemate.models.policy import QwenPolicy
from lifemate.tools.toolenv_adapter import OfficialToolEnvAdapter, load_catalog
from lifemate.utils.logging import JsonlLogger


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--original-dir", default=os.environ.get("ORIGINAL_DATA_DIR"))
    p.add_argument("--sft-dir", default=os.environ.get("SFT_DATA_DIR", str(repo.parent/"sft_full_history_v1")))
    p.add_argument("--model-dir", required=True)
    p.add_argument("--adapter-dir")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--max-input-tokens", type=int, default=8192)
    args = p.parse_args()
    candidates = [args.original_dir] if args.original_dir else [repo.parent/"train+public_test", repo.parent/"zalo-ai-challenge/train+public_test", "/kaggle/input/zalo-original-data"]
    data = discover_original_data(candidates)
    sft_dir = Path(args.sft_dir)
    split = json.loads((sft_dir/"split_manifest.json").read_text(encoding="utf-8"))
    split_sources = {src for src, value in split.items() if value == "validation"}
    train = [r for r in iter_jsonl(data/"train/train.jsonl") if r["id"] in split_sources]
    if args.max_samples:
        train = train[:args.max_samples]
    _, tool_specs = load_catalog(data, "train")
    model, tokenizer, model_info = load_local_qwen(args.model_dir, quantized=False, model_parallel=True)
    if args.adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=False)
    policy = QwenPolicy(model, tokenizer)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    log = JsonlLogger(out/"agent_steps.jsonl")
    results = []
    for i, item in enumerate(train, 1):
        current_text = item.get("audio_text", "")
        history_text = {turn["audio"]: turn.get("audio_text", "") for turn in item.get("history", []) if turn.get("role") == "user" and turn.get("audio")}
        messages = build_messages(item, current_text, tool_specs, history_text)
        token_count=len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
        context_limit=min(model_info.get("max_position_embeddings") or 32768,args.max_input_tokens)
        if token_count > context_limit:
            results.append({"id":item["id"],"tool_sequence_exact":False,"changed_final_state_match":False,
                           "task_success":False,"gold_tool_calls":item.get("label",{}).get("tool_calls",[]),
                           "predicted_tool_calls":[],"gold_final_state_apps":sorted(item.get("label",{}).get("final_state",{})),
                           "episode_status":"skipped_context","failure_category":"context_overflow",
                           "average_latency_seconds":0.0,"state_hash":None,"token_count":token_count})
            continue
        allowed = {t for app in item["context"]["installed_apps"].values() for t in app.get("tools", [])} | {"sys.ask_missing_info"}
        env = OfficialToolEnvAdapter(data, "train", item["id"])
        controller = ToolController(env, allowed, tool_specs, WorkflowState(task=item["id"]))
        episode = run_episode(sample_id=item["id"], initial_messages=messages, controller=controller,
                              policy=policy, max_steps=args.max_steps, log_sink=log)
        gold = item.get("label", {})
        gold_calls = gold.get("tool_calls", [])
        actual_calls = episode["tool_calls"]
        call_match = actual_calls == gold_calls
        expected_final = gold.get("final_state", {})
        actual_state = env.get_state_for_evaluation()
        state_match = all(actual_state.get(k) == v for k, v in expected_final.items())
        gold_finished = bool(gold_calls) and gold_calls[-1].get("name") == "sys.ask_missing_info"
        success = call_match and state_match and (episode["status"] == "finished" or (gold_finished and episode["status"] == "waiting_for_user"))
        row = {"id": item["id"], "tool_sequence_exact": call_match, "changed_final_state_match": state_match,
               "task_success": success, "gold_tool_calls": gold_calls, "predicted_tool_calls": actual_calls,
               "gold_final_state_apps": sorted(expected_final), "episode_status": episode["status"],
               "failure_category": None if success else categorize(call_match=call_match, state_match=state_match, status=episode["status"], calls=actual_calls),
               "average_latency_seconds": episode["latency_seconds"], "state_hash": episode["state_hash"],
               "max_loop_steps": args.max_steps, "model_output_steps": episode["steps"]}
        results.append(row)
        if i % 10 == 0:
            print(f"evaluated {i}/{len(train)} validation trajectories")
    n = len(results)
    summary = {"model": model_info, "adapter_dir": args.adapter_dir, "validation_trajectories": n,
               "task_completion_rate": sum(r["task_success"] for r in results)/n if n else None,
               "exact_tool_sequence_rate": sum(r["tool_sequence_exact"] for r in results)/n if n else None,
               "final_changed_state_match_rate": sum(r["changed_final_state_match"] for r in results)/n if n else None,
               "average_tool_calls": sum(len(r["predicted_tool_calls"]) for r in results)/n if n else None,
               "average_latency_seconds": sum(r["average_latency_seconds"] for r in results)/n if n else None,
               "skipped_context_count":sum(r.get("episode_status")=="skipped_context" for r in results),
               "failure_categories": {}, "results": results}
    for row in results:
        if row["failure_category"]:
            summary["failure_categories"][row["failure_category"]] = summary["failure_categories"].get(row["failure_category"], 0)+1
    (out/"trajectory_evaluation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out/"failure_cases.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in results if not r["task_success"]), encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
