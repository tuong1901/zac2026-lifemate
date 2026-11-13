from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from lifemate.agent.controller import ToolController
from lifemate.agent.loop import run_episode
from lifemate.agent.state import WorkflowState
from lifemate.config import discover_original_data
from lifemate.data.loader import iter_jsonl
from lifemate.deployment.asr import LocalWhisperASR
from lifemate.deployment.prompt import build_messages
from lifemate.models.loader import load_local_qwen
from lifemate.models.policy import QwenPolicy
from lifemate.tools.toolenv_adapter import OfficialToolEnvAdapter, load_catalog
from lifemate.utils.logging import JsonlLogger


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Offline public_test inference using local Qwen, local ASR and official ToolEnv")
    p.add_argument("--original-dir", default=os.environ.get("ORIGINAL_DATA_DIR"))
    p.add_argument("--model-dir", default=os.environ.get("BASE_MODEL_DIR"), required=False)
    p.add_argument("--adapter-dir", default=os.environ.get("ADAPTER_DIR"))
    p.add_argument("--asr-model-dir", default=os.environ.get("ASR_MODEL_DIR"))
    p.add_argument("--output", default=str(repo / "result/submission.jsonl"))
    p.add_argument("--max-samples", type=int)
    p.add_argument("--max-steps", type=int, default=12)
    args = p.parse_args()
    if not args.model_dir or not Path(args.model_dir).is_dir():
        p.error("--model-dir/BASE_MODEL_DIR must point to local Qwen3-4B-Instruct-2507 assets")
    if not args.asr_model_dir or not Path(args.asr_model_dir).is_dir():
        p.error("--asr-model-dir/ASR_MODEL_DIR must point to a local Vietnamese-capable ASR model")
    candidates = [args.original_dir] if args.original_dir else [repo.parent/"train+public_test", repo.parent/"zalo-ai-challenge/train+public_test", "/kaggle/input/zalo-original-data"]
    data = discover_original_data(candidates)
    rows = list(iter_jsonl(data/"public_test/public_test.jsonl"))
    if args.max_samples:
        rows = rows[:args.max_samples]
    _, tool_specs = load_catalog(data, "public_test")
    model, tokenizer, model_info = load_local_qwen(args.model_dir, quantized=True)
    if args.adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=False)
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    asr = LocalWhisperASR(args.asr_model_dir)
    policy = QwenPolicy(model, tokenizer)
    output_path = Path(args.output); output_path.parent.mkdir(parents=True, exist_ok=True)
    log = JsonlLogger(output_path.parent / "agent_steps.jsonl")
    with output_path.open("w", encoding="utf-8", newline="\n") as fout:
        for i, item in enumerate(rows, 1):
            audio_cache = {}
            for turn in item.get("history", []):
                ref = turn.get("audio")
                if turn.get("role") == "user" and ref and ref not in audio_cache:
                    audio_cache[ref] = asr.transcribe(data/"public_test"/ref)
            final_ref = item["audio"]
            if final_ref not in audio_cache:
                audio_cache[final_ref] = asr.transcribe(data/"public_test"/final_ref)
            messages = build_messages(item, audio_cache[final_ref], tool_specs, audio_cache)
            allowed = {t for app in item["context"]["installed_apps"].values() for t in app.get("tools", [])} | {"sys.ask_missing_info"}
            env = OfficialToolEnvAdapter(data, "public_test", item["id"])
            controller = ToolController(env, allowed, tool_specs, WorkflowState(task=item["id"]))
            result = run_episode(sample_id=item["id"], initial_messages=messages, controller=controller,
                                 policy=policy, max_steps=args.max_steps, log_sink=log)
            fout.write(json.dumps({"id": item["id"], "tool_calls": result["tool_calls"], "state_hash": result["state_hash"]}, ensure_ascii=False)+"\n")
            fout.flush()
            if i % 25 == 0:
                print(f"completed {i}/{len(rows)} public samples")
    manifest = {"count": len(rows), "model": model_info, "adapter": args.adapter_dir, "asr_model_dir": args.asr_model_dir,
                "data_dir": str(data), "output": str(output_path), "offline_local_assets_only": True}
    (output_path.parent/"public_inference_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
