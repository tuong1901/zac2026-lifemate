from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from lifemate.agent.parser import parse_agent_output
from lifemate.data.loader import iter_jsonl
from lifemate.data.parser import parse_target
from lifemate.evaluation.step_metrics import score_prediction, summarize
from lifemate.models.loader import load_local_qwen
from lifemate.tools.toolenv_adapter import load_catalog


@torch.inference_mode()
def generate(model, tokenizer, messages, max_new_tokens):
    enc = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    device = model.get_input_embeddings().weight.device
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0, enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--adapter-dir")
    p.add_argument("--data", default=str(repo / "data/processed/validation.jsonl"))
    p.add_argument("--original-dir", default=os.environ.get("ORIGINAL_DATA_DIR", str(repo.parent / "train+public_test")))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-input-tokens", type=int, default=8192)
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.original_dir) / "train"))
    from zaic_tools import schema_validate
    model, tokenizer, model_info = load_local_qwen(args.model_dir, quantized=True)
    if args.adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=False)
    _, catalog = load_catalog(args.original_dir, "train")
    original_by_id = {r["id"]: r for r in iter_jsonl(Path(args.original_dir)/"train/train.jsonl")}
    data = list(iter_jsonl(args.data))
    if args.max_samples:
        data = data[:args.max_samples]
    records, outputs, skipped = [], [], []
    for i, sample in enumerate(data):
        prompt = sample["messages"]
        inp = tokenizer.apply_chat_template(prompt[:-1], tokenize=True, add_generation_prompt=True)
        context_limit=model_info.get("max_position_embeddings") or 32768
        if len(inp) > min(context_limit,args.max_input_tokens):
            skipped.append({"id": sample["id"], "reason": "configured_context_limit_exceeded", "tokens": len(inp),"limit":min(context_limit,args.max_input_tokens)})
            continue
        start = time.perf_counter()
        raw = generate(model, tokenizer, prompt, args.max_new_tokens)
        elapsed = time.perf_counter()-start
        target = parse_target(prompt[-1]["content"])
        source = sample["source_id"]
        original = None
        # The SFT context embeds the installed-tools list. Extract its source sample once through JSONL lookup cache.
        original = original_by_id.get(source)
        if original is None:
            raise ValueError(f"cannot map validation source {source} to official train")
        allowed = {t for app in original["context"]["installed_apps"].values() for t in app.get("tools", [])} | {"sys.ask_missing_info"}
        result = score_prediction(raw, target, catalog, allowed)
        result.update({"id": sample["id"], "source_id": source, "latency_seconds": elapsed, "model_output": raw})
        if result["prediction"]:
            schema = schema_validate(result["prediction"]["actions"], catalog, allowed)
            result["invalid_tool"] = any(not x["ok"] for x in schema)
        records.append(result)
        outputs.append({"id": sample["id"], "source_id": source, "model_output": raw})
        if (i+1) % 25 == 0:
            print(f"evaluated {i+1}/{len(data)}")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out/"predictions.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in outputs), encoding="utf-8")
    metrics = {"model": model_info, "adapter_dir": args.adapter_dir, "data": args.data,
               "metrics": summarize(records), "mean_latency_seconds": sum(r["latency_seconds"] for r in records)/len(records) if records else None,
               "skipped": skipped, "failures": [r for r in records if not r["decision_correct"]]}
    (out/"metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"metrics": metrics["metrics"], "skipped": len(skipped), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
