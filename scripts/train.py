from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lifemate.training.trainer import train


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Offline QLoRA SFT for local Qwen3-4B-Instruct-2507 assets")
    p.add_argument("--model-dir", required=True, help="Local model folder; no network fallback")
    p.add_argument("--train-jsonl", default=str(repo / "data/processed/train.jsonl"))
    p.add_argument("--validation-jsonl", default=str(repo / "data/processed/validation.jsonl"))
    p.add_argument("--output-dir", default=str(repo / "artifacts/lora/run1"))
    p.add_argument("--max-seq-len", type=int)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from-checkpoint")
    args = p.parse_args()
    if not Path(args.model_dir).is_dir():
        p.error(f"model directory does not exist: {args.model_dir}")
    if not Path(args.train_jsonl).is_file() or not Path(args.validation_jsonl).is_file():
        p.error("prepared train/validation files are missing; run scripts/prepare_data.py first")
    result = train(args)
    print(json.dumps({k: v for k, v in result.items() if k != "hardware"}, indent=2, default=str))


if __name__ == "__main__":
    main()
