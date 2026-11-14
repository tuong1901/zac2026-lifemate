from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path


def choose_sequence_length(total_vram_bytes: int) -> int:
    """Conservative sequence cap for single-GPU FP16 LoRA; longer rows are logged/excluded."""
    gib = total_vram_bytes / (1024**3)
    if gib >= 40:
        return 16384
    if gib >= 24:
        return 12288
    if gib >= 14:
        return 8192
    raise RuntimeError(f"FP16 Qwen training requires at least 14 GiB GPU VRAM; detected {gib:.1f} GiB")


def check_training_hardware(hw: dict) -> dict:
    if platform.system() != "Linux" or not hw.get("cuda_available"):
        raise RuntimeError("FP16 LoRA training requires Linux and CUDA. No model was downloaded.")
    gpus = hw.get("gpu", [])
    if not gpus:
        raise RuntimeError("No CUDA GPU was detected. No model was downloaded.")
    gpu = gpus[0]
    if "T4" not in gpu["name"].upper():
        raise RuntimeError(f"This Kaggle run is configured for Tesla T4; detected {gpu['name']}. No model was downloaded.")
    if gpu["total_vram_bytes"] < 14 * 1024**3:
        raise RuntimeError(f"FP16 Qwen training requires at least 14 GiB on the selected GPU; detected {gpu['total_vram_bytes']/1024**3:.1f} GiB. Select a larger GPU or revise the model/precision requirements; no fallback is automatic.")
    if gpu["free_vram_bytes"] < 12 * 1024**3:
        raise RuntimeError(f"FP16 Qwen training requires at least 12 GiB free before model download; detected {gpu['free_vram_bytes']/1024**3:.1f} GiB. Stop competing GPU jobs or select a clean Kaggle session.")
    return gpu


def train(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    from lifemate.data.formatter import encode_assistant_only
    from lifemate.data.loader import iter_jsonl
    from lifemate.models.loader import discover_lora_targets, inspect_local_model
    from lifemate.training.collator import AssistantOnlyCollator
    from lifemate.utils.hardware import inspect_hardware

    hw = inspect_hardware([args.train_jsonl, args.validation_jsonl, args.output_dir])
    gpu = check_training_hardware(hw)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("This FP16 Kaggle configuration uses one GPU and ordinary Trainer; do not launch with torchrun/DDP.")
    seq_len = args.max_seq_len or choose_sequence_length(gpu["total_vram_bytes"])
    if seq_len > 16384:
        raise ValueError("This workflow caps sequence length at 16384")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    set_seed(args.seed)

    model_dir = Path(args.model_dir).resolve()
    model_info = inspect_local_model(model_dir)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="sdpa")
    if next(model.parameters()).dtype != torch.float16:
        raise RuntimeError("Base model did not load in FP16; refusing precision fallback")
    targets = discover_lora_targets(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=targets))
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def encode(rows):
        kept, excluded = [], []
        for row in rows:
            feature = encode_assistant_only(row["messages"], tokenizer)
            if len(feature["input_ids"]) > seq_len:
                excluded.append({"id": row["id"], "source_id": row["source_id"], "tokens": len(feature["input_ids"])})
            else:
                kept.append(feature)
        return kept, excluded

    train_features, train_excluded = encode(list(iter_jsonl(args.train_jsonl)))
    val_features, val_excluded = encode(list(iter_jsonl(args.validation_jsonl)))
    if not train_features or not val_features:
        raise RuntimeError("No train/validation data remains after explicit length filter")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_record = {"model_dir": str(model_dir), "model_id": model_info["model_id_verified"],
                     "model_revision": model_info["commit_hash"], "precision": "float16",
                     "parallelism": "single_gpu", "gpu": gpu, "sequence_length": seq_len,
                     "epochs": args.epochs, "gradient_accumulation_steps": args.grad_accum,
                     "learning_rate": args.learning_rate, "lora_r": args.lora_r,
                     "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
                     "seed": args.seed, "resume_from_checkpoint": args.resume_from_checkpoint}
    (output / "training_config.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")
    exclusions = {"sequence_length": seq_len, "train_excluded": train_excluded, "validation_excluded": val_excluded}
    (output / "length_exclusions.json").write_text(json.dumps(exclusions, ensure_ascii=False, indent=2), encoding="utf-8")

    class FirstStepGradientCheck(TrainerCallback):
        checked = False
        report = {}

        def on_pre_optimizer_step(self, training_args, state, control, model=None, **kwargs):
            if self.checked or model is None:
                return control
            grads = [p.grad.detach() for p in model.parameters() if p.requires_grad and p.grad is not None]
            finite = all(bool(torch.isfinite(g).all()) for g in grads)
            nonzero = any(bool(torch.count_nonzero(g)) for g in grads)
            self.report = {"gradient_tensors": len(grads), "all_finite": finite, "any_nonzero": nonzero}
            self.checked = True
            if not grads or not finite or not nonzero:
                raise RuntimeError(f"FP16 backward smoke check failed: {self.report}")
            return control

    grad_check = FirstStepGradientCheck()
    training_args = TrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.learning_rate,
        warmup_ratio=0.03, weight_decay=0.01, logging_steps=10, save_steps=args.save_steps,
        eval_steps=args.eval_steps, save_strategy="steps", eval_strategy="steps",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0, optim="adamw_torch", fp16=True, bf16=False, tf32=False,
        report_to="none", seed=args.seed, save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, remove_unused_columns=False)
    trainer = Trainer(model=model, args=training_args, train_dataset=Dataset.from_list(train_features),
                      eval_dataset=Dataset.from_list(val_features),
                      data_collator=AssistantOnlyCollator(tokenizer), processing_class=tokenizer,
                      callbacks=[grad_check])
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_time = time.perf_counter() - started
    if not grad_check.checked:
        raise RuntimeError("Training completed without observing finite nonzero LoRA gradients")
    metrics = trainer.evaluate()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    (output / "training_log.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, default=str)+"\n" for row in trainer.state.log_history), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps({"train": result.metrics, "evaluation": metrics}, indent=2, default=str), encoding="utf-8")
    manifest = {"model": model_info, "precision": "float16", "parallelism": "single_gpu",
                "lora_targets_inspected": targets, "sequence_length": seq_len,
                "train_examples": len(train_features), "validation_examples": len(val_features),
                "first_step_gradient_check": grad_check.report, "train_metrics": result.metrics,
                "eval_metrics": metrics, "training_seconds": train_time, "hardware": hw,
                "seed": args.seed, "resumed_from": args.resume_from_checkpoint}
    (output / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest
