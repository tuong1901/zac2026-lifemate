from __future__ import annotations

import json
import platform
import time
from pathlib import Path


def choose_sequence_length(total_vram_bytes: int) -> int:
    """Conservative starting values; actual forward/backward smoke pass remains mandatory."""
    gib = total_vram_bytes / (1024**3)
    if gib >= 40:
        return 12288
    if gib >= 24:
        return 8192
    if gib >= 15:
        return 4096
    raise RuntimeError(f"QLoRA training is disabled below 15 GiB VRAM by default; detected {gib:.1f} GiB")


def smoke_forward_backward(model, tokenizer, example, output_path: Path) -> dict:
    """One reversible LoRA update verifies quantized forward/backward before long training."""
    import torch
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable adapter parameters found")
    saved = [p.detach().cpu().clone() for p in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    device = model.get_input_embeddings().weight.device
    batch = {k: torch.tensor([example[k]], device=device) for k in ("input_ids", "attention_mask", "labels")}
    model.train()
    started = time.perf_counter()
    try:
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        if not torch.isfinite(loss):
            raise RuntimeError(f"smoke loss is not finite: {value}")
    finally:
        with torch.no_grad():
            for param, original in zip(trainable, saved):
                param.copy_(original.to(param.device, dtype=param.dtype))
        optimizer.zero_grad(set_to_none=True)
        del optimizer, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    model.eval()
    report = {"passed": True, "loss": value, "seconds": time.perf_counter()-started,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def train(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, set_seed

    from lifemate.data.formatter import encode_assistant_only
    from lifemate.data.loader import iter_jsonl
    from lifemate.models.loader import discover_lora_targets, load_local_qwen
    from lifemate.training.collator import AssistantOnlyCollator
    from lifemate.utils.hardware import inspect_hardware

    if platform.system() != "Linux":
        raise RuntimeError("QLoRA training is supported only in the configured Linux/Kaggle workflow, not this local Windows host")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this QLoRA run")
    hw = inspect_hardware([args.train_jsonl, args.validation_jsonl, args.output_dir])
    gpu = hw["gpu"][0]
    if gpu["total_vram_bytes"] < 15 * 1024**3:
        raise RuntimeError(f"Detected {gpu['total_vram_bytes']/1024**3:.1f} GiB VRAM; need at least 15 GiB for this 4-bit smoke-gated setup")
    if gpu["free_vram_bytes"] < 12 * 1024**3:
        raise RuntimeError(f"Only {gpu['free_vram_bytes']/1024**3:.1f} GiB VRAM is free; clear competing GPU jobs before loading the 4B QLoRA model")
    seq_len = args.max_seq_len or choose_sequence_length(gpu["total_vram_bytes"])
    set_seed(args.seed)
    model, tokenizer, model_info = load_local_qwen(args.model_dir, quantized=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    targets = discover_lora_targets(model)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                          bias="none", task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, lora_cfg)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    train_raw = list(iter_jsonl(args.train_jsonl))
    val_raw = list(iter_jsonl(args.validation_jsonl))

    def encode(rows):
        kept, excluded = [], []
        for row in rows:
            feature = encode_assistant_only(row["messages"], tokenizer)
            if len(feature["input_ids"]) > seq_len:
                excluded.append({"id": row["id"], "source_id": row["source_id"], "tokens": len(feature["input_ids"])})
            else:
                kept.append(feature)
        return kept, excluded

    train_features, train_excluded = encode(train_raw)
    val_features, val_excluded = encode(val_raw)
    if not train_features or not val_features:
        raise RuntimeError("No train/validation data remains after explicit length filter")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_record = {"model_dir": str(Path(args.model_dir).resolve()), "sequence_length": seq_len,
                     "epochs": args.epochs, "gradient_accumulation_steps": args.grad_accum,
                     "learning_rate": args.learning_rate, "lora_r": args.lora_r,
                     "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
                     "seed": args.seed, "resume_from_checkpoint": args.resume_from_checkpoint}
    (output / "training_config.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")
    exclusions = {"sequence_length": seq_len, "train_excluded": train_excluded, "validation_excluded": val_excluded}
    (output / "length_exclusions.json").write_text(json.dumps(exclusions, ensure_ascii=False, indent=2), encoding="utf-8")
    smoke_example = max(train_features, key=lambda x: len(x["input_ids"]))
    smoke = smoke_forward_backward(model, tokenizer, smoke_example, output / "smoke_train_step.json")
    smoke["training_time_estimate_seconds"] = smoke["seconds"] * len(train_features) * args.epochs
    (output / "smoke_train_step.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    train_ds, val_ds = Dataset.from_list(train_features), Dataset.from_list(val_features)
    training_args = TrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.learning_rate,
        warmup_ratio=0.03, weight_decay=0.01, logging_steps=10, save_steps=args.save_steps,
        eval_steps=args.eval_steps, save_strategy="steps", eval_strategy="steps",
        gradient_checkpointing=True, max_grad_norm=0.3, optim="paged_adamw_8bit",
        fp16=not gpu["bf16"], bf16=gpu["bf16"], report_to="none", seed=args.seed,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, remove_unused_columns=False)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=AssistantOnlyCollator(tokenizer), processing_class=tokenizer)
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_time = time.perf_counter() - started
    metrics = trainer.evaluate()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    (output / "training_log.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, default=str)+"\n" for row in trainer.state.log_history), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps({"train": result.metrics, "evaluation": metrics}, indent=2, default=str), encoding="utf-8")
    manifest = {"model": model_info, "model_id": "Qwen/Qwen3-4B-Instruct-2507", "lora_targets_inspected": targets,
                "sequence_length": seq_len, "train_examples": len(train_features), "validation_examples": len(val_features),
                "smoke_step": smoke, "train_metrics": result.metrics, "eval_metrics": metrics,
                "training_seconds": train_time, "hardware": hw, "seed": args.seed,
                "resumed_from": args.resume_from_checkpoint}
    (output / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return manifest
