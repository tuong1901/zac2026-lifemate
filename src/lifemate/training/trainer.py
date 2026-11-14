from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path


def choose_sequence_length(total_vram_bytes: int) -> int:
    """Choose a sequence cap from per-GPU VRAM; FSDP does not shard activations."""
    gib = total_vram_bytes / (1024**3)
    if gib >= 40:
        return 16384
    if gib >= 24:
        return 12288
    if gib >= 14:
        return 8192
    raise RuntimeError(f"FP16 Qwen training requires at least 14 GiB per GPU; detected {gib:.1f} GiB")


def check_fsdp_hardware(hw: dict) -> list[dict]:
    if platform.system() != "Linux" or not hw.get("cuda_available"):
        raise RuntimeError("FP16 FSDP LoRA training requires Linux and CUDA. No model was downloaded.")
    gpus = hw.get("gpu", [])
    if len(gpus) < 2:
        raise RuntimeError(f"FP16 FSDP training requires at least two CUDA GPUs; detected {len(gpus)}. No model was downloaded.")
    selected = gpus[:2]
    for gpu in selected:
        if "T4" not in gpu["name"].upper():
            raise RuntimeError(f"This Kaggle workflow requires two Tesla T4 GPUs; detected {gpu['name']}. No model was downloaded.")
        total_gib = gpu["total_vram_bytes"] / 1024**3
        free_gib = gpu["free_vram_bytes"] / 1024**3
        if total_gib < 14:
            raise RuntimeError(f"FP16 FSDP requires at least 14 GiB per GPU; GPU {gpu['index']} has {total_gib:.1f} GiB. No model was downloaded.")
        if free_gib < 10:
            raise RuntimeError(f"FP16 FSDP requires at least 10 GiB free per GPU before model download; GPU {gpu['index']} has {free_gib:.1f} GiB free. No model was downloaded.")
    return selected


def train(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    from lifemate.data.formatter import encode_assistant_only
    from lifemate.data.loader import iter_jsonl
    from lifemate.models.loader import discover_lora_targets, inspect_local_model
    from lifemate.training.collator import AssistantOnlyCollator
    from lifemate.training.chunked_loss import install_qwen3_chunked_loss, latest_loss_diagnostic
    from lifemate.utils.hardware import inspect_hardware
    from lifemate.utils.runtime_compat import fp16_dtype_argument

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        raise RuntimeError("Launch this FP16 FSDP training with torchrun on two GPUs; refusing single-GPU execution.")
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    hw = inspect_hardware([args.train_jsonl, args.validation_jsonl, args.output_dir])
    gpus = check_fsdp_hardware(hw)
    if world_size != len(gpus):
        raise RuntimeError(f"torchrun world size {world_size} does not match selected GPU count {len(gpus)}")
    seq_len = args.max_seq_len or choose_sequence_length(min(gpu["total_vram_bytes"] for gpu in gpus))
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
    load_kwargs = {"local_files_only": True, "trust_remote_code": False,
                   "low_cpu_mem_usage": True, "attn_implementation": "sdpa"}
    load_kwargs.update(fp16_dtype_argument(AutoModelForCausalLM, torch.float16))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
    if next(model.parameters()).dtype != torch.float16:
        raise RuntimeError("Base model did not load in FP16; refusing precision fallback")
    targets = discover_lora_targets(model)
    transformer_layers = model.model.layers
    transformer_layer_cls = type(transformer_layers[0]).__name__
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=targets))
    install_qwen3_chunked_loss(model, chunk_size=64)

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
                     "parallelism": "pytorch_fsdp_full_shard", "gpus": gpus, "sequence_length": seq_len,
                     "transformer_layer_cls_to_wrap": transformer_layer_cls,
                     "loss": "chunked_causal_lm", "loss_chunk_size": 64,
                     "fsdp_activation_checkpointing": True,
                     "epochs": args.epochs, "gradient_accumulation_steps": args.grad_accum,
                     "learning_rate": args.learning_rate, "lora_r": args.lora_r,
                     "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
                     "seed": args.seed, "resume_from_checkpoint": args.resume_from_checkpoint}
    if local_rank == 0:
        (output / "training_config.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")
        exclusions = {"sequence_length": seq_len, "train_excluded": train_excluded, "validation_excluded": val_excluded}
        (output / "length_exclusions.json").write_text(json.dumps(exclusions, ensure_ascii=False, indent=2), encoding="utf-8")

    class FirstStepGradientCheck(TrainerCallback):
        checked = False
        report = {}

        def on_step_begin(self, training_args, state, control, **kwargs):
            if not self.checked:
                torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
            return control

        def on_pre_optimizer_step(self, training_args, state, control, model=None, **kwargs):
            if self.checked or model is None:
                return control
            grads = [p.grad.detach() for p in model.parameters() if p.requires_grad and p.grad is not None]
            finite = all(bool(torch.isfinite(g).all()) for g in grads)
            nonzero = any(bool(torch.count_nonzero(g)) for g in grads)
            device = torch.cuda.current_device()
            torch.cuda.synchronize(device)
            self.report = {"gradient_tensors": len(grads), "all_finite": finite, "any_nonzero": nonzero,
                           "rank": int(os.environ.get("RANK", "0")),
                           "peak_allocated_gib_forward_backward": round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
                           "peak_reserved_gib_forward_backward": round(torch.cuda.max_memory_reserved(device) / 1024**3, 3)}
            print("LIFEMATE_FIRST_STEP_SMOKE " + json.dumps(self.report), flush=True)
            self.checked = True
            if not grads or not finite or not nonzero:
                raise RuntimeError(f"FP16 backward smoke check failed: {self.report}")
            return control

    grad_check = FirstStepGradientCheck()
    smoke_only = bool(getattr(args, "smoke_only", False))
    training_args = TrainingArguments(
        output_dir=str(output / ("smoke_checkpoints" if smoke_only else "checkpoints")),
        num_train_epochs=1 if smoke_only else args.epochs,
        max_steps=1 if smoke_only else -1,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if smoke_only else args.grad_accum, learning_rate=args.learning_rate,
        warmup_ratio=0.03, weight_decay=0.01, logging_steps=10, save_steps=args.save_steps,
        eval_steps=args.eval_steps, save_strategy="no" if smoke_only else "steps",
        eval_strategy="no" if smoke_only else "steps",
        max_grad_norm=1.0, optim="adamw_torch", fp16=True, bf16=False, tf32=False,
        fsdp="full_shard auto_wrap", fsdp_config={
            "transformer_layer_cls_to_wrap": transformer_layer_cls,
            "use_orig_params": True, "sync_module_states": True,
            "cpu_ram_efficient_loading": True, "limit_all_gathers": True,
            "activation_checkpointing": True,
            "state_dict_type": "FULL_STATE_DICT",
        },
        report_to="none", seed=args.seed, save_total_limit=2, load_best_model_at_end=not smoke_only,
        metric_for_best_model="eval_loss", greater_is_better=False, remove_unused_columns=False,
        prediction_loss_only=True)
    trainer = Trainer(model=model, args=training_args, train_dataset=Dataset.from_list(train_features),
                      eval_dataset=None if smoke_only else Dataset.from_list(val_features),
                      data_collator=AssistantOnlyCollator(tokenizer), processing_class=tokenizer,
                      callbacks=[grad_check])
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_time = time.perf_counter() - started
    if not grad_check.checked:
        raise RuntimeError("Training completed without observing finite nonzero LoRA gradients")
    if smoke_only:
        local_smoke = {"gradient_check": grad_check.report,
                       "loss_diagnostic": latest_loss_diagnostic()}
        rank_smokes = [None] * world_size
        torch.distributed.all_gather_object(rank_smokes, local_smoke)
        smoke_result = {"passed": True, "optimizer_steps": int(trainer.state.global_step),
                        "loss": result.metrics.get("train_loss"), "per_rank": rank_smokes,
                        "loss_correctness": "chunked loss matches shifted, ignored-label mean CE in unit tests; runtime reports finite loss",
                        "training_seconds": train_time,
                        "precision": "float16", "parallelism": "pytorch_fsdp_full_shard"}
        if local_rank == 0:
            (output / "fsdp_smoke_result.json").write_text(json.dumps(smoke_result, indent=2), encoding="utf-8")
        print("LIFEMATE_FSDP_SMOKE_RESULT " + json.dumps(smoke_result), flush=True)
        return smoke_result
    metrics = trainer.evaluate()
    trainer.save_model(str(output / "adapter"))
    if local_rank == 0:
        tokenizer.save_pretrained(output / "adapter")
        (output / "training_log.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, default=str)+"\n" for row in trainer.state.log_history), encoding="utf-8")
        (output / "metrics.json").write_text(json.dumps({"train": result.metrics, "evaluation": metrics}, indent=2, default=str), encoding="utf-8")
        manifest = {"model": model_info, "precision": "float16", "parallelism": "pytorch_fsdp_full_shard",
                    "gpus": gpus, "lora_targets_inspected": targets, "sequence_length": seq_len,
                    "transformer_layer_cls_to_wrap": transformer_layer_cls,
                    "loss": "chunked_causal_lm", "loss_chunk_size": 64,
                    "fsdp_activation_checkpointing": True,
                    "train_examples": len(train_features), "validation_examples": len(val_features),
                    "first_step_gradient_check": grad_check.report, "train_metrics": result.metrics,
                    "eval_metrics": metrics, "training_seconds": train_time, "hardware": hw,
                    "seed": args.seed, "resumed_from": args.resume_from_checkpoint}
        (output / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"parallelism": "pytorch_fsdp_full_shard", "precision": "float16"}
