"""PhoWhisper LoRA training/merge, integrated with src.asr inference.

python -m src.asr.finetune train --train-json /path/train/train.jsonl --out /path/out
python -m src.asr.finetune merge --adapter /path/out/best_adapter --out /path/model
"""
import argparse
import json
import os
from pathlib import Path

# This entry point uses PyTorch only. Set these before PEFT/Transformers imports.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["USE_FLAX"] = "0"

from src.asr.training_data import (collect_clips, split_clips, cast_acoustic_features,
                                   select_saved_checkpoint, speed_output_length)


def write_json(path, value):
    if int(os.environ.get("RANK", "0")) != 0:
        return
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def train(args):
    import numpy as np
    import soundfile as sf
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"PyTorch cannot access CUDA: torch={torch.__version__}, "
                           f"built_cuda={torch.version.cuda}, visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    print(f"[ASR startup] rank={os.environ.get('RANK', '0')} local_rank={local_rank} "
          f"GPU={torch.cuda.get_device_name(local_rank)} torch={torch.__version__} "
          f"CUDA={torch.version.cuda}", flush=True)
    from scipy.signal import resample_poly
    from math import gcd
    from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
    from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)
    from src.asr.evaluate import compute_levenshtein_distance, normalize_text_for_eval

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and list(out.glob("checkpoint-*")):
        raise ValueError("Output already has checkpoints; use --resume or a new --out")
    processor = WhisperProcessor.from_pretrained(args.model, language="vi", task="transcribe")
    if args.split_manifest:
        fixed = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
        train_rows, val_rows = fixed["train"], fixed["validation"]
        if {row["audio"] for row in train_rows} & {row["audio"] for row in val_rows}:
            raise ValueError("Split manifest has overlapping train/validation audio")
        for row in train_rows + val_rows:
            if not Path(row["audio"]).is_file():
                raise FileNotFoundError(row["audio"])
    else:
        train_rows, val_rows = split_clips(collect_clips(args.train_json, args.audio_root), args.val_ratio, args.seed)
    skipped = []

    def prepare(rows):
        kept = []
        for row in rows:
            info = sf.info(row["audio"])
            labels = processor.tokenizer(row["text"]).input_ids
            # Never silently truncate audio or target text.
            if not 0 < info.duration <= 30 or len(labels) > 448:
                skipped.append({**row, "reason": "audio outside (0,30] seconds or target >448 tokens"})
                continue
            kept.append(row)
        return kept

    train_rows, val_rows = prepare(train_rows), prepare(val_rows)
    if not train_rows or not val_rows:
        raise ValueError("No usable clips in train or validation after length checks")
    manifest = {"model": args.model, "seed": args.seed, "val_ratio": args.val_ratio,
                "train": train_rows, "validation": val_rows, "skipped": skipped}
    if args.resume and (out / "split_manifest.json").exists():
        old = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
        if old != manifest:
            raise ValueError("Resume manifest differs; keep the same model, data paths, split and seed")
    write_json(out / "split_manifest.json", manifest)
    config = {key: value for key, value in vars(args).items() if key != "handler"}
    config.pop("resume", None)
    config_file = out / "run_config.json"
    if args.resume and config_file.exists():
        if json.loads(config_file.read_text(encoding="utf-8")) != config:
            raise ValueError("Resume configuration changed; use a new run directory instead")
    write_json(config_file, config)
    print(f"Train={len(train_rows)}; validation={len(val_rows)}; skipped={len(skipped)}", flush=True)

    class AudioDataset(torch.utils.data.Dataset):
        def __init__(self, rows, augment=False):
            self.rows = rows
            self.augment = augment

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            wave, sr = sf.read(row["audio"], dtype="float32", always_2d=True)
            wave = wave.mean(axis=1)
            if sr != 16000:
                common = gcd(sr, 16000)
                wave = resample_poly(wave, 16000 // common, sr // common).astype(np.float32)
            if self.augment and args.speed:
                factor = np.random.uniform(1 - args.speed, 1 + args.speed)
                length = speed_output_length(len(wave), factor)
                if length != len(wave):
                    wave = np.interp(np.linspace(0, len(wave) - 1, length),
                                     np.arange(len(wave)), wave).astype(np.float32)
            acoustic = processor.feature_extractor(wave, sampling_rate=16000, return_attention_mask=True)
            return {"input_features": acoustic.input_features[0],
                    "attention_mask": acoustic.attention_mask[0],
                    "labels": processor.tokenizer(row["text"]).input_ids}

    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16)
    model.config.use_cache = False
    model.config.apply_spec_augment = args.specaug > 0
    model.config.mask_time_prob = args.specaug
    model.config.mask_time_length = 10
    model.config.mask_feature_prob = args.specaug
    model.config.mask_feature_length = 10
    model.generation_config.language = "vi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.do_sample = False
    model.generation_config.condition_on_prev_tokens = False
    model.generation_config.no_repeat_ngram_size = 3
    # Generic PeftModel preserves Whisper's input_features interface (not text input_ids).
    if args.init_adapter:
        adapter_config = PeftConfig.from_pretrained(args.init_adapter)
        if adapter_config.base_model_name_or_path.rstrip('/') != args.model.rstrip('/'):
            raise ValueError("Initial adapter base differs from --model; use its original base")
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"], bias="none"))
    # Generic PEFT exposes **kwargs, but Whisper does not accept Trainer's
    # num_items_in_batch loss argument.
    model.accepts_loss_kwargs = False
    model.print_trainable_parameters()
    start_token = model.config.decoder_start_token_id
    acoustic_dtype = model.get_base_model().model.encoder.conv1.weight.dtype

    def collate(features):
        batch = processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        target = processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        labels = target.input_ids.masked_fill(target.attention_mask.ne(1), -100)
        if (labels[:, 0] == start_token).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        batch["attention_mask"] = torch.as_tensor(
            np.stack([f["attention_mask"] for f in features]), dtype=torch.long)
        return cast_acoustic_features(batch, acoustic_dtype)

    def metrics(prediction):
        ids = prediction.predictions
        if isinstance(ids, tuple):
            ids = ids[0]
        refs = processor.tokenizer.batch_decode(
            np.where(prediction.label_ids == -100, processor.tokenizer.pad_token_id, prediction.label_ids),
            skip_special_tokens=True)
        hyps = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
        raw_errors = raw_words = errors = words = 0
        for ref, hyp in zip(refs, hyps):
            raw_errors += compute_levenshtein_distance(ref.split(), hyp.split())[0]
            raw_words += len(ref.split())
            ref, hyp = normalize_text_for_eval(ref), normalize_text_for_eval(hyp)
            errors += compute_levenshtein_distance(ref.split(), hyp.split())[0]
            words += len(ref.split())
        return {"wer": errors / max(1, words), "raw_wer": raw_errors / max(1, raw_words)}

    options = Seq2SeqTrainingArguments(
        output_dir=str(out), per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, num_train_epochs=args.epochs, warmup_ratio=0.05,
        lr_scheduler_type="cosine", fp16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        # Select/load only after every rank finishes training. Checkpoint path
        # existence can differ between ranks while rank 0 is still saving it.
        load_best_model_at_end=False, metric_for_best_model="wer", greater_is_better=False,
        predict_with_generate=True, generation_max_length=448, generation_num_beams=1,
        remove_unused_columns=False, label_names=["labels"], report_to=[],
        ddp_find_unused_parameters=False,
        seed=args.seed, data_seed=args.seed, logging_steps=10,
        dataloader_num_workers=0, eval_accumulation_steps=1)
    trainer = Seq2SeqTrainer(
        model=model, args=options, train_dataset=AudioDataset(train_rows, augment=True),
        eval_dataset=AudioDataset(val_rows), data_collator=collate,
        processing_class=processor, compute_metrics=metrics)
    # Baseline is the original base, or the initial adapter when warm-starting.
    if not args.resume and not args.skip_baseline:
        write_json(out / "baseline_metrics.json", trainer.evaluate(metric_key_prefix="baseline"))
    trainer.train(resume_from_checkpoint=args.resume)
    import torch.distributed as dist
    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        dist.barrier()
    selected = [None]
    if trainer.is_world_process_zero():
        try:
            checkpoint, details = select_saved_checkpoint(out)
            selected[0] = {"path": str(checkpoint), "wer": details["eval_wer"]}
            write_json(out / "selected_checkpoint.json", details)
        except Exception as error:
            selected[0] = {"error": str(error)}
    if distributed:
        dist.broadcast_object_list(selected, src=0)
    if "error" in selected[0]:
        raise RuntimeError(f"Checkpoint selection failed: {selected[0]['error']}")
    trainer.state.best_model_checkpoint = selected[0]["path"]
    trainer.state.best_metric = selected[0]["wer"]
    trainer._load_best_model()
    trainer.save_model(str(out / "best_adapter"))
    if trainer.is_world_process_zero():
        processor.save_pretrained(out / "best_adapter")
    write_json(out / "validation_metrics.json", trainer.evaluate())
    write_json(out / "training_summary.json", {
        "base_model": args.model, "best_checkpoint": trainer.state.best_model_checkpoint,
        "initial_adapter": args.init_adapter, "optimizer_restarted": bool(args.init_adapter and not args.resume),
        "best_wer": trainer.state.best_metric, "train_clips": len(train_rows),
        "validation_clips": len(val_rows), "skipped": len(skipped),
        "selection": "minimum normalized validation WER among trained checkpoints",
        "note": "WER is a ratio; 0.1 means 10%. Large is an experimental model per user request."})


def merge(args):
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    config = PeftConfig.from_pretrained(args.adapter)
    base = args.model or config.base_model_name_or_path
    model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.config.use_cache = True
    model.config.apply_spec_augment = False
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = "vi"
    model.generation_config.task = "transcribe"
    model.generation_config.do_sample = False
    model.generation_config.condition_on_prev_tokens = False
    model.save_pretrained(args.out, safe_serialization=True)
    processor_source = args.adapter if (Path(args.adapter) / "preprocessor_config.json").is_file() else base
    WhisperProcessor.from_pretrained(processor_source).save_pretrained(args.out)
    print(f"Merged model ready for src.asr: {args.out}")


def recover(args):
    checkpoint, details = select_saved_checkpoint(args.run_dir)
    print(f"Recovering step={details['step']} saved WER={details['eval_wer']:.4%} from {checkpoint}", flush=True)
    args.adapter = str(checkpoint)
    merge(args)
    write_json(Path(args.out) / "recovery_summary.json", details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--train-json", required=True)
    fit.add_argument("--audio-root", help="Parent of relative audio paths; defaults to train.jsonl directory")
    fit.add_argument("--model", default="vinai/PhoWhisper-large")
    fit.add_argument("--out", required=True)
    fit.add_argument("--epochs", type=int, default=3)
    fit.add_argument("--batch-size", type=int, default=1)
    fit.add_argument("--grad-accum", type=int, default=16)
    fit.add_argument("--lr", type=float, default=1e-4)
    fit.add_argument("--rank", type=int, default=32)
    fit.add_argument("--specaug", type=float, default=0.05)
    fit.add_argument("--speed", type=float, default=0.0, help="Train-only speed variation, e.g. 0.1 = +/-10%%")
    fit.add_argument("--split-manifest", help="Reuse the exact train/validation clips of an earlier run")
    fit.add_argument("--skip-baseline", action="store_true", help="Skip initial base WER; never skips epoch validation")
    fit.add_argument("--val-ratio", type=float, default=0.1)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--resume", help="Path to checkpoint-N including optimizer/RNG state")
    fit.add_argument("--init-adapter", help="Warm-start LoRA weights with a new optimizer/schedule; original base required")
    fit.set_defaults(handler=train)
    export = sub.add_parser("merge")
    export.add_argument("--adapter", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--model", help="Optional local base model path; defaults to adapter's base")
    export.set_defaults(handler=merge)
    recovery = sub.add_parser("recover", help="Merge best saved adapter after an interrupted run; no training")
    recovery.add_argument("--run-dir", required=True)
    recovery.add_argument("--out", required=True)
    recovery.add_argument("--model", help="Optional local base model path")
    recovery.set_defaults(handler=recover)
    args = parser.parse_args()
    if args.command == "train":
        if min(args.epochs, args.batch_size, args.grad_accum, args.rank, args.lr) <= 0:
            parser.error("Training counts, rank and learning rate must be positive")
        if not 0 <= args.specaug < 1 or not 0 < args.val_ratio < 1:
            parser.error("Require 0<=specaug<1 and 0<val-ratio<1")
        if not 0 <= args.speed < 1:
            parser.error("Require 0<=speed<1")
    args.handler(args)


if __name__ == "__main__":
    main()
