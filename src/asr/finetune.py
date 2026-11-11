"""PhoWhisper LoRA training/merge, integrated with src.asr inference.

python -m src.asr.finetune train --train-json /path/train/train.jsonl --out /path/out
python -m src.asr.finetune merge --adapter /path/out/best_adapter --out /path/model
"""
import argparse
import json
import os
from pathlib import Path

from src.asr.training_data import collect_clips, split_clips


def write_json(path, value):
    if int(os.environ.get("RANK", "0")) != 0:
        return
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def train(args):
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from math import gcd
    from peft import LoraConfig, get_peft_model
    from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)
    from src.asr.evaluate import compute_levenshtein_distance, normalize_text_for_eval

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU before training")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and list(out.glob("checkpoint-*")):
        raise ValueError("Output already has checkpoints; use --resume or a new --out")
    processor = WhisperProcessor.from_pretrained(args.model, language="vi", task="transcribe")
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
    print(f"Train={len(train_rows)}; validation={len(val_rows)}; skipped={len(skipped)}", flush=True)

    class AudioDataset(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            wave, sr = sf.read(row["audio"], dtype="float32", always_2d=True)
            wave = wave.mean(axis=1)
            if sr != 16000:
                common = gcd(sr, 16000)
                wave = resample_poly(wave, 16000 // common, sr // common).astype(np.float32)
            return {"input_features": processor.feature_extractor(wave, sampling_rate=16000).input_features[0],
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
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"], bias="none"))
    # Generic PEFT exposes **kwargs, but Whisper does not accept Trainer's
    # num_items_in_batch loss argument.
    model.accepts_loss_kwargs = False
    model.print_trainable_parameters()
    start_token = model.config.decoder_start_token_id

    def collate(features):
        batch = processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        target = processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        labels = target.input_ids.masked_fill(target.attention_mask.ne(1), -100)
        if (labels[:, 0] == start_token).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

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
        load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
        predict_with_generate=True, generation_max_length=448, generation_num_beams=1,
        remove_unused_columns=False, label_names=["labels"], report_to=[],
        ddp_find_unused_parameters=False,
        seed=args.seed, data_seed=args.seed, logging_steps=10,
        dataloader_num_workers=0, eval_accumulation_steps=1)
    trainer = Seq2SeqTrainer(
        model=model, args=options, train_dataset=AudioDataset(train_rows),
        eval_dataset=AudioDataset(val_rows), data_collator=collate,
        processing_class=processor, compute_metrics=metrics)
    # Fresh zero-initialized LoRA is the base model; compare on the same holdout.
    if not args.resume:
        write_json(out / "baseline_metrics.json", trainer.evaluate(metric_key_prefix="baseline"))
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(out / "best_adapter"))
    if trainer.is_world_process_zero():
        processor.save_pretrained(out / "best_adapter")
    write_json(out / "validation_metrics.json", trainer.evaluate())
    write_json(out / "training_summary.json", {
        "base_model": args.model, "best_checkpoint": trainer.state.best_model_checkpoint,
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
    WhisperProcessor.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"Merged model ready for src.asr: {args.out}")


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
    fit.add_argument("--val-ratio", type=float, default=0.1)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--resume", help="Path to checkpoint-N including optimizer/RNG state")
    fit.set_defaults(handler=train)
    export = sub.add_parser("merge")
    export.add_argument("--adapter", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--model", help="Optional local base model path; defaults to adapter's base")
    export.set_defaults(handler=merge)
    args = parser.parse_args()
    if args.command == "train":
        if min(args.epochs, args.batch_size, args.grad_accum, args.rank, args.lr) <= 0:
            parser.error("Training counts, rank and learning rate must be positive")
        if not 0 <= args.specaug < 1 or not 0 < args.val_ratio < 1:
            parser.error("Require 0<=specaug<1 and 0<val-ratio<1")
    args.handler(args)


if __name__ == "__main__":
    main()
