"""Compare decoding on a fixed validation manifest; shard audio across two GPUs.

torchrun --standalone --nproc_per_node=2 -m src.asr.benchmark \
  --manifest /path/asr_training/split_manifest.json --model /path/merged --out /path/decoding
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["USE_FLAX"] = "0"

from src.asr.evaluate import compute_levenshtein_distance, normalize_text_for_eval


def score_predictions(rows):
    errors = words = raw_errors = raw_words = 0
    for row in rows:
        ref, hyp = row["reference"], row["prediction"]
        raw_errors += compute_levenshtein_distance(ref.split(), hyp.split())[0]
        raw_words += len(ref.split())
        ref, hyp = normalize_text_for_eval(ref), normalize_text_for_eval(hyp)
        errors += compute_levenshtein_distance(ref.split(), hyp.split())[0]
        words += len(ref.split())
    return {"wer": errors / max(1, words), "raw_wer": raw_errors / max(1, raw_words),
            "errors": errors, "reference_words": words, "clips": len(rows)}


def decoding_configs(value):
    configs = []
    for entry in value.split(","):
        beam, ngram = map(int, entry.split(":"))
        if beam < 1 or ngram < 0:
            raise ValueError("Require beam>=1 and ngram>=0")
        if (beam, ngram) not in configs:
            configs.append((beam, ngram))
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True, help="Merged model directory or Hugging Face base model")
    parser.add_argument("--out", required=True, help="New, empty results directory")
    parser.add_argument("--configs", default="1:3,1:0,5:0", help="beam:no_repeat_ngram pairs")
    args = parser.parse_args()
    configs = decoding_configs(args.configs)
    import numpy as np
    import torch
    import torch.distributed as dist
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device)
    out = Path(args.out)
    # All ranks inspect the same directory before any rank writes predictions.
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a new --out to preserve previous benchmark results")
    if world > 1:
        dist.init_process_group("nccl")
        dist.barrier()
    out.mkdir(parents=True, exist_ok=True)
    try:
        manifest_bytes = Path(args.manifest).read_bytes()
        rows = json.loads(manifest_bytes)["validation"]
        if not rows or len({row['audio'] for row in rows}) != len(rows):
            raise ValueError("Validation must contain nonempty, unique audio paths")
        processor = WhisperProcessor.from_pretrained(args.model, language="vi", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
        model.config.apply_spec_augment = False
        model.generation_config.forced_decoder_ids = None
        model.generation_config.use_cache = True
        for beam, ngram in configs:
            name = f"beam{beam}_ngram{ngram}"
            shard = out / f"{name}.rank{rank}.jsonl"
            with shard.open("w", encoding="utf-8") as stream:
                for index in range(rank, len(rows), world):
                    row = rows[index]
                    wave, sr = sf.read(row["audio"], dtype="float32", always_2d=True)
                    wave = wave.mean(axis=1)
                    if not 0 < len(wave) / sr <= 30:
                        raise ValueError(f"Clip outside (0,30] seconds: {row['audio']}")
                    if sr != 16000:
                        common = gcd(sr, 16000)
                        wave = resample_poly(wave, 16000 // common, sr // common).astype(np.float32)
                    features = processor.feature_extractor(
                        wave, sampling_rate=16000, return_attention_mask=True, return_tensors="pt")
                    with torch.inference_mode():
                        tokens = model.generate(
                            input_features=features.input_features.to(device=device, dtype=torch.float16),
                            attention_mask=features.attention_mask.to(device),
                            language="vi", task="transcribe", do_sample=False,
                            num_beams=beam, no_repeat_ngram_size=ngram,
                            condition_on_prev_tokens=False, max_length=448, synced_gpus=False)
                    prediction = processor.batch_decode(tokens, skip_special_tokens=True)[0]
                    record = {"index": index, "sample_id": row["sample_id"], "audio": row["audio"],
                              "reference": row["text"], "prediction": prediction}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    if (index // world + 1) % 25 == 0:
                        print(f"[{name}] rank={rank} completed {index // world + 1} clips", flush=True)
            if world > 1:
                dist.barrier()
            if rank == 0:
                print(f"Completed {name}", flush=True)
        if rank == 0:
            scores = {}
            for beam, ngram in configs:
                name = f"beam{beam}_ngram{ngram}"
                predictions = []
                for worker in range(world):
                    with (out / f"{name}.rank{worker}.jsonl").open(encoding="utf-8") as stream:
                        predictions.extend(json.loads(line) for line in stream if line.strip())
                predictions.sort(key=lambda row: row["index"])
                if [row["index"] for row in predictions] != list(range(len(rows))):
                    raise ValueError("Missing or duplicate prediction indices")
                scores[name] = score_predictions(predictions)
                with (out / f"{name}.predictions.jsonl").open("w", encoding="utf-8") as stream:
                    for prediction in predictions:
                        stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            report = {"model": args.model, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                      "scores": scores, "best_config": min(scores, key=lambda name: scores[name]["wer"]),
                      "note": "Decoding selected on validation; not an independent test-set estimate."}
            (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if world > 1:
            dist.barrier()
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
