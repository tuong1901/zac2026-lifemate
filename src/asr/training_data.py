"""Audio/text manifests and leakage-safe splitting; no runtime or GPU imports."""
import json
import math
import random
from pathlib import Path


def speed_output_length(length, factor, max_samples=480000):
    """Skip a slowdown if it would exceed Whisper's 30s window; never crop speech."""
    if length <= 0 or not math.isfinite(factor) or factor <= 0:
        raise ValueError("Require nonempty audio and a finite positive speed factor")
    proposed = max(1, math.ceil(length / factor))
    return length if proposed > max_samples else proposed


def cast_acoustic_features(batch, dtype):
    """Generation runs outside training autocast; match the frozen encoder dtype.

    Token labels and attention masks must keep their integer dtype.
    """
    batch["input_features"] = batch["input_features"].to(dtype=dtype)
    return batch


def select_saved_checkpoint(run_dir):
    """Select the lowest logged validation WER among complete saved adapters."""
    root = Path(run_dir)
    candidates = []
    for checkpoint in root.glob("checkpoint-*"):
        if not (checkpoint / "adapter_config.json").is_file():
            continue
        if not any((checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0
                   for name in ("adapter_model.safetensors", "adapter_model.bin")):
            continue
        state_file = checkpoint / "trainer_state.json"
        if not state_file.is_file():
            continue
        state = json.loads(state_file.read_text(encoding="utf-8"))
        step = int(checkpoint.name.split("-")[-1])
        for row in state.get("log_history", []):
            if row.get("step") == step and isinstance(row.get("eval_wer"), (float, int)):
                candidates.append((float(row["eval_wer"]), -step, checkpoint))
    if not candidates:
        raise ValueError("No complete checkpoint with saved validation WER found; inspect run directory")
    wer, negative_step, checkpoint = min(candidates)
    return checkpoint, {"checkpoint": str(checkpoint), "step": -negative_step, "eval_wer": wer,
                        "selection": "lowest saved validation WER among available adapter checkpoints"}


def collect_clips(train_json, audio_root=None):
    path = Path(train_json).resolve()
    root = Path(audio_root).resolve() if audio_root else path.parent
    clips = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            turns = [item] + [turn for turn in item.get("history", []) if turn.get("role") == "user"]
            for turn in turns:
                audio, text = turn.get("audio"), turn.get("audio_text")
                if not audio or not isinstance(text, str) or not text.strip():
                    continue
                audio_path = (root / audio).resolve()
                if not audio_path.is_relative_to(root):
                    raise ValueError(f"Audio path escapes audio root: {audio}")
                if not audio_path.is_file():
                    raise FileNotFoundError(audio_path)
                clips.append({"sample_id": str(item["id"]), "audio": str(audio_path), "text": text})
    if not clips:
        raise ValueError("No audio/audio_text pairs found")
    return clips


def split_clips(clips, val_ratio=0.1, seed=42):
    """Keep connected conversations together when they share an audio path."""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be strictly between 0 and 1")
    parent = {clip["sample_id"]: clip["sample_id"] for clip in clips}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    owners = {}
    for clip in clips:
        owner = owners.setdefault(clip["audio"], clip["sample_id"])
        parent[find(clip["sample_id"])] = find(owner)
    groups = {}
    for clip in clips:
        groups.setdefault(find(clip["sample_id"]), []).append(clip)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("Need at least two independent conversation groups for validation")
    random.Random(seed).shuffle(keys)
    target, held, count = max(1, round(len(clips) * val_ratio)), set(), 0
    for key in keys[:-1]:
        held.add(key)
        count += len(groups[key])
        if count >= target:
            break
    train, val = [], []
    for key, group in groups.items():
        (val if key in held else train).extend(group)

    def dedup(rows):
        seen, out = {}, []
        for row in rows:
            if row["audio"] in seen:
                if seen[row["audio"]] != row["text"]:
                    raise ValueError(f"Conflicting transcripts for {row['audio']}")
                continue
            seen[row["audio"]] = row["text"]
            out.append(row)
        return out

    return dedup(train), dedup(val)
