from __future__ import annotations

import json
import re
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
PROJECTION_SUFFIXES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def download_qwen_model(model_dir: str | Path) -> dict:
    """Download an immutable Hugging Face snapshot and record its resolved revision."""
    from huggingface_hub import HfApi, snapshot_download

    target = Path(model_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    revision = HfApi().model_info(MODEL_ID, revision="main").sha
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision or ""):
        raise RuntimeError(f"Hugging Face returned an invalid immutable revision for {MODEL_ID}: {revision!r}")
    snapshot_download(repo_id=MODEL_ID, revision=revision, local_dir=str(target))
    manifest = {"model_id": MODEL_ID, "revision": revision, "source": "huggingface.co", "load_precision": "float16"}
    (target / "model_source.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return inspect_local_model(target)


def inspect_local_model(model_dir: str | Path) -> dict:
    path = Path(model_dir)
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"local model config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer_config = path / "tokenizer_config.json"
    generation_config = path / "generation_config.json"
    source_manifest = path / "model_source.json"
    source = json.loads(source_manifest.read_text(encoding="utf-8")) if source_manifest.exists() else {}
    identity = source.get("model_id") or config.get("_name_or_path")
    if not identity:
        raise ValueError("model provenance is unknown; download Qwen with download_qwen_model or provide a verified model_source.json")
    if identity.rstrip("/").lower() != MODEL_ID.lower():
        raise ValueError(f"expected {MODEL_ID}, found {identity!r}")
    revision = source.get("revision") or config.get("_commit_hash")
    if not revision:
        raise ValueError("model revision is unknown; expected an immutable Hugging Face commit SHA")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError(f"model revision must be a 40-character commit SHA, found {revision!r}")
    if config.get("model_type") != "qwen3":
        raise ValueError(f"expected Qwen3 architecture, found {config.get('model_type')!r}")
    weight_files = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    if not weight_files:
        raise FileNotFoundError(f"no model weight shards found in {path}")
    return {"path": str(path.resolve()), "model_type": config.get("model_type"),
            "name_or_path": config.get("_name_or_path"), "commit_hash": revision,
            "source_manifest": source, "model_id_verified": identity,
            "architectures": config.get("architectures"), "max_position_embeddings": config.get("max_position_embeddings"),
            "torch_dtype": config.get("torch_dtype"), "tokenizer_config_present": tokenizer_config.is_file(),
            "generation_config": json.loads(generation_config.read_text(encoding="utf-8")) if generation_config.exists() else {}}


def load_local_qwen(model_dir: str | Path, *, precision: str = "float16", model_parallel: bool = True):
    """Load the exact local Qwen snapshot in FP16. Model dispatch is for inference only."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if precision != "float16":
        raise ValueError(f"This Kaggle workflow requires float16, got {precision!r}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    path = str(Path(model_dir).resolve())
    metadata = inspect_local_model(path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    kwargs = {"local_files_only": True, "trust_remote_code": False, "torch_dtype": torch.float16,
              "low_cpu_mem_usage": True, "attn_implementation": "sdpa"}
    if torch.cuda.is_available() and model_parallel:
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.eval()
    return model, tokenizer, metadata


def discover_lora_targets(model) -> list[str]:
    selected = set()
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        class_name = type(module).__name__.lower()
        if leaf in PROJECTION_SUFFIXES and ("linear" in class_name or "conv1d" in class_name):
            selected.add(leaf)
    if not selected:
        raise RuntimeError("No recognized linear projection modules found by model inspection")
    return sorted(selected)
