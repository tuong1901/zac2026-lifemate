from __future__ import annotations

import json
import re
from pathlib import Path


PROJECTION_SUFFIXES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


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
        raise ValueError("model provenance is unknown; add model_source.json containing model_id and revision")
    if identity.rstrip("/").lower() != "qwen/qwen3-4b-instruct-2507":
        raise ValueError(f"expected Qwen/Qwen3-4B-Instruct-2507, found {identity!r}")
    revision = source.get("revision") or config.get("_commit_hash")
    if not revision:
        raise ValueError("model revision is unknown; provide a pinned commit SHA in model_source.json")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError(f"model revision must be a 40-character commit SHA, found {revision!r}")
    return {"path": str(path.resolve()), "model_type": config.get("model_type"),
            "name_or_path": config.get("_name_or_path"), "commit_hash": revision,
            "source_manifest": source, "model_id_verified": identity,
            "architectures": config.get("architectures"), "max_position_embeddings": config.get("max_position_embeddings"),
            "torch_dtype": config.get("torch_dtype"), "tokenizer_config_present": tokenizer_config.is_file(),
            "generation_config": json.loads(generation_config.read_text(encoding="utf-8")) if generation_config.exists() else {}}


def load_local_qwen(model_dir: str | Path, *, quantized: bool = True):
    """Load only local assets. 4-bit is enabled only on CUDA and never falls back to a different model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = str(Path(model_dir).resolve())
    metadata = inspect_local_model(path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    kwargs = {"local_files_only": True, "trust_remote_code": False, "device_map": "auto"}
    if quantized:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit QLoRA/inference requested without CUDA; refusing an implicit CPU model load")
        from transformers import BitsAndBytesConfig
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.eval()
    if metadata["model_type"] != "qwen3":
        raise ValueError(f"Expected Qwen3 architecture; found {metadata['model_type']!r}")
    return model, tokenizer, metadata


def discover_lora_targets(model) -> list[str]:
    """Inspect loaded modules and return projection suffixes actually present and linear."""
    selected = set()
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        class_name = type(module).__name__.lower()
        if leaf in PROJECTION_SUFFIXES and ("linear" in class_name or "conv1d" in class_name):
            selected.add(leaf)
    if not selected:
        raise RuntimeError("No recognized linear projection modules found by model inspection")
    return sorted(selected)
