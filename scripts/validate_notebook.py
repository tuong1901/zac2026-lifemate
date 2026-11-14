"""Static checks for the generated single-entry Kaggle FSDP notebook."""
from __future__ import annotations

import ast
import json
from pathlib import Path


def validate(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError("expected a Jupyter notebook v4 document")
    if notebook.get("metadata", {}).get("kaggle", {}).get("internet") is not True:
        raise ValueError("Kaggle Internet must be enabled for repository/model/dependency downloads")
    cells = notebook.get("cells", [])
    code = ["".join(cell.get("source", [])) for cell in cells if cell.get("cell_type") == "code"]
    all_text = "\n".join("".join(cell.get("source", [])) for cell in cells)
    for index, source in enumerate(code):
        ast.parse(source, filename=f"{path}:code-cell-{index}")
    merged = "\n".join(code)
    required = (
        "https://github.com/tuong1901/zac2026-lifemate.git", "frame-agent",
        "/kaggle/input/datasets/estherle/sft-data/sft_full_history_v1",
        "/kaggle/input/datasets/estherle/zalo-dataset/zalo-ai-challenge/train+public_test",
        "Qwen/Qwen3-4B-Instruct-2507", "float16", "check_fsdp_hardware", "FULL_SHARD", "torch.distributed.run",
        "--smoke-only", "fsdp_smoke_result.json",
    )
    missing = [marker for marker in required if marker not in all_text]
    if missing:
        raise ValueError(f"notebook is missing required workflow markers: {missing}")
    gate = merged.find("check_fsdp_hardware(runtime)")
    download = merged.find("download_qwen_model(MODEL_DIR)")
    if gate < 0 or download < 0 or gate > download:
        raise ValueError("hardware feasibility gate must precede the model download")
    if "INSTALL_STACK = False" in merged or "RUN_FULL_TRAINING = False" in merged:
        raise ValueError("one-click notebook still contains opt-in workflow flags")
    if "--nproc_per_node" not in merged or "len(gpus)" not in merged:
        raise ValueError("notebook must launch one FSDP process for each checked GPU")
    return {"valid": True, "cells": len(cells), "code_cells": len(code), "path": str(path)}


if __name__ == "__main__":
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks/kaggle_train_eval.ipynb"
    print(json.dumps(validate(notebook_path), indent=2))
