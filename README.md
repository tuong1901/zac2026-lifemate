# LifeMate Agent Training

Portable local-development and Kaggle-training project for the Zalo AI Challenge 2026 LifeMate task. The repository contains code/configuration only. Competition data, model weights, adapters, and generated results are separate assets and are excluded from Git.

## Current recheck status

The earlier implementation source and test files were absent from the current workspace. The old reports remain unchanged in `../../old/reports/`. This project independently rechecked their data claims against the present datasets and official `ToolEnv`:

- 1,545 train / 634 public-test samples; 4,869 SFT decisions and 1,545 sources.
- Source split: 1,390 train / 155 validation; overlap 0; source match 1,545/1,545.
- Full official train replay: 3,541 calls, 0 result/schema/final-state discrepancies.
- SFT decisions reconstructed the original call sequences: 0 discrepancies.
- Source train SHA-256: `24c42c79fbe8db932fe9817d041c0d0ca422ac8177b413cc1be0cda1fd8941a8`.

The historical 10-test suite cannot be rerun because its test files were not present. This recreated code has its own CPU unit tests and ToolEnv integration test.

## Local setup (Windows PowerShell)

```powershell
cd E:\document\zalo-challenge\lifemate_agent_training
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements/local.txt
$env:SFT_DATA_DIR = 'E:\document\zalo-challenge\sft_full_history_v1'
$env:ORIGINAL_DATA_DIR = 'E:\document\zalo-challenge\train+public_test'
```

The supplied competition package currently exists at the workspace-level `train+public_test`; the requested nested `zalo-ai-challenge\train+public_test` path is absent in this checkout. `scripts/prepare_data.py` and `scripts/verify_toolenv.py` discover either layout or accept `ORIGINAL_DATA_DIR` explicitly.

Run local CPU checks and preparation:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m unittest discover -s tests -p test_core.py -v
python -m unittest discover -s tests\integration -v
python scripts\audit_data.py
python scripts\prepare_data.py
python scripts\verify_toolenv.py
```

These commands never train Qwen on the local RTX 3050 Ti. The recorded local GPU has 4 GiB VRAM; `scripts/train.py` also refuses non-Linux execution.

## Kaggle workflow

The only entry point is [notebooks/kaggle_train_eval.ipynb](notebooks/kaggle_train_eval.ipynb). It clones `https://github.com/tuong1901/zac2026-lifemate.git`, branch `frame-agent`, and uses the two fixed Kaggle dataset paths in the notebook. Enable Kaggle Internet for GitHub cloning, missing Python dependencies, and the Hugging Face download. No model is expected in Kaggle Input.

Training is FP32 LoRA with PyTorch FSDP `FULL_SHARD`, transformer decoder-layer auto-wrap, and `use_orig_params=True`; training does not use DDP, `device_map`, FP16/BF16, or quantization. The notebook checks the live runtime and requires at least two Tesla T4 GPUs, 30 GiB aggregate VRAM, and 12 GiB free on each before downloading the model. A one-T4 session stops before download. The check is a conservative gate, not a guarantee of runtime fit. Lengths beyond the selected limit are recorded and excluded, not truncated.

The notebook clones source into `/kaggle/working/lifemate_repo`; run outputs, downloaded base weights, checkpoints, adapter, logs and metrics go under `/kaggle/working/lifemate_run`. It exports a timestamped `/kaggle/working/lifemate_export_*.zip` without base weights or datasets and prints the exact file path. See [Kaggle execution guide](reports/KAGGLE_EXECUTION_GUIDE.md) for output locations and retrieval.

Qwen3-4B-Instruct-2507 is the non-thinking variant; its shipped chat template is used without injecting a thinking-mode flag. Transformers support for Qwen3 begins at 4.51.0 and the Qwen guide recommends PyTorch 2.6 or newer. See the [model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) and [Qwen Transformers guide](https://github.com/QwenLM/Qwen3/blob/main/docs/source/inference/transformers.md).

## Model provenance

The notebook downloads `Qwen/Qwen3-4B-Instruct-2507` directly from Hugging Face after checking the GPU gate, pins the resolved commit SHA, and writes `model_source.json` automatically. Training uses the local downloaded snapshot and does not use hosted inference. Keep the base model and LoRA adapter separate; the adapter alone is not a complete deployment.

## Git/GitHub

The project directory has the `origin` remote and `frame-agent` branch. `.gitignore` excludes data, weights, checkpoints, virtual environments, Kaggle files, logs, and generated submissions. Changes in this working tree are not pushed automatically. Review `git status`/`git diff`, then commit and push to `frame-agent` before the Kaggle notebook clones it.

After reviewing `git status` and `git diff`, the initial commit can be made with:

```powershell
git add .
git diff --cached --stat
git diff --cached
git commit -m "Refactor Kaggle training to FP32 FSDP"
git push origin frame-agent
```

Never add the SFT/original data or model assets. A code-only ZIP is the Kaggle transfer alternative when GitHub access is unavailable.

## Reports and experiment outputs

`reports/` contains current recheck/status reports. `artifacts/` and `data/processed/` are generated locally and ignored by Git. Kaggle run artifacts persist under `/kaggle/working/lifemate_run`; download the exported ZIP or save a private Kaggle Dataset/version before the session ends.
