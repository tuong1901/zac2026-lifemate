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

See [data attachment guide](docs/DATA_SETUP.md) and [Kaggle execution guide](reports/KAGGLE_EXECUTION_GUIDE.md). Attach the SFT data, original competition package, Qwen base model, ASR model, and source ZIP/private repository as separate inputs. The notebook discovers mount locations from file markers. It loads model and ASR assets locally with Hugging Face offline flags; inference never calls a hosted LLM or transcription API.

Open `notebooks/kaggle_train_eval.ipynb`. The notebook checks GPU/VRAM, RAM, storage and installed packages before running. Costly full training and public inference are controlled by explicit notebook flags. QLoRA starts only on Linux/CUDA with at least 15 GiB total and 12 GiB free VRAM, after a reversible forward/backward smoke step. Context over the selected sequence length is recorded in `length_exclusions.json`, not silently truncated.

Qwen3-4B-Instruct-2507 is the non-thinking variant; its chat template is used without injecting `enable_thinking`. The upstream model card lists native 262,144-token context; actual training sequence length is GPU-bounded. Transformers support for Qwen3 begins at 4.51.0 and the Qwen guide recommends PyTorch 2.6 or newer. See the [model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) and [Qwen Transformers guide](https://github.com/QwenLM/Qwen3/blob/main/docs/source/inference/transformers.md). Kaggle still must be checked dynamically; the pinned candidate stack is in `requirements/kaggle.txt` and is not claimed tested on Kaggle.

## Model provenance

The model directory must identify `Qwen/Qwen3-4B-Instruct-2507` and its exact 40-character source revision. Add `model_source.json` beside the attached model files using `model_source.json.example`. The loader refuses unknown IDs/revisions, network fallback, and alternate model sizes. Keep the base model and LoRA adapter separate; the adapter alone is not an offline deployment.

## Git/GitHub

The project directory is a separate repository from datasets and legacy work. `.gitignore` excludes data, weights, checkpoints, virtual environments, Kaggle files, logs, and generated submissions. No GitHub remote is configured and no push has been performed.

After reviewing `git status` and `git diff`, the initial commit can be made with:

```powershell
git add .
git diff --cached --stat
git diff --cached
git commit -m "Build LifeMate workflow agent training pipeline"
git remote add origin <PRIVATE_OR_CODE_ONLY_GITHUB_REMOTE>
git push -u origin main
```

Never add the SFT/original data or model assets. A code-only ZIP is the Kaggle transfer alternative when GitHub access is unavailable.

## Reports and experiment outputs

`reports/` contains current recheck/status reports. `artifacts/` and `data/processed/` are generated locally and ignored by Git. Kaggle run artifacts persist under `/kaggle/working/lifemate_run`; download the exported ZIP or save a private Kaggle Dataset/version before the session ends.
