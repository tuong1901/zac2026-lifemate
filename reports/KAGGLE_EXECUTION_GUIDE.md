# Kaggle one-notebook workflow

## Run

Open only `notebooks/kaggle_train_eval.ipynb`, select GPU, enable Internet, and use **Run All**. Internet is required to clone GitHub branch `frame-agent`, install missing Python packages, and download `Qwen/Qwen3-4B-Instruct-2507` from Hugging Face. The model is not expected in Kaggle Input.

The notebook uses these fixed inputs:

- `/kaggle/input/datasets/estherle/sft-data/sft_full_history_v1`
- `/kaggle/input/datasets/estherle/zalo-dataset/zalo-ai-challenge/train+public_test`

It preserves the SFT manifest split, never trains on public_test, runs data audit and official train ToolEnv replay, then performs baseline and fine-tuned validation with the existing evaluation scripts.

## FP16 and GPU feasibility

Training uses FP16 LoRA on one selected Tesla T4 with the standard Trainer and SDPA attention. It does not use FSDP, DDP, or quantization. Before downloading weights, the notebook requires at least 14 GiB total and 12 GiB free on the selected GPU. If the runtime fails either condition, it stops before model download. A 16 GiB-class T4 selects an 8,192-token cap with gradient checkpointing; rows exceeding it are explicitly logged/excluded. Existing lengths indicate 3,924/4,397 train rows and 419/472 validation rows fit that cap, while the preparation report produced during training is authoritative. The memory gate does not guarantee a successful run; the model smoke and first optimizer step still validate actual runtime fit and gradients.

The dependency step does not replace PyTorch/CUDA. It requires Torch >=2.6 and Transformers >=4.51 after inspecting the image. If Kaggle's current Torch build does not satisfy this, the notebook stops and asks for a runtime choice rather than changing precision or CUDA stack.

The setup checks the optional `torchao` package before importing PEFT. If the Kaggle image contains `torchao<0.16`, it removes only that unused optional package; the FP16 model and standard LoRA path do not use TorchAO. This avoids an upstream PEFT compatibility error that can occur when LoRA injection sees an old TorchAO installation. Torch and CUDA are left untouched.

If the gate fails, no Qwen weights are downloaded and no training starts. Select a clean Tesla T4 session with enough free VRAM, or revise the model/precision requirements explicitly. Precision is never changed automatically.

## Outputs

- Cloned source: `/kaggle/working/lifemate_repo`
- Run root and downloaded base model: `/kaggle/working/lifemate_run`
- Checkpoints: `/kaggle/working/lifemate_run/training/checkpoints`
- LoRA adapter and tokenizer: `/kaggle/working/lifemate_run/training/adapter`
- Training config, logs, metrics, manifest: `/kaggle/working/lifemate_run/training/`
- Data audit/preparation: `/kaggle/working/lifemate_run/audit` and `/kaggle/working/lifemate_run/processed`
- Base and fine-tuned validation reports: `/kaggle/working/lifemate_run/baseline`, `/kaggle/working/lifemate_run/finetuned`, `/kaggle/working/lifemate_run/evaluation`
- Downloadable export: `/kaggle/working/lifemate_export_<timestamp>.zip` (the notebook prints the exact filename)

Download the printed ZIP from the notebook's Output/Files panel before the Kaggle session ends. It includes reports, checkpoints, adapter, logs, and metrics; it excludes the large base-model snapshot and input datasets. Pair the exported adapter with the exact base revision recorded in `training_config.json` and `model_source.json`.

This pipeline has been syntax- and consistency-checked locally only. It has not been executed on Kaggle, and no training result is claimed.
