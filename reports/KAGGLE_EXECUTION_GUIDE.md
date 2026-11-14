# Kaggle one-notebook workflow

## Run

Open only `notebooks/kaggle_train_eval.ipynb`, select GPU, enable Internet, and use **Run All**. Internet is required to clone GitHub branch `frame-agent`, install missing Python packages, and download `Qwen/Qwen3-4B-Instruct-2507` from Hugging Face. The model is not expected in Kaggle Input.

The notebook uses these fixed inputs:

- `/kaggle/input/datasets/estherle/sft-data/sft_full_history_v1`
- `/kaggle/input/datasets/estherle/zalo-dataset/zalo-ai-challenge/train+public_test`

It preserves the SFT manifest split, never trains on public_test, runs data audit and official train ToolEnv replay, then performs baseline and fine-tuned validation with the existing evaluation scripts.

## FP32 and GPU feasibility

Training uses FP32 LoRA with PyTorch FSDP `FULL_SHARD`, auto-wrap at Qwen decoder layers, and `use_orig_params=True`. DDP alone would replicate the full model per GPU and is not used as a substitute for model sharding. Training does not use FP16, BF16, `device_map`, or quantization. Before downloading weights, the notebook requires at least two Tesla T4 GPUs, 30 GiB aggregate VRAM, and 12 GiB free on each GPU. If the runtime fails any condition, it stops before model download. Even passing this conservative gate does not constitute a verified training run; FSDP initialization and a real backward step remain runtime-dependent.

The dependency step does not replace PyTorch/CUDA. It requires Torch >=2.6 and Transformers >=4.51 after inspecting the image. If Kaggle's current Torch build does not satisfy this, the notebook stops and asks for a runtime choice rather than changing precision or CUDA stack.

If the gate fails, no Qwen weights are downloaded and no training starts. The FP32 options are a Kaggle 2x T4 session or another supported multi-GPU runtime meeting the stated aggregate/free VRAM checks. The other technical option is to explicitly change the precision requirement (for example FP16/BF16 or quantization); the notebook never does that on its own.

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
