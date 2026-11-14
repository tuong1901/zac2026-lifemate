# Training report

No Qwen training run has been performed. The local machine has an RTX 3050 Ti with 4 GiB VRAM and no cached Qwen weights.

The revised implementation uses FP32 LoRA with PyTorch FSDP `FULL_SHARD`, transformer-layer auto-wrap, `use_orig_params=True`, dynamically detected LoRA targets, assistant-only loss, explicit length exclusions, finite/nonzero gradient checking on the first optimizer step, training logs, and resumable checkpoints. It does not use DDP, inference `device_map`, FP16/BF16, or quantization for training.

The notebook refuses model download/training unless at least two Tesla T4 GPUs, 30 GiB aggregate VRAM, and 12 GiB free on every GPU are detected. These gates do not guarantee a successful runtime fit. Configuration and dependency minimums are in `configs/` and `requirements/kaggle.txt`; they have not been tested on Kaggle. The notebook does not replace Torch/CUDA. No loss curve, adapter, or trained checkpoint is available yet.
