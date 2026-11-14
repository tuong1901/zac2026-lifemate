# Training report

No Qwen training run has been performed. The local machine has an RTX 3050 Ti with 4 GiB VRAM and no cached Qwen weights.

The revised implementation uses FP16 LoRA on one GPU with dynamically detected LoRA targets, assistant-only loss, explicit length exclusions, finite/nonzero gradient checking on the first optimizer step, training logs, and resumable checkpoints. It does not use FSDP, DDP, or quantization for training.

The notebook refuses model download/training unless two Tesla T4 GPUs each have at least 14 GiB total VRAM and 10 GiB free. Training uses FP16, SDPA, gradient checkpointing, and PyTorch FSDP `FULL_SHARD`, which shards model parameters, gradients, and optimizer state across the two GPUs; activations remain local. A 16 GiB-class T4 selects sequence length 8,192. Existing `lengths.csv` has 3,924/4,397 train rows and 419/472 validation rows at or below 8,192; training-time tokenization produces the authoritative exclusion report. This setup has not been run on Kaggle, so the first backward pass may still expose an out-of-memory condition. Configuration and dependency minimums are in `configs/` and `requirements/kaggle.txt`; the notebook does not replace Torch/CUDA. No loss curve, adapter, or trained checkpoint is available yet.
