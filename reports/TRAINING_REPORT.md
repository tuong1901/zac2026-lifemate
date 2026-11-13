# Training report

No Qwen training run has been performed. The machine has an RTX 3050 Ti with 4 GiB VRAM, and no Qwen base weights are cached. Training therefore remains a Kaggle execution task.

The implementation provides 4-bit NF4 QLoRA, dynamically detected target modules, assistant-only loss, length accounting without silent truncation, a reversible forward/backward smoke step, training/evaluation logs, and bounded checkpoint retention. It refuses to start unless Linux/CUDA reports at least 15 GiB total VRAM and 12 GiB free. These thresholds are gates, not evidence that a particular Kaggle accelerator will fit the run.

The current training configuration and candidate dependencies are in `configs/` and `requirements/kaggle.txt`. The pinned environment has not been installed or tested in Kaggle; inspect the live image before changing Torch or installing packages. No loss curve, adapter, or trained checkpoint is available yet.
