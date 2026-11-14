# Kaggle dataset inputs

The one entry-point notebook uses the fixed Kaggle mount locations below. Do not copy or modify files under `/kaggle/input`.

- SFT split: `/kaggle/input/datasets/estherle/sft-data/sft_full_history_v1`
- Zalo package: `/kaggle/input/datasets/estherle/zalo-dataset/zalo-ai-challenge/train+public_test`

The notebook verifies the SFT train/validation files and split manifest, plus the original `train/train.jsonl`, `train/zaic_tools/`, and `public_test/public_test.jsonl`. It then reuses `prepare_data.py` and `verify_toolenv.py`. The provided split is retained; public_test is not read into training or validation. Missing files stop the notebook without creating replacement data.

No base model is needed in Kaggle Input. After the GPU gate passes, the notebook downloads `Qwen/Qwen3-4B-Instruct-2507` from Hugging Face and automatically records the exact commit revision.
