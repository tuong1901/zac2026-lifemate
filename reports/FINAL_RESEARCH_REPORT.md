# LifeMate agent training: research and implementation report

## Scope

This report describes the portable local-development and Kaggle-oriented pipeline currently in `lifemate_agent_training/`. The workspace did not contain the source tree or tests from the previous implementation. Earlier reports are preserved in `old/reports/`; the current pipeline and its checks are new, independently reconstructed work.

## Data and reproducibility checks

- Original data: 1,545 training samples and 634 public-test samples.
- SFT data: 4,869 decisions from 1,545 unique source trajectories.
- Source split: 1,390 training sources and 155 validation sources, with zero overlap; all 1,545 SFT sources matched the original training set.
- Original training JSONL SHA-256: `24c42c79fbe8db932fe9817d041c0d0ca422ac8177b413cc1be0cda1fd8941a8`.
- SFT split-manifest SHA-256: `d2c8952240d80b5b7c4618a1cbccf4a1f9611fa4bf283e2657f6504de10ebaab`.
- Replay against the official environment: 3,541 calls, zero result/schema/final-state discrepancies; reconstructed SFT action sequences had zero discrepancies.
- New tests: 16 CPU unit tests and one official ToolEnv integration test pass. The historical ten-test suite could not be rerun because its files are absent.

## System design

Training and inference use a structured, single-action-per-decision policy. Tool calls are checked against the supplied catalog and official environment; only actual tool results update agent state. Trajectory evaluation uses fresh environments per sample, and training excludes overlength rows with an explicit record rather than truncating them. QLoRA and local ASR use attached local assets with network fallback disabled.

The loader requires the exact Qwen model ID and a recorded 40-character source revision. The selected upstream model is Qwen3-4B-Instruct-2507, the non-thinking variant; its official card documents a native 262,144-token context window. This is not the configured training length, which is bounded by available GPU resources. See [official model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) and [Qwen Transformers guide](https://github.com/QwenLM/Qwen3/blob/main/docs/source/inference/transformers.md).

## Status and limits

No base weights or ASR snapshot were present, and no Kaggle session was available. Therefore no model load, baseline inference, fine-tuning, checkpoint comparison, public-test inference, or Kaggle run is claimed. The laptop GPU has 4 GiB VRAM; full training is gated to Linux/CUDA with at least 15 GiB total and 12 GiB free VRAM. The Kaggle dependency file is a candidate stack and has not been validated against a live Kaggle image. No GitHub remote or push exists.

## Artifacts

See [data analysis](DATASET_ANALYSIS.md), [training status](TRAINING_REPORT.md), [evaluation status](EVALUATION_REPORT.md), [failure analysis](FAILURE_ANALYSIS.md), [reproducibility](REPRODUCIBILITY.md), and [Kaggle guide](KAGGLE_EXECUTION_GUIDE.md).
