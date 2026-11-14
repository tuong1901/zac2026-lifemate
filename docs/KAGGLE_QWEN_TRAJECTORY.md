# Train all v2 targets with less repeated context

Use `notebooks/kaggle_qwen_trajectory_t4.ipynb`. Attach the same private v2
Dataset as before, enable Internet and T4 x2, and run cells in order.
The notebook reuses the isolated Python 3.11 / PyTorch 2.8 CUDA 12.8 environment.
It clones public branch `test`, then converts the attached v2 JSONL locally.
No new Dataset upload is required.

## What changes

The v2 decision dataset repeats the entire tool catalog, context and history for
each action. Trajectories store the initial prompt once, followed by alternating
assistant actions and incremental observed results. All original assistant JSON
targets remain unchanged, including ask/stop decisions. Source train/validation
membership stays unchanged: 1,390 train sources / 155 validation sources, containing
4,397 / 472 assistant targets respectively. No ASR text correction, tool pruning,
sample dropping or token truncation is introduced.

Local audit with the model tokenizer: train tokens per epoch fall from 25,753,391
to 8,129,677 (68.4% fewer), while all 161,532 supervised tokens remain. Validation
tokens fall from 2,702,721 to 893,315, preserving all 17,414 supervised tokens.
The longest trajectory is 10,433 tokens, within the unchanged 12,288 limit.
Token reduction is not a measured GPU speedup; longer individual trajectories
and evaluation/checkpoint overhead also affect wall time.

All assistant turns contribute to loss; prompts and tool results are masked.
Causal attention prevents earlier actions from seeing later tool results.
The data converter rejects changed contexts, missing steps and mismatched results.

The initial experiment uses **one epoch**, compared with two in the previous run.
This is a separate training choice: reduced repetition alone does not imply that
one epoch matches the quality of two. Evaluate tool execution on validation before
deciding whether to run another experiment with two epochs. There is no measured
GPU runtime or guaranteed accuracy improvement for this format yet.

Do not reduce gradient accumulation just to make the progress bar advance faster:
at fixed data and epochs, this does not substantially reduce total token work.

## Existing runtime

Allow checkpoint saving to finish before interrupting the old training cell.
Download the checkpoint and `run_config.json` or save the notebook outputs.
Keep the old `qwen_v2_fast_training` directory. This notebook writes to
`/kaggle/working/qwen_trajectory_training` and uses a new smoke receipt.
Changing data format is a new experiment; do not resume the old v2 optimizer with
these trajectories. Exact resume requires the old data, script and settings.

If using the current runtime, run setup/pull again and run the new notebook's data,
smoke and training cells. Environment installation is reusable. Do not replace
files while the old training process is still running.

## Inference contract

This adapter needs the trajectory message layout during evaluation. The legacy
per-decision prompt runner is not automatically switched by this training notebook.

1. Start with the existing `RAW_SYSTEM_PROMPT` and `build_raw_user_prompt` using
   complete initial context/tools/history and empty current-session previous steps.
2. Generate assistant JSON and append that exact response as an assistant message.
3. Execute its action. Build an observation with `tool_name`, `arguments`, `result`.
4. Append `scripts.prepare_qwen_trajectories.build_observation_message(observation)`
   as a user message, then generate the next assistant response.
5. Stop when `is_finished` is true. Do not append future results in advance.

The initial conversation history remains complete; only duplicated prefixes across
decisions are removed. GPU smoke checks numerical updates and memory, not execution
accuracy. Save/download `final_adapter` and recovery outputs before runtime expiry.
