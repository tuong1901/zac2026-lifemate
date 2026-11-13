# Kaggle data and model attachment guide

Attach each asset as its own private Kaggle Dataset/Model input. Do not put restricted competition data in a public GitHub repository.

## Inputs

1. **SFT data** — folder containing `train_all_lengths.jsonl`, `validation_all_lengths.jsonl`, `split_manifest.json`, and `lengths.csv`. Keep public-test data out of this asset.
2. **Original competition package** — root containing `train/train.jsonl`, `train/zaic_tools/`, `public_test/public_test.jsonl`, and corresponding catalogs/environments/audio.
3. **Qwen base model** — local snapshot of `Qwen/Qwen3-4B-Instruct-2507` with tokenizer/config/weight shards and `model_source.json` containing its exact model ID and 40-character commit SHA.
4. **Offline ASR** — local Vietnamese-capable Whisper-family model snapshot (required for public audio inference). Store separately from Qwen.
5. **Project code** — private repository or a ZIP containing source/config/docs/notebook only. A source ZIP can be uploaded as another Kaggle Dataset. Do not include the other four assets in it.

## Mount discovery

The notebook scans `/kaggle/input` for file markers; no Kaggle slug is hardcoded. To override a discovered path, set `SFT_DATA_DIR`, `ORIGINAL_DATA_DIR`, `BASE_MODEL_DIR`, and `ASR_MODEL_DIR` in the notebook before the input-discovery cell. The original root must directly contain `train/` and `public_test/`.

The notebook's optional `LIFEMATE_GIT_URL` is for source transfer only and is used only when code is not attached. Training, model inference, and transcription use attached local inputs with offline flags.

## Artifact export

The notebook writes to `/kaggle/working/lifemate_run`, then creates `/kaggle/working/lifemate_run_export.zip`. It exports reports, metrics, adapter, logs, and submission, but does not copy attached datasets or base-model weights. The adapter must be paired with the exact base-model revision for offline competition deployment. Save the exported ZIP as a private Kaggle Dataset/version or download it before ending the session.
