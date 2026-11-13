# Kaggle execution guide

1. Create separate private Kaggle inputs for SFT data, the original competition package, the exact Qwen base-model snapshot, offline Vietnamese ASR, and the code-only project ZIP/repository.
2. Ensure the original package root contains `train/` and `public_test/`; ensure the SFT root contains the prepared split and manifest. Add `model_source.json` beside Qwen weights with the exact model ID and immutable revision.
3. Open `notebooks/kaggle_train_eval.ipynb`, attach the inputs, and run discovery, hardware inspection, dependency inspection, data audit/preparation, and ToolEnv replay first.
4. Inspect the live Kaggle image and available GPU before enabling model smoke, baseline, training, checkpoint evaluation, or public-test flags. The notebook defaults expensive stages to off. Do not upgrade Torch blindly.
5. Save `/kaggle/working/lifemate_run_export.zip` and the run reports as a private Kaggle artifact or download them before the session ends.

This notebook has only been generated and syntax-checked locally. It has not been run on Kaggle. The requirements file is a candidate pin set, not a validated Kaggle environment. Never place competition data, model weights, Kaggle credentials, or public-test outputs in a public repository.
