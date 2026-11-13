# Evaluation report

No baseline or fine-tuned model evaluation has been run because model weights and a Kaggle session are unavailable. Do not interpret the data replay as model performance.

The pipeline includes decision-level metrics, closed-loop trajectory evaluation through the official ToolEnv, public-test inference, baseline/fine-tuned comparison, and checkpoint selection using the mean of decision accuracy and trajectory task completion. Each trajectory receives a fresh environment. Context overflows are recorded as skipped evaluations; prompts are not silently truncated.

After training, run the base and selected adapter on the same validation examples and report decision accuracy, schema/tool validity, task completion, tool-call count, and failures. Only then enable public-test inference and submission generation.
