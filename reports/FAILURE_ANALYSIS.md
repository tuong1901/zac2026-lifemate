# Failure analysis status

There are no model-generated failures to analyze yet because no baseline or fine-tuned inference was run. The dataset/environment replay produced zero discrepancies over 3,541 official tool calls, and SFT call reconstruction produced zero action-sequence discrepancies.

Evaluation code records invalid structured outputs, invalid/failed tool calls, task completion, context overflow, and trajectory-level results. Once model runs exist, aggregate these categories and inspect representative failures before changing prompts, data, or training settings. Do not infer model quality from environment replay.
