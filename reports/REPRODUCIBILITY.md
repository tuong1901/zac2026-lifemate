# Reproducibility record

## Inputs

- Original train SHA-256: `24c42c79fbe8db932fe9817d041c0d0ca422ac8177b413cc1be0cda1fd8941a8`
- SFT split manifest SHA-256: `d2c8952240d80b5b7c4618a1cbccf4a1f9611fa4bf283e2657f6504de10ebaab`
- Model identity requirement: `Qwen/Qwen3-4B-Instruct-2507` plus exact 40-character source revision in `model_source.json`.
- Data paths are external and set with `SFT_DATA_DIR` and `ORIGINAL_DATA_DIR`.

## Checks completed

- 16/16 recreated CPU unit tests passed using `unittest`.
- 1/1 official ToolEnv integration test passed.
- Full official train replay: 3,541 calls, zero discrepancies.
- SFT action reconstruction: zero discrepancies.
- All notebook code cells passed Python syntax parsing; `compileall` passed for source, scripts, and tests.

The historical test files were not present, so the prior ten-test suite cannot be reproduced. No model run, Kaggle run, or GitHub action is represented as completed. ToolEnv emits upstream `ResourceWarning`s for unclosed JSON files during its integration test; the test itself passes.
