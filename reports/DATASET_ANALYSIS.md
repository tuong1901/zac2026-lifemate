# Dataset analysis

## Verified inventory

| Asset | Count |
|---|---:|
| Original training trajectories | 1,545 |
| Public-test trajectories | 634 |
| SFT decision rows | 4,869 |
| Unique SFT source trajectories | 1,545 |
| Training / validation source trajectories | 1,390 / 155 |
| Source overlap | 0 |

All 1,545 SFT sources matched the original train set. The train JSONL SHA-256 is `24c42c79fbe8db932fe9817d041c0d0ca422ac8177b413cc1be0cda1fd8941a8`; the split manifest SHA-256 is `d2c8952240d80b5b7c4618a1cbccf4a1f9611fa4bf283e2657f6504de10ebaab`.

The official ToolEnv replay executed 3,541 calls with no result, schema, or final-state discrepancies. Reconstructed SFT actions matched the original call sequences. Preparation found no excluded rows. Tokenized prompt-plus-target length statistics were median 6,179, p90 8,256, maximum 12,276 under the selected local tokenizer/configuration.

## Data separation

Keep the restricted competition package and SFT data outside the code repository. The supplied root is `../train+public_test`; the previously referenced nested `zalo-ai-challenge\train+public_test` path is absent in this workspace. `scripts/audit_data.py`, `scripts/prepare_data.py`, and `scripts/verify_toolenv.py` accept explicit environment paths.

The public-test set is not used for fitting or checkpoint selection. Keep test audio and labels private according to competition rules. The split is by source trajectory, preventing decisions from the same trajectory appearing in both train and validation.
