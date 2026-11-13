from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lifemate.config import discover_original_data, load_yaml, resolve_path
from lifemate.data.prepare import prepare


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-dir")
    parser.add_argument("--original-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    paths = load_yaml(repo / "configs/paths.yaml")
    sft = Path(args.sft_dir or os.environ.get("SFT_DATA_DIR") or paths["sft_data_dir"]).expanduser()
    sft = resolve_path(sft, repo)
    configured_original = args.original_dir or os.environ.get("ORIGINAL_DATA_DIR")
    candidates = [configured_original] if configured_original else [paths["original_data_dir"], "../zalo-ai-challenge/train+public_test", "/kaggle/input/zalo-original-data"]
    original = discover_original_data([resolve_path(x, repo) if not Path(x).is_absolute() else x for x in candidates])
    out = Path(args.output_dir or os.environ.get("LIFEMATE_DATA_OUTPUT", repo / "data/processed"))
    if not out.is_absolute():
        out = (repo / out).resolve()
    manifest = prepare(sft, original, out)
    print(json.dumps({k: v for k, v in manifest.items() if k != "source_split"}, indent=2))


if __name__ == "__main__":
    main()
