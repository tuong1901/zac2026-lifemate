from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from lifemate.config import discover_original_data
from lifemate.data.loader import iter_jsonl
from lifemate.deployment.submission import validate_submission


def main():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--original-dir", default=os.environ.get("ORIGINAL_DATA_DIR"))
    p.add_argument("--submission", default=str(repo/"result/submission.jsonl"))
    args = p.parse_args()
    candidates = [args.original_dir] if args.original_dir else [repo.parent/"train+public_test", repo.parent/"zalo-ai-challenge/train+public_test", "/kaggle/input/zalo-original-data"]
    data = discover_original_data(candidates)
    expected = [r["id"] for r in iter_jsonl(data/"public_test/public_test.jsonl")]
    result = validate_submission(args.submission, expected)
    command = [sys.executable, "-m", "zaic_tools", "--split", "public_test", "--root", str(data), "--validate", str(Path(args.submission).resolve())]
    completed = subprocess.run(command, cwd=data/"public_test", text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(completed.stdout+completed.stderr)
    result["official_validation"] = completed.stdout.strip()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
