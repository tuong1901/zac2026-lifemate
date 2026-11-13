"""Run validation step and trajectory evaluation on the unchanged local base model."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--original-dir")
    p.add_argument("--sft-dir")
    p.add_argument("--data")
    p.add_argument("--output-dir", default="artifacts/evaluation/baseline")
    p.add_argument("--max-samples", type=int)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    original_args = ["--original-dir", args.original_dir] if args.original_dir else []
    trajectory_args = original_args + (["--sft-dir", args.sft_dir] if args.sft_dir else [])
    step_args = (["--data", args.data] if args.data else [])
    if args.max_samples:
        step_args += ["--max-samples", str(args.max_samples)]
        trajectory_args += ["--max-samples", str(args.max_samples)]
    cmds = [[sys.executable, str(repo/"scripts/evaluate_steps.py"), "--model-dir", args.model_dir,
             "--output-dir", str(Path(args.output_dir)/"steps"), *original_args, *step_args],
            [sys.executable, str(repo/"scripts/evaluate_trajectories.py"), "--model-dir", args.model_dir,
             "--output-dir", str(Path(args.output_dir)/"trajectories"), *trajectory_args]]
    for cmd in cmds: subprocess.run(cmd, check=True, cwd=repo)


if __name__ == "__main__": main()
