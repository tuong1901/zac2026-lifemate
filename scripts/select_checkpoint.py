from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description="Select among retained LoRA checkpoints using the same labeled validation set.")
    p.add_argument("--model-dir",required=True);p.add_argument("--checkpoint-root",required=True)
    p.add_argument("--original-dir",required=True);p.add_argument("--sft-dir",required=True)
    p.add_argument("--output-dir",required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    root=Path(a.checkpoint_root);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    candidates=sorted((p for p in root.glob("checkpoint-*") if (p/"adapter_config.json").is_file() and any(p.glob("adapter_model.*"))),key=lambda p:int(p.name.rsplit('-',1)[1]))
    if not candidates: raise FileNotFoundError(f"No adapter checkpoints in {root}")
    selection=[]
    for idx,candidate in enumerate(candidates):
        run=out/f"candidate_{idx:02d}"
        commands=[
            [sys.executable,str(repo/"scripts/evaluate_steps.py"),"--model-dir",a.model_dir,"--adapter-dir",str(candidate),"--original-dir",a.original_dir,"--data",str(Path(a.sft_dir)/"validation_all_lengths.jsonl"),"--output-dir",str(run/"steps")],
            [sys.executable,str(repo/"scripts/evaluate_trajectories.py"),"--model-dir",a.model_dir,"--adapter-dir",str(candidate),"--original-dir",a.original_dir,"--sft-dir",a.sft_dir,"--output-dir",str(run/"trajectories")],
        ]
        for command in commands: subprocess.run(command,cwd=repo,check=True)
        step=json.loads((run/"steps/metrics.json").read_text(encoding="utf-8"))["metrics"]["decision_correct"]
        traj=json.loads((run/"trajectories/trajectory_evaluation.json").read_text(encoding="utf-8"))["task_completion_rate"]
        selection.append({"checkpoint":str(candidate),"step_decision_accuracy":step,"trajectory_completion_rate":traj,
                          "selection_score":(step+traj)/2})
    best=max(selection,key=lambda r:(r["selection_score"],int(Path(r["checkpoint"]).name.rsplit('-',1)[1])))
    destination=out/"adapter_selected";destination.mkdir(parents=True,exist_ok=True)
    source=Path(best["checkpoint"])
    for file in source.glob("adapter_*"):
        if file.is_file():shutil.copy2(file,destination/file.name)
    meta={"selected_checkpoint":best["checkpoint"],"rule":"highest arithmetic mean of validation step decision accuracy and trajectory task completion rate; exact ties choose newer checkpoint",
          "candidates":selection,"selected":best}
    (out/"checkpoint_selection.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(meta,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
