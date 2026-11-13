from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline-steps",required=True)
    p.add_argument("--finetuned-steps",required=True)
    p.add_argument("--baseline-trajectories",required=True)
    p.add_argument("--finetuned-trajectories",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    files={k:json.loads(Path(v).read_text(encoding="utf-8")) for k,v in vars(a).items() if k.endswith("steps") or k.endswith("trajectories")}
    step_metrics={name:obj.get("metrics",obj) for name,obj in files.items() if name.endswith("steps")}
    traj_metrics={name:{k:v for k,v in obj.items() if isinstance(v,(int,float))} for name,obj in files.items() if name.endswith("trajectories")}
    comparisons={}
    for category, pair in (("steps",step_metrics),("trajectories",traj_metrics)):
        if not {"baseline_"+category,"finetuned_"+category}.issubset(pair):
            continue
        base,ft=pair["baseline_"+category],pair["finetuned_"+category]
        comparisons[category]={}
        for metric in sorted(set(base)&set(ft)):
            bv,fv=base[metric],ft[metric]
            if isinstance(bv,(int,float)) and isinstance(fv,(int,float)):
                comparisons[category][metric]={"baseline":bv,"finetuned":fv,"absolute_delta":fv-bv,
                                               "relative_delta_percent":((fv-bv)/abs(bv)*100 if bv else None)}
    result={"comparisons":comparisons,"inputs":{k:v for k,v in vars(a).items() if k!="output"}}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
