from __future__ import annotations

import argparse
import json
from pathlib import Path

from lifemate.models.loader import inspect_local_model, load_local_qwen
from lifemate.utils.hardware import inspect_hardware


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir",required=True)
    p.add_argument("--output",default="artifacts/model_smoke.json")
    p.add_argument("--no-quantization",action="store_true")
    a=p.parse_args()
    metadata=inspect_local_model(a.model_dir)
    hardware=inspect_hardware([a.model_dir])
    model,tokenizer,_=load_local_qwen(a.model_dir,quantized=not a.no_quantization)
    encoded=tokenizer.apply_chat_template([{"role":"user","content":"Return the JSON object {\"ok\":true}."}],tokenize=True,add_generation_prompt=True,return_tensors="pt",return_dict=True)
    device=model.get_input_embeddings().weight.device
    encoded={k:v.to(device) for k,v in encoded.items()}
    import torch
    with torch.inference_mode():
        logits=model(**encoded).logits
        output=model.generate(**encoded,max_new_tokens=32,do_sample=False,pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    result={"passed":True,"model":metadata,"hardware":hardware,"forward_shape":list(logits.shape),
            "generated":tokenizer.decode(output[0,encoded["input_ids"].shape[-1]:],skip_special_tokens=True)}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps({"passed":True,"forward_shape":result["forward_shape"],"generated":result["generated"]},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
