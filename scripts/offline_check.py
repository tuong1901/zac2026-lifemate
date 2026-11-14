from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

from lifemate.models.loader import inspect_local_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir",required=True)
    p.add_argument("--adapter-dir")
    p.add_argument("--asr-model-dir")
    p.add_argument("--load-model",action="store_true",help="also load with outbound socket calls disabled (requires compatible GPU/runtime)")
    p.add_argument("--output",default="artifacts/offline_check.json")
    a=p.parse_args()
    os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1";os.environ["HF_DATASETS_OFFLINE"]="1"
    model=inspect_local_model(a.model_dir)
    for ext in ("*.safetensors","pytorch_model*.bin"):
        if not list(Path(a.model_dir).glob(ext)) and not list(Path(a.model_dir).glob("*/"+ext)):
            raise FileNotFoundError(f"base-model weight files absent: {ext}")
    adapter_info=None
    if a.adapter_dir:
        adapter=Path(a.adapter_dir)
        cfg=json.loads((adapter/"adapter_config.json").read_text(encoding="utf-8"))
        if not any(adapter.glob("adapter_model.*")):
            raise FileNotFoundError("LoRA adapter weights missing")
        adapter_info={"base_model_name_or_path":cfg.get("base_model_name_or_path"),"config":cfg}
    asr_exists=bool(a.asr_model_dir and Path(a.asr_model_dir).is_dir())
    load_result="not_requested"
    if a.load_model:
        def blocked(*_args,**_kwargs): raise RuntimeError("offline check blocked an outbound socket connection")
        socket.create_connection=blocked
        original=socket.socket.connect
        socket.socket.connect=blocked
        try:
            from lifemate.models.loader import load_local_qwen
            loaded,tokenizer,_=load_local_qwen(a.model_dir,precision="float16",model_parallel=True)
            if a.adapter_dir:
                from peft import PeftModel
                loaded=PeftModel.from_pretrained(loaded,a.adapter_dir,is_trainable=False)
            load_result="loaded_local_only"
        finally:
            socket.socket.connect=original
    result={"base_model":model,"adapter":adapter_info,"asr_assets_present":asr_exists,
            "load_result":load_result,"offline_environment":{"HF_HUB_OFFLINE":os.environ["HF_HUB_OFFLINE"],"TRANSFORMERS_OFFLINE":os.environ["TRANSFORMERS_OFFLINE"]}}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps({"model_id":model["model_id_verified"],"load_result":load_result,"asr_assets_present":asr_exists},indent=2))


if __name__=="__main__":main()
