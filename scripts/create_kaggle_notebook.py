from __future__ import annotations

import json
from pathlib import Path


CELLS = [
    ("markdown", "# LifeMate: two-GPU Kaggle FP16 training\n\nRun this notebook top to bottom. It clones `frame-agent`, validates the two fixed Kaggle inputs, checks for two Tesla T4 GPUs before downloading Qwen, downloads the exact Hugging Face model, prepares data, runs FP16 LoRA training with FSDP full sharding and validation evaluation, then exports reports and adapter artifacts. Internet must be enabled for GitHub, Hugging Face, and any missing Python packages. If the two-GPU memory gate fails, the workflow stops before model download."),
    ("code", """from pathlib import Path
import json, os, shutil, subprocess, sys, time

RUN_ROOT = Path('/kaggle/working/lifemate_run')
RUN_ROOT.mkdir(parents=True, exist_ok=True)
probe = RUN_ROOT / '.write_test'
probe.write_text('ok')
probe.unlink()
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

import torch
runtime = {'python':sys.version,'torch':torch.__version__,'cuda_version':torch.version.cuda,
           'cuda_available':torch.cuda.is_available(),'gpu':[]}
if runtime['cuda_available']:
    for index in range(torch.cuda.device_count()):
        props=torch.cuda.get_device_properties(index)
        free,total=torch.cuda.mem_get_info(index)
        runtime['gpu'].append({'index':index,'name':props.name,'total_vram_bytes':total,'free_vram_bytes':free})
runtime['nvidia_smi']=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout if shutil.which('nvidia-smi') else None
(RUN_ROOT / 'runtime_initial.json').write_text(json.dumps(runtime, indent=2, default=str))
print(json.dumps(runtime, indent=2, default=str))
if not runtime['cuda_available']:
    raise RuntimeError('CUDA is unavailable. Select Kaggle GPU accelerator and rerun; model download is not started.')
"""),
    ("markdown", "## 1. Clone the required repository and branch"),
    ("code", """REPO_URL = 'https://github.com/tuong1901/zac2026-lifemate.git'
BRANCH = 'frame-agent'
PROJECT = Path('/kaggle/working/lifemate_repo')
if PROJECT.exists():
    origin = subprocess.run(['git','-C',str(PROJECT),'remote','get-url','origin'],capture_output=True,text=True)
    branch = subprocess.run(['git','-C',str(PROJECT),'branch','--show-current'],capture_output=True,text=True)
    if origin.returncode or origin.stdout.strip() != REPO_URL or branch.stdout.strip() != BRANCH:
        raise RuntimeError(f'Existing {PROJECT} is not the requested repository/branch; remove that Kaggle working folder and rerun.')
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only','origin',BRANCH],check=True)
else:
    subprocess.run(['git','clone','--depth','1','--branch',BRANCH,'--single-branch',REPO_URL,str(PROJECT)],check=True)
actual_branch = subprocess.check_output(['git','-C',str(PROJECT),'branch','--show-current'],text=True).strip()
actual_commit = subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip()
if actual_branch != BRANCH: raise RuntimeError(f'Expected {BRANCH}, got {actual_branch}')
sys.path.insert(0, str(PROJECT/'src'))
os.environ['PYTHONPATH'] = str(PROJECT/'src')
print({'project':str(PROJECT),'branch':actual_branch,'commit':actual_commit})
"""),
    ("markdown", "## 2. Inspect and install only project dependencies\nThe notebook does not install or replace PyTorch. If the Kaggle image has an incompatible Torch/CUDA build, it stops instead of silently changing precision or CUDA packages."),
    ("code", """import importlib.metadata as metadata
for package in ('torch','transformers','datasets','accelerate','peft','huggingface_hub','PyYAML','torchao'):
    try: print(package, metadata.version(package))
    except metadata.PackageNotFoundError: print(package, 'not installed')

subprocess.run([sys.executable,'-m','pip','install','-r',str(PROJECT/'requirements/kaggle.txt'),
                '--disable-pip-version-check'],check=True)
from lifemate.utils.runtime_compat import torchao_compat_action
try: torchao_version=metadata.version('torchao')
except metadata.PackageNotFoundError: torchao_version=None
if torchao_compat_action(torchao_version) == 'remove':
    print(f'Removing unused incompatible torchao {torchao_version}; FP16 LoRA does not use TorchAO and Torch/CUDA will not be changed.')
    subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True)
    try: metadata.version('torchao')
    except metadata.PackageNotFoundError: pass
    else: raise RuntimeError('Incompatible torchao remains installed; stopping before model/training process.')
import torch, transformers
def installed_version(package):
    try: return metadata.version(package)
    except metadata.PackageNotFoundError: return None
versions={package:installed_version(package) for package in ('torch','transformers','datasets','accelerate','peft','huggingface-hub','PyYAML','torchao')}
print(versions)
if tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) < (2,6):
    raise RuntimeError(f'Kaggle Torch {torch.__version__} is below Qwen3 recommended 2.6; refusing to replace the CUDA build.')
if tuple(int(x) for x in transformers.__version__.split('.')[:2]) < (4,51):
    raise RuntimeError(f'Transformers {transformers.__version__} lacks supported Qwen3 integration.')
print('Torch/CUDA:',torch.__version__,torch.version.cuda,'Transformers:',transformers.__version__)
"""),
    ("markdown", "## 3. Validate the two fixed Kaggle datasets and preserve the supplied source split"),
    ("code", """SFT_DIR = Path('/kaggle/input/datasets/estherle/sft-data/sft_full_history_v1')
ORIGINAL_DIR = Path('/kaggle/input/datasets/estherle/zalo-dataset/zalo-ai-challenge/train+public_test')
required_sft = ('train_all_lengths.jsonl','validation_all_lengths.jsonl','split_manifest.json','lengths.csv')
required_original = ('train/train.jsonl','train/zaic_tools','public_test/public_test.jsonl')
missing = [str(p/name) for p,names in ((SFT_DIR,required_sft),(ORIGINAL_DIR,required_original)) for name in names if not (p/name).exists()]
if missing: raise FileNotFoundError('Required Kaggle dataset files are missing: '+json.dumps(missing,indent=2))
os.environ.update({'SFT_DATA_DIR':str(SFT_DIR),'ORIGINAL_DATA_DIR':str(ORIGINAL_DIR),
                   'LIFEMATE_WORK_DIR':str(RUN_ROOT)})
print({'sft':str(SFT_DIR),'original':str(ORIGINAL_DIR)})

def run_script(name,*args):
    cmd=[sys.executable,str(PROJECT/'scripts'/name),*map(str,args)]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd,cwd=PROJECT,check=True)

run_script('audit_data.py','--sft-dir',SFT_DIR,'--original-dir',ORIGINAL_DIR,'--output-dir',RUN_ROOT/'audit')
run_script('prepare_data.py','--sft-dir',SFT_DIR,'--original-dir',ORIGINAL_DIR,'--output-dir',RUN_ROOT/'processed')
run_script('verify_toolenv.py','--sft-dir',SFT_DIR,'--original-dir',ORIGINAL_DIR,'--output',RUN_ROOT/'audit/toolenv_verification.json')
prep=json.loads((RUN_ROOT/'processed/split_manifest.json').read_text())
if prep.get('public_test_used_for_training') is not False or prep.get('source_overlap'):
    raise RuntimeError('Data leakage/source split verification failed.')
print(json.dumps({k:v for k,v in prep.items() if k!='source_split'},indent=2))
"""),
    ("markdown", "## 4. Inspect actual hardware and stop before model download unless FP16 fits\nThis run uses two Tesla T4 GPUs with FP16 weights and Trainer autocast under PyTorch FSDP FULL_SHARD. The model parameters, gradients, and optimizer state are sharded across GPUs; activations remain local, so sequence length and gradient checkpointing still matter. The gate requires at least 14 GiB total and 10 GiB free on each GPU. Longer examples are reported as excluded. The first optimizer step checks actual memory, finite loss, and nonzero gradients."),
    ("code", """from lifemate.utils.hardware import inspect_hardware
from lifemate.training.trainer import check_fsdp_hardware
runtime = inspect_hardware(['/kaggle/working', '/kaggle/input'])
(RUN_ROOT/'runtime_training.json').write_text(json.dumps(runtime,indent=2,default=str))
print(json.dumps(runtime,indent=2,default=str))
gpus = check_fsdp_hardware(runtime)  # raises before model download unless two T4s are available
print('FSDP GPUs:',[(g['index'],g['name'],round(g['total_vram_bytes']/1024**3,1),round(g['free_vram_bytes']/1024**3,1)) for g in gpus])
"""),
    ("markdown", "## 5. Download and validate the exact Qwen model from Hugging Face\nThis stage downloads only after the GPU feasibility gate. The resolved model commit SHA is recorded automatically in `model_source.json`; no manual manifest or Kaggle-mounted model is used."),
    ("code", """from lifemate.models.loader import download_qwen_model
MODEL_DIR = RUN_ROOT/'base_model'
model_info = download_qwen_model(MODEL_DIR)
print(json.dumps(model_info,indent=2,default=str))
if model_info['model_id_verified'] != 'Qwen/Qwen3-4B-Instruct-2507': raise RuntimeError('Unexpected model identity.')
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(str(MODEL_DIR),local_files_only=True,trust_remote_code=False)
if not tokenizer.chat_template: raise RuntimeError('Downloaded tokenizer has no chat template.')
print('Model revision:',model_info['commit_hash'],'tokenizer template verified.')
"""),
    ("markdown", "## 6. FP16 model smoke test\nChecks that the loaded weights are FP16 and that a forward pass returns a finite loss."),
    ("code", """run_script('model_smoke.py','--model-dir',MODEL_DIR,'--output',RUN_ROOT/'model_smoke.json')
smoke=json.loads((RUN_ROOT/'model_smoke.json').read_text())
if smoke.get('passed') is not True or smoke.get('precision') != 'float16' or not __import__('math').isfinite(smoke['loss']): raise RuntimeError('FP16 forward/loss smoke failed.')
"""),
    ("markdown", "## 7. FP16 LoRA training with two-GPU FSDP\nThe notebook launches one process per GPU with torchrun. FSDP FULL_SHARD distributes model parameters, gradients, and optimizer state. The first optimizer step checks that LoRA gradients exist, are finite, and are nonzero. Training can resume from the latest checkpoint under the same output directory."),
    ("code", """TRAIN_DIR=RUN_ROOT/'training'
checkpoints=sorted((TRAIN_DIR/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1])) if (TRAIN_DIR/'checkpoints').exists() else []
train_cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(len(gpus)),str(PROJECT/'scripts/train.py'),
           '--model-dir',str(MODEL_DIR),'--train-jsonl',str(RUN_ROOT/'processed/train.jsonl'),
           '--validation-jsonl',str(RUN_ROOT/'processed/validation.jsonl'),'--output-dir',str(TRAIN_DIR)]
smoke_cmd=train_cmd+['--smoke-only']
print('Running two-GPU FSDP loss smoke test before full training:', ' '.join(smoke_cmd))
subprocess.run(smoke_cmd,cwd=PROJECT,check=True)
smoke_result=json.loads((TRAIN_DIR/'fsdp_smoke_result.json').read_text())
if smoke_result.get('passed') is not True or smoke_result.get('optimizer_steps') != 3:
    raise RuntimeError('Three-step two-GPU FSDP real-batch attention smoke test did not pass.')
for rank_result in smoke_result.get('per_rank',[]):
    reports=rank_result.get('attention_steps',[])
    if len(reports)!=3 or any(not row.get('attention') or row['attention'].get('backend_selected')!='pytorch_mem_efficient' for row in reports):
        raise RuntimeError('A rank did not confirm memory-efficient SDPA on all three real training batches.')
print('FSDP smoke passed:',json.dumps(smoke_result,indent=2))
if checkpoints: train_cmd += ['--resume-from-checkpoint',str(checkpoints[-1])]
print('Running two-GPU FP16 FSDP training:', ' '.join(train_cmd))
subprocess.run(train_cmd,cwd=PROJECT,check=True)
adapter_dir=TRAIN_DIR/'adapter'
if not (adapter_dir/'adapter_config.json').is_file(): raise FileNotFoundError('Training ended without a saved LoRA adapter.')
"""),
    ("markdown", "## 8. Baseline and fine-tuned validation evaluation\nBoth evaluations use the original validation split and official ToolEnv trajectory loop. Public test is not used here."),
    ("code", """run_script('evaluate_baseline.py','--model-dir',MODEL_DIR,'--original-dir',ORIGINAL_DIR,'--sft-dir',SFT_DIR,
           '--data',RUN_ROOT/'processed/validation.jsonl','--output-dir',RUN_ROOT/'baseline')
run_script('evaluate_finetuned.py','--model-dir',MODEL_DIR,'--adapter-dir',adapter_dir,'--original-dir',ORIGINAL_DIR,
           '--sft-dir',SFT_DIR,'--data',RUN_ROOT/'processed/validation.jsonl','--output-dir',RUN_ROOT/'finetuned')
run_script('compare_models.py','--baseline-steps',RUN_ROOT/'baseline/steps/metrics.json',
           '--finetuned-steps',RUN_ROOT/'finetuned/steps/metrics.json',
           '--baseline-trajectories',RUN_ROOT/'baseline/trajectories/trajectory_evaluation.json',
           '--finetuned-trajectories',RUN_ROOT/'finetuned/trajectories/trajectory_evaluation.json',
           '--output',RUN_ROOT/'evaluation/model_comparison.json')
"""),
    ("markdown", "## 9. Save and locate artifacts\nThe export excludes the downloaded 16+ GiB base model and the input datasets. The adapter must be paired with the recorded base-model revision."),
    ("code", """EXPORT_DIR=Path('/kaggle/working/lifemate_export_'+time.strftime('%Y%m%d_%H%M%S'))
EXPORT_DIR.mkdir(parents=True,exist_ok=True)
for source in ('audit','training','baseline','finetuned','evaluation'):
    path=RUN_ROOT/source
    if path.exists(): shutil.copytree(path,EXPORT_DIR/source,dirs_exist_ok=True)
for source in ('model_smoke.json','runtime_initial.json','runtime_training.json','training/fsdp_smoke_result.json'):
    path=RUN_ROOT/source
    if path.exists(): shutil.copy2(path,EXPORT_DIR/source)
shutil.copy2(PROJECT/'reports/REPRODUCIBILITY.md',EXPORT_DIR/'REPRODUCIBILITY.md')
archive=shutil.make_archive(str(EXPORT_DIR),'zip',root_dir=EXPORT_DIR)
print('Run root:',RUN_ROOT)
print('Checkpoint directory:',TRAIN_DIR/'checkpoints')
print('LoRA adapter:',adapter_dir)
print('Logs/metrics:')
for p in sorted(TRAIN_DIR.glob('*.json*')): print(' -',p)
print('Evaluation reports:',RUN_ROOT/'baseline',RUN_ROOT/'finetuned',RUN_ROOT/'evaluation')
print('Download this file from Kaggle Output:',archive)
"""),
]


def cell(kind: str, source: str) -> dict:
    base = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


def main():
    repo = Path(__file__).resolve().parents[1]
    notebook = {"cells": [cell(*c) for c in CELLS], "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10+"},
        "kaggle": {"accelerator": "GPU", "internet": True}}, "nbformat": 4, "nbformat_minor": 5}
    out = repo / "notebooks/kaggle_train_eval.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
