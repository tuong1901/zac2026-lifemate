"""Qwen3-4B raw-history v2 QLoRA, single GPU or torchrun DDP.

Fail on overlength data; mask all prompt tokens; retain all source decisions.
The long-context GPU smoke test is required before starting the full run.
"""
import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent.sft_encoding import encode_decision, check_split_rows
from src.agent.training_attention import configure_training_sdpa, probe_training_sdpa, native_training_dtype
from src.agent.training_safety import UpdateHealth, make_update_callback


def read_rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model', default='Qwen/Qwen3-4B-Instruct-2507')
    ap.add_argument('--max-length', type=int, default=12288)
    ap.add_argument('--epochs', type=float, default=2)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--save-steps', type=int, default=50)
    ap.add_argument('--attention-mode', choices=['chunked-math', 'fused'], default='chunked-math')
    ap.add_argument('--query-chunk-size', type=int, default=128)
    ap.add_argument('--max-smoke-seconds-per-microbatch', type=float, default=60.0)
    ap.add_argument('--fp16-init-scale', type=float, default=8.0)
    ap.add_argument('--resume', help='Full checkpoint-N directory, or auto')
    ap.add_argument('--smoke', action='store_true', help='Two updates on longest samples, plus validation')
    ap.add_argument('--smoke-receipt', type=Path, help='Receipt from matching GPU smoke run')
    args = ap.parse_args()
    if not 0 < args.max_length <= 16384 or min(args.epochs, args.lr, args.rank, args.grad_accum, args.save_steps, args.query_chunk_size, args.max_smoke_seconds_per_microbatch, args.fp16_init_scale) <= 0:
        ap.error('Invalid training settings')

    import torch
    from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                              Trainer, TrainingArguments, TrainerCallback, set_seed)
    from transformers.integrations import sdpa_attention
    from transformers.trainer_utils import get_last_checkpoint
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3

    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if not torch.cuda.is_available():
        raise RuntimeError('Enable a CUDA GPU; this script does not train on CPU')
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl', device_id=torch.device('cuda', local_rank))
    set_seed(42)
    policy = configure_training_sdpa(torch, sdpa_attention, mode=args.attention_mode,
                                     query_chunk_size=args.query_chunk_size)
    compute_dtype = native_training_dtype(torch, local_rank)
    capability = torch.cuda.get_device_capability(local_rank)
    print(f'[rank {local_rank}] GPU={torch.cuda.get_device_name(local_rank)} '
          f'capability={capability} compute_dtype={compute_dtype}', flush=True)
    model_config = AutoConfig.from_pretrained(args.model)
    if model_config.model_type != 'qwen3':
        raise ValueError('This long-context attention recipe is for Qwen3 only')
    attention_preflight = probe_training_sdpa(
        torch, sdpa_attention, model_config, torch.device('cuda', local_rank), compute_dtype)
    print(f'[rank {local_rank}] attention preflight: {attention_preflight}', flush=True)
    train_path, val_path = args.data_dir / 'train.jsonl', args.data_dir / 'validation.jsonl'
    rows, val_rows = read_rows(train_path), read_rows(val_path)
    check_split_rows(rows, val_rows)
    fingerprint = {
        'model': args.model, 'max_length': args.max_length, 'rank': args.rank,
        'world_size': world_size, 'grad_accum': args.grad_accum,
        'learning_rate': args.lr, 'epochs': args.epochs, 'seed': 42,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'encoding_sha256': hashlib.sha256((Path(__file__).resolve().parents[1] / 'src/agent/sft_encoding.py').read_bytes()).hexdigest(),
        'attention_policy': policy,
        'query_chunk_size': args.query_chunk_size,
        'attention_sha256': hashlib.sha256((Path(__file__).resolve().parents[1] / 'src/agent/training_attention.py').read_bytes()).hexdigest(),
        'torch_version': torch.__version__,
        'compute_dtype': str(compute_dtype), 'gpu_capability': list(capability),
        'fp16_init_scale': args.fp16_init_scale,
        'safety_sha256': hashlib.sha256((Path(__file__).resolve().parents[1] / 'src/agent/training_safety.py').read_bytes()).hexdigest(),
        'train_sha256': hashlib.sha256(train_path.read_bytes()).hexdigest(),
        'validation_sha256': hashlib.sha256(val_path.read_bytes()).hexdigest(),
        'gpu': torch.cuda.get_device_name(local_rank),
    }
    if not args.smoke:
        if not args.smoke_receipt or not args.smoke_receipt.is_file():
            raise ValueError('Run --smoke first, then provide its smoke_passed.json via --smoke-receipt')
        receipt = json.loads(args.smoke_receipt.read_text())
        if receipt.get('fingerprint') != fingerprint:
            raise ValueError('Smoke settings/data/GPU do not match this run')
        if not receipt.get('ready_for_full_training', False):
            raise ValueError('Smoke did not pass the speed gate; do not start an impractically slow full run')
    conflict = args.out.exists() and any(args.out.iterdir()) and not args.resume
    if world_size > 1:
        # Decide before any rank creates files in the shared output directory.
        torch.distributed.barrier(device_ids=[local_rank])
    if conflict:
        raise ValueError('Output is not empty; choose a new --out or --resume auto')
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError('A fast tokenizer with offsets is required')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    encoded = [encode_decision(r, tokenizer, args.max_length) for r in rows]
    encoded_val = [encode_decision(r, tokenizer, args.max_length) for r in val_rows]
    lengths = [len(r['input_ids']) for r in encoded + encoded_val]
    supervised = sum(sum(t != -100 for t in r['labels']) for r in encoded)
    if args.smoke:
        # Every rank exercises the longest training sequence, not an easy short row.
        longest = max(encoded, key=lambda r: len(r['input_ids']))
        # Accumulation changes effective batch, not the longest-sequence activation
        # shape. Two updates with accumulation=1 test optimizer states and avoid
        # repeating 16 expensive backward passes per GPU just for a smoke test.
        encoded = [longest] * (2 * world_size)
        encoded_val = [max(encoded_val, key=lambda r: len(r['input_ids']))] * world_size

    # Qwen3 fused linear cross entropy avoids materializing full [12K, vocabulary] logits.
    # Apply before PEFT; every DDP rank holds its own complete 4-bit replica.
    # Keep the memory-saving fused loss; use native Qwen RoPE/RMSNorm/MLP
    # kernels to reduce the FP16/T4 numerical surface area.
    apply_liger_kernel_to_qwen3(rope=False, rms_norm=False, swiglu=False,
                               fused_linear_cross_entropy=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={'': local_rank}, dtype=compute_dtype,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype), attn_implementation='sdpa')
    if model.config.model_type != 'qwen3':
        raise ValueError('This recipe and Liger patch are tested structurally for Qwen3 only')
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2*args.rank,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.05, bias='none', task_type='CAUSAL_LM'))

    def collate(batch):
        size = max(len(r['input_ids']) for r in batch)
        result = {key: torch.tensor([r[key] + [pad] * (size-len(r[key])) for r in batch], dtype=torch.long)
                  for key, pad in [('input_ids', tokenizer.pad_token_id), ('attention_mask', 0), ('labels', -100)]}
        # Liger otherwise materializes full-vocabulary logits in eval mode.
        result['skip_logits'] = True
        return result

    training_args = TrainingArguments(
        output_dir=str(args.out), num_train_epochs=args.epochs,
        max_steps=2 if args.smoke else -1, learning_rate=args.lr,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if args.smoke else args.grad_accum,
        fp16=compute_dtype==torch.float16, bf16=compute_dtype==torch.bfloat16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        ddp_find_unused_parameters=False, ddp_timeout=3600,
        eval_strategy='no' if args.smoke else 'epoch', prediction_loss_only=True,
        save_strategy='no' if args.smoke else 'steps', save_steps=args.save_steps,
        save_total_limit=3, save_safetensors=True, logging_steps=1 if args.smoke else 5,
        logging_nan_inf_filter=False,
        lr_scheduler_type='cosine', warmup_ratio=0.0 if args.smoke else 0.05, optim='adamw_torch',
        max_grad_norm=1.0, weight_decay=0.01, report_to='none',
        remove_unused_columns=False, label_names=['labels'], seed=42, data_seed=42,
        dataloader_num_workers=0, dataloader_pin_memory=True,
        average_tokens_across_devices=True)
    health = UpdateHealth()
    trainer = Trainer(model=model, args=training_args, train_dataset=encoded,
                      eval_dataset=encoded_val, data_collator=collate,
                      processing_class=tokenizer,
                      callbacks=[make_update_callback(TrainerCallback, health)])
    if compute_dtype == torch.float16:
        if trainer.accelerator.scaler is None:
            raise RuntimeError('FP16 training requires an active GradScaler')
        # Before optimizer preparation: AcceleratedOptimizer will capture this scaler.
        trainer.accelerator.scaler = torch.amp.GradScaler(
            'cuda', init_scale=args.fp16_init_scale, growth_interval=2000)
        print(f'[rank {local_rank}] FP16 GradScaler init_scale={args.fp16_init_scale}', flush=True)
    probe_name, probe_parameter = next((n, p) for n, p in model.named_parameters()
                                      if p.requires_grad and 'lora_B' in n)
    before_update = probe_parameter.detach().clone() if args.smoke else None
    resume = get_last_checkpoint(str(args.out)) if args.resume == 'auto' else args.resume
    if args.resume == 'auto' and resume is None:
        raise ValueError('No full checkpoint available for resume')
    if resume:
        config_path = args.out / 'run_config.json'
        if not config_path.is_file() or json.loads(config_path.read_text()).get('fingerprint') != fingerprint:
            raise ValueError('Resume requires matching run_config.json and dataset/settings')
    if trainer.is_world_process_zero():
        config = {'fingerprint': fingerprint, 'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  'train_decisions': len(rows), 'validation_decisions': len(val_rows),
                  'max_actual_tokens': max(lengths), 'supervised_train_tokens': supervised,
                  'attention_preflight': attention_preflight,
                  'actual_gradient_accumulation': 1 if args.smoke else args.grad_accum,
                  'effective_batch': world_size * (1 if args.smoke else args.grad_accum),
                  'planned_full_effective_batch': world_size * args.grad_accum, 'smoke': args.smoke}
        (args.out / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        print(json.dumps(config, indent=2), flush=True)
        model.print_trainable_parameters()
    try:
        result = trainer.train(resume_from_checkpoint=resume)
    except FloatingPointError as exc:
        if trainer.is_world_process_zero():
            (args.out / 'training_failure.json').write_text(
                json.dumps({'error': str(exc), 'health': health.summary()}, indent=2), encoding='utf-8')
        raise
    metrics = trainer.evaluate()
    if trainer.is_world_process_zero():
        metrics.update(result.metrics)
        metrics.update(health.summary())
        if 'eval_loss' not in metrics or not math.isfinite(metrics['eval_loss']):
            raise RuntimeError('Validation loss is missing/nonfinite; do not mark this run successful')
        if 'train_loss' not in metrics or not math.isfinite(metrics['train_loss']) or metrics['train_loss'] <= 0:
            raise RuntimeError('Training loss is invalid; do not mark this run successful')
        metrics['gpu_peak_allocated_gib_rank0'] = torch.cuda.max_memory_allocated(local_rank)/2**30
        (args.out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        if args.smoke:
            seconds_per_microbatch = result.metrics['train_runtime'] / 2
            delta = (probe_parameter.detach() - before_update).abs().max().item()
            numerically_healthy = health.smoke_is_healthy(delta)
            ready = numerically_healthy and seconds_per_microbatch <= args.max_smoke_seconds_per_microbatch
            smoke_report = {'fingerprint': fingerprint, 'metrics': metrics,
                            'seconds_per_microbatch_per_rank': seconds_per_microbatch,
                            'ready_for_full_training': ready,
                            'numerically_healthy': numerically_healthy,
                            'probe_parameter': probe_name, 'probe_parameter_max_delta': delta,
                            'smoke_actual_gradient_accumulation': 1}
            (args.out / 'smoke_passed.json').write_text(json.dumps(smoke_report, indent=2), encoding='utf-8')
            print('Smoke performance:', json.dumps(smoke_report), flush=True)
            if not ready:
                print('Smoke failed numerical or speed checks; full training is blocked.', flush=True)
        else:
            final = args.out / 'final_adapter'
            trainer.save_model(str(final))
            tokenizer.save_pretrained(final)
            trainer.save_state()
            if not (final / 'adapter_config.json').is_file() or not (final / 'adapter_model.safetensors').is_file():
                raise RuntimeError('Adapter export incomplete')
            print('Final adapter:', final, flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier(device_ids=[local_rank])
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    finally:
        torch_module = sys.modules.get('torch')
        if torch_module is not None and torch_module.distributed.is_initialized():
            torch_module.distributed.destroy_process_group()
