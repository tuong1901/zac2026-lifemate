"""Turing-safe SDPA selection and per-rank runtime telemetry."""
from __future__ import annotations

import json
import inspect
import os

_ORIGINAL_SDPA = None
_ORIGINAL_USE_GQA = None
_CURRENT_STEP = None
_CURRENT_STEP_REPORTED = False
_STEP_ATTENTION_REPORTS = []
_STRICT_EFFICIENT = False


def _attention_backend(profiler) -> str:
    keys = {event.key.lower() for event in profiler.key_averages()}
    if any("_scaled_dot_product_efficient_attention" in key or "_efficient_attention_forward" in key for key in keys):
        return "pytorch_mem_efficient"
    if any("_scaled_dot_product_flash_attention" in key for key in keys):
        return "pytorch_flash"
    if any("_scaled_dot_product_cudnn_attention" in key or "_scaled_dot_product_cudnn" in key for key in keys):
        return "pytorch_cudnn"
    if any("_scaled_dot_product_attention_math" in key for key in keys):
        return "pytorch_math"
    return "unknown"


def _make_turing_safe_gqa(original, torch):
    """Wrap the installed Transformers helper without assuming its arity."""
    signature = inspect.signature(original)

    def turing_safe_use_gqa(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        key = bound.arguments.get("key")
        if key is None or not torch.is_tensor(key):
            raise TypeError(f"Cannot identify the key tensor using installed signature {signature}")
        if key.device.type == "cuda" and torch.cuda.get_device_capability(key.device)[0] < 8:
            return False
        return original(*args, **kwargs)

    turing_safe_use_gqa.__name__ = getattr(original, "__name__", "use_gqa_in_sdpa")
    turing_safe_use_gqa.__doc__ = getattr(original, "__doc__", None)
    turing_safe_use_gqa.__signature__ = signature
    return turing_safe_use_gqa


def configure_t4_sdpa(torch) -> dict:
    """Force the probed memory-efficient PyTorch SDPA path on pre-SM80 GPUs.

    On T4, current Transformers may enable GQA directly in SDPA. PyTorch's
    efficient backend cannot consume that unequal-head form on SM75, while
    Flash/cuDNN attention are unavailable. Disable that GQA shortcut so
    Transformers explicitly repeats K/V heads, then prohibit the quadratic
    math fallback. A smoke step must prove the efficient kernel really runs.
    """
    global _ORIGINAL_SDPA, _ORIGINAL_USE_GQA, _STRICT_EFFICIENT
    if not torch.cuda.is_available():
        raise RuntimeError("SDPA backend selection requires CUDA")
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    if capability[0] >= 8:
        return {"device": device, "compute_capability": list(capability), "policy": "pytorch_auto"}

    import torch.nn.functional as F
    import transformers.integrations.sdpa_attention as sdpa_integration

    if _ORIGINAL_SDPA is None:
        _ORIGINAL_SDPA = F.scaled_dot_product_attention
        _ORIGINAL_USE_GQA = getattr(sdpa_integration, "use_gqa_in_sdpa", None)

        def profiled_sdpa(query, key, value, *args, **kwargs):
            global _CURRENT_STEP_REPORTED
            if _CURRENT_STEP_REPORTED:
                return _ORIGINAL_SDPA(query, key, value, *args, **kwargs)

            from torch.profiler import ProfilerActivity, profile
            report = {
                "rank": int(os.environ.get("RANK", "0")),
                "step": _CURRENT_STEP,
                "backend_policy": "efficient_only_on_sm_lt_80",
                "query_shape": list(query.shape),
                "key_shape_after_transformers_gqa_handling": list(key.shape),
                "value_shape_after_transformers_gqa_handling": list(value.shape),
                "attention_matrix_shape_if_materialized": [query.shape[0], query.shape[1], query.shape[2], key.shape[2]],
                "attention_mask_shape": list(kwargs["attn_mask"].shape) if kwargs.get("attn_mask") is not None else None,
                "is_causal": kwargs.get("is_causal", False),
                "enable_gqa": kwargs.get("enable_gqa", False),
                "allocated_gib_before_sdpa": round(torch.cuda.memory_allocated(query.device) / 1024**3, 3),
            }
            try:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    output = _ORIGINAL_SDPA(query, key, value, *args, **kwargs)
                torch.cuda.synchronize(query.device)
                report["backend_selected"] = _attention_backend(prof)
                report["allocated_gib_after_sdpa"] = round(torch.cuda.memory_allocated(query.device) / 1024**3, 3)
                report["peak_allocated_gib_through_sdpa"] = round(torch.cuda.max_memory_allocated(query.device) / 1024**3, 3)
                if _STRICT_EFFICIENT and report["backend_selected"] != "pytorch_mem_efficient":
                    raise RuntimeError(f"T4 must use memory-efficient SDPA, profiler observed {report['backend_selected']}: {report}")
                _STEP_ATTENTION_REPORTS.append(report)
                print("LIFEMATE_SDPA_DIAGNOSTIC " + json.dumps(report), flush=True)
                _CURRENT_STEP_REPORTED = True
                return output
            except torch.cuda.OutOfMemoryError as exc:
                report["error"] = repr(exc)
                print("LIFEMATE_SDPA_OOM " + json.dumps(report), flush=True)
                raise
            except Exception as exc:
                report["error"] = repr(exc)
                print("LIFEMATE_SDPA_ERROR " + json.dumps(report), flush=True)
                raise

        F.scaled_dot_product_attention = profiled_sdpa

        if _ORIGINAL_USE_GQA is not None:
            sdpa_integration.use_gqa_in_sdpa = _make_turing_safe_gqa(_ORIGINAL_USE_GQA, torch)

    # Do not permit an unnoticed quadratic math fallback on T4. The real-batch
    # smoke below is the compatibility check; if efficient attention is not
    # eligible for the actual Q/K/V/mask shape, it fails before full training.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    disable_cudnn = getattr(torch.backends.cuda, "enable_cudnn_sdp", None)
    if disable_cudnn is not None:
        disable_cudnn(False)
    _STRICT_EFFICIENT = True
    return {"device": device, "compute_capability": list(capability),
            "policy": "pytorch_mem_efficient_only", "gqa": "explicit_kv_head_repeat"}


def begin_attention_step(step: int) -> None:
    global _CURRENT_STEP, _CURRENT_STEP_REPORTED
    _CURRENT_STEP = int(step)
    _CURRENT_STEP_REPORTED = False


def latest_attention_report() -> dict | None:
    return _STEP_ATTENTION_REPORTS[-1] if _STEP_ATTENTION_REPORTS else None

