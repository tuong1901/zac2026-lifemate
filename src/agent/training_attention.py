"""Memory-bounded SDPA policy for Qwen GQA on Turing/T4 training workers.

Native SDPA GQA can dispatch to the quadratic math backend on T4. Expand
KV heads explicitly through Transformers instead, and prohibit math fallback.
This policy is confined to the training subprocess, never inference defaults.
"""
from types import SimpleNamespace

POLICY = 'explicit-kv-fused-sdpa-v1'


def disable_native_gqa(attention_mask, key):
    return False


def configure_training_sdpa(torch, integration):
    if not hasattr(integration, 'use_gqa_in_sdpa'):
        raise RuntimeError('Transformers SDPA API changed; use the notebook pinned versions')
    integration.use_gqa_in_sdpa = disable_native_gqa
    cuda = torch.backends.cuda
    cuda.enable_flash_sdp(True)
    cuda.enable_mem_efficient_sdp(True)
    cuda.enable_math_sdp(False)
    cuda.enable_cudnn_sdp(False)
    return POLICY


def probe_training_sdpa(torch, integration, config, device, dtype):
    """Exercise real fused attention forward/backward before loading 4B weights."""
    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    dim = getattr(config, 'head_dim', config.hidden_size // heads)
    kwargs = dict(device=device, dtype=dtype, requires_grad=True)
    q = torch.randn(1, heads, 256, dim, **kwargs)
    k = torch.randn(1, kv_heads, 256, dim, **kwargs)
    v = torch.randn(1, kv_heads, 256, dim, **kwargs)
    module = SimpleNamespace(num_key_value_groups=heads // kv_heads, is_causal=True)
    try:
        output, _ = integration.sdpa_attention_forward(
            module, q, k, v, attention_mask=None, dropout=0.0, scaling=dim**-0.5)
        output.float().square().mean().backward()
        if not all(t.grad is not None and torch.isfinite(t.grad).all().item() for t in (q, k, v)):
            raise RuntimeError('Nonfinite/missing attention gradients')
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        raise RuntimeError(
            'Fused SDPA preflight failed on this GPU/PyTorch build. Math fallback '
            'is disabled to prevent the 12+ GiB attention allocation. Keep this '
            'log; do not enable math or silently shorten/drop training samples.'
        ) from exc
    return {'policy': POLICY, 'query_heads': heads, 'kv_heads': kv_heads,
            'head_dim': dim, 'forward_backward_passed': True, 'math_backend': False}
