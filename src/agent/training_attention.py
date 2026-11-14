"""Memory-bounded SDPA policies for Qwen training workers.

Fused policy forbids math fallback. The portable query-chunked policy explicitly
uses math only on small blocks and checkpoints them; it never creates full-size
attention scores. These policies do not change inference defaults.
"""
from types import SimpleNamespace
from functools import partial

POLICY = 'explicit-kv-fused-sdpa-v1'
CHUNKED_POLICY = 'query-chunked-math-sdpa-v2'


def disable_native_gqa(attention_mask, key):
    return False


def chunked_sdpa_attention_forward(module, query, key, value, attention_mask,
                                   dropout=0.0, scaling=None, is_causal=None,
                                   query_chunk_size=128, **kwargs):
    """Exact full-context attention in query blocks; checkpoint each block.

    Each query still sees ALL permitted past keys. This is not sliding-window
    attention, sequence truncation, or independent training chunks. Cache decode
    is deliberately unsupported: this function is for self-attention SFT only.
    """
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint
    length = query.shape[-2]
    if length != key.shape[-2] or length != value.shape[-2]:
        raise ValueError('Chunked training attention requires equal Q/K/V lengths; KV cache is unsupported')
    if query_chunk_size <= 0 or query.shape[1] % key.shape[1]:
        raise ValueError('Invalid query chunk size or GQA heads')
    repeats = query.shape[1] // key.shape[1]
    if repeats != 1:
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
    causal = getattr(module, 'is_causal', True) if is_causal is None else is_causal
    pieces = []
    for start in range(0, length, query_chunk_size):
        end = min(start + query_chunk_size, length)
        key_end = end if causal else length

        # Defaults bind this block's indices for checkpoint recomputation.
        def attend(q, k, v, offset=start, stop=end, nkeys=key_end):
            mask = None
            if causal:
                mask = (torch.arange(offset, stop, device=q.device)[:, None]
                        >= torch.arange(nkeys, device=q.device)[None, :])
            if attention_mask is not None:
                if attention_mask.ndim == 2:
                    supplied = attention_mask[:, None, None, :nkeys].bool()
                elif attention_mask.ndim == 4:
                    supplied = attention_mask[..., :nkeys]
                    if supplied.shape[-2] != 1:
                        supplied = supplied[..., offset:stop, :]
                else:
                    raise ValueError('Expected 2D padding or 4D attention mask')
                if mask is None:
                    mask = supplied
                elif supplied.dtype == torch.bool:
                    mask = supplied & mask
                else:
                    mask = supplied.masked_fill(~mask, float('-inf'))
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                       dropout_p=dropout, is_causal=False, scale=scaling)

        tensors = (query[..., start:end, :], key[..., :key_end, :], value[..., :key_end, :])
        if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
            result = checkpoint(attend, *tensors, use_reentrant=False)
        else:
            result = attend(*tensors)
        pieces.append(result)
    return torch.cat(pieces, dim=-2).transpose(1, 2).contiguous(), None


def configure_training_sdpa(torch, integration, mode='fused', query_chunk_size=128, registry=None):
    if not hasattr(integration, 'use_gqa_in_sdpa'):
        raise RuntimeError('Transformers SDPA API changed; use the notebook pinned versions')
    integration.use_gqa_in_sdpa = disable_native_gqa
    cuda = torch.backends.cuda
    if mode not in ('fused', 'chunked-math') or query_chunk_size <= 0:
        raise ValueError('Unsupported attention policy/chunk size')
    bounded = mode == 'chunked-math'
    cuda.enable_flash_sdp(not bounded)
    cuda.enable_mem_efficient_sdp(not bounded)
    cuda.enable_math_sdp(bounded)
    cuda.enable_cudnn_sdp(False)
    if bounded:
        if registry is None:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            registry = ALL_ATTENTION_FUNCTIONS
        forward = partial(chunked_sdpa_attention_forward, query_chunk_size=query_chunk_size)
        registry.register('sdpa', forward)
        integration.sdpa_attention_forward = forward
    integration._lifemate_policy = CHUNKED_POLICY if bounded else POLICY
    return integration._lifemate_policy


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
            'Attention forward/backward preflight failed. Keep this log; '
            'do not silently shorten/drop training samples.'
        ) from exc
    return {'policy': integration._lifemate_policy, 'query_heads': heads, 'kv_heads': kv_heads,
            'head_dim': dim, 'forward_backward_passed': True,
            'math_backend': integration._lifemate_policy == CHUNKED_POLICY}
