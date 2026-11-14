"""Memory-bounded causal LM loss for decoder models with a linear LM head."""
from __future__ import annotations

from types import MethodType

_DIAGNOSTIC_REPORTED = False
LAST_LOSS_DIAGNOSTIC = None
_NUMERICAL_STEP = None
_NUMERICAL_REPORT = None


def begin_numerical_step(step: int) -> None:
    global _NUMERICAL_STEP, _NUMERICAL_REPORT
    _NUMERICAL_STEP = int(step)
    _NUMERICAL_REPORT = None


def latest_numerical_report():
    return _NUMERICAL_REPORT


def _tensor_numerics(tensor):
    import torch

    detached = tensor.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite]
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "finite": int(finite.sum().item()),
        "elements": detached.numel(),
        "nan": int(torch.isnan(detached).sum().item()),
        "inf": int(torch.isinf(detached).sum().item()),
        "finite_abs_max": float(finite_values.abs().max().item()) if finite_values.numel() else None,
        "finite_abs_mean": float(finite_values.abs().float().mean().item()) if finite_values.numel() else None,
    }


def chunked_causal_lm_loss(hidden_states, lm_head, labels, vocab_size: int,
                           chunk_size: int = 64, num_items_in_batch=None,
                           ignore_index: int = -100, input_token_lengths=None):
    """Compute exact shifted CE while keeping only one sequence chunk of logits live.

    Each chunk's projection and cross entropy are activation-checkpointed, so its
    `[batch, chunk, vocab]` FP32 logits are recomputed during backward instead of
    retained for every sequence position. The denominator matches Transformers'
    ForCausalLMLoss semantics, including Trainer's optional item count.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    sequence_length = hidden_states.shape[1]
    if labels.shape[-1] != sequence_length:
        raise ValueError(f"Shifted labels length {labels.shape[-1]} != hidden length {sequence_length}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    first_logits_shape = None
    numerical_report = None
    numerical_chunk_stats = []
    if _NUMERICAL_STEP is not None:
        numerical_report = {
            "step": _NUMERICAL_STEP,
            "rank": int(__import__("os").environ.get("RANK", "0")),
            "hidden_states": _tensor_numerics(hidden_states),
            "input_token_lengths": input_token_lengths,
            "logits": {"chunks": 0, "elements": 0, "nonfinite": 0, "nan": 0, "inf": 0,
                       "finite_abs_max": 0.0, "first_nonfinite_chunk": None,
                       "chunk_shape": None},
        }

    total = None
    for start in range(0, sequence_length, chunk_size):
        stop = min(start + chunk_size, sequence_length)
        hidden_chunk = hidden_states[:, start:stop, :]
        label_chunk = labels[:, start:stop]

        chunk_index = start // chunk_size

        def chunk_sum(hidden, targets):
            nonlocal first_logits_shape
            logits = lm_head(hidden).float()
            if first_logits_shape is None:
                first_logits_shape = tuple(logits.shape)
            chunk_loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1),
                                         ignore_index=ignore_index, reduction="sum")
            if numerical_report is not None and chunk_index == len(numerical_chunk_stats):
                detached_logits = logits.detach()
                finite_mask = torch.isfinite(detached_logits)
                chunk_stats = torch.stack((
                    finite_mask.sum().to(torch.float64), torch.isnan(detached_logits).sum().to(torch.float64),
                    torch.isinf(detached_logits).sum().to(torch.float64),
                    torch.where(finite_mask, detached_logits.abs(), 0).amax().to(torch.float64),
                    torch.isfinite(chunk_loss.detach()).to(torch.float64),
                    torch.tensor(detached_logits.numel(), device=detached_logits.device, dtype=torch.float64),
                )).detach()
                numerical_chunk_stats.append(chunk_stats)
            return chunk_loss

        part = checkpoint(chunk_sum, hidden_chunk, label_chunk, use_reentrant=False)
        total = part if total is None else total + part

    if total is None:
        raise ValueError("Cannot compute causal loss for an empty sequence")
    if num_items_in_batch is None:
        denominator = labels.ne(ignore_index).sum().clamp_min(1).to(device=total.device, dtype=total.dtype)
    else:
        denominator = num_items_in_batch
        if torch.is_tensor(denominator):
            denominator = denominator.to(device=total.device, dtype=total.dtype)
        else:
            denominator = total.new_tensor(denominator)
        denominator = denominator.clamp_min(1)
    loss = total / denominator
    if numerical_report is not None:
        chunk_rows = torch.stack(numerical_chunk_stats).cpu().tolist() if numerical_chunk_stats else []
        logit_report = numerical_report["logits"]
        logit_report.update({"chunks": len(chunk_rows), "elements": 0,
                             "nonfinite": 0, "nan": 0, "inf": 0,
                             "finite_abs_max": 0.0, "first_nonfinite_chunk": None,
                             "chunk_shape": list(first_logits_shape or ())})
        for index, (finite_count, nan_count, inf_count, abs_max, loss_finite, chunk_elements) in enumerate(chunk_rows):
            chunk_elements = int(chunk_elements)
            logit_report["elements"] += chunk_elements
            nonfinite_count = chunk_elements - int(finite_count)
            logit_report["nonfinite"] += nonfinite_count
            logit_report["nan"] += int(nan_count)
            logit_report["inf"] += int(inf_count)
            logit_report["finite_abs_max"] = max(logit_report["finite_abs_max"], float(abs_max))
            if nonfinite_count and logit_report["first_nonfinite_chunk"] is None:
                logit_report["first_nonfinite_chunk"] = {"index": index, "start": index * chunk_size,
                                                          "nonfinite": nonfinite_count,
                                                          "nan": int(nan_count), "inf": int(inf_count)}
            if not bool(loss_finite):
                logit_report["nonfinite_chunk_loss"] = index
        numerical_report["loss"] = {"value": float(loss.detach().item()),
                                    "finite": bool(torch.isfinite(loss.detach()).item()),
                                    "valid_targets": int(labels.ne(ignore_index).sum().item()),
                                    "denominator": float(denominator.detach().item())}
        global _NUMERICAL_REPORT
        _NUMERICAL_REPORT = numerical_report
        print("LIFEMATE_NUMERICS_FORWARD " + __import__("json").dumps(numerical_report), flush=True)
    global _DIAGNOSTIC_REPORTED, LAST_LOSS_DIAGNOSTIC
    if not _DIAGNOSTIC_REPORTED and hidden_states.is_cuda:
        rank = int(__import__("os").environ.get("RANK", "0"))
        device = hidden_states.device
        diagnostics = {
            "rank": rank,
            "input_hidden_shape": list(hidden_states.shape),
            "unpadded_input_token_lengths": input_token_lengths,
            "full_logits_shape_avoided": [*hidden_states.shape[:2], vocab_size],
            "actual_logits_chunk_shape": list(first_logits_shape or ()),
            "chunk_size": chunk_size,
            "valid_shifted_labels": int(labels.ne(ignore_index).sum().item()),
            "loss": float(loss.detach().item()),
            "loss_finite": bool(torch.isfinite(loss.detach()).item()),
            "allocated_gib_after_loss_forward": round(torch.cuda.memory_allocated(device) / 1024**3, 3),
            "reserved_gib_after_loss_forward": round(torch.cuda.memory_reserved(device) / 1024**3, 3),
        }
        LAST_LOSS_DIAGNOSTIC = diagnostics
        print("LIFEMATE_CHUNKED_LOSS_DIAGNOSTIC " + __import__("json").dumps(diagnostics), flush=True)
        _DIAGNOSTIC_REPORTED = True
    return loss


def latest_loss_diagnostic():
    return LAST_LOSS_DIAGNOSTIC


def install_qwen3_chunked_loss(peft_model, chunk_size: int = 64) -> None:
    """Patch the wrapped Qwen3 CausalLM forward to avoid returning full logits in SFT."""
    import torch
    from transformers.modeling_outputs import CausalLMOutputWithPast

    base_model = peft_model.get_base_model()
    if base_model.__class__.__name__ != "Qwen3ForCausalLM":
        raise TypeError(f"Chunked loss currently supports Qwen3ForCausalLM, got {base_model.__class__.__name__}")
    original_forward = base_model.forward

    def forward_with_chunked_loss(self, input_ids=None, attention_mask=None, position_ids=None,
                                  past_key_values=None, inputs_embeds=None, labels=None,
                                  use_cache=None, logits_to_keep=0, **kwargs):
        # Preserve ordinary forward semantics for callers that are not training on labels.
        if labels is None:
            return original_forward(input_ids=input_ids, attention_mask=attention_mask,
                                    position_ids=position_ids, past_key_values=past_key_values,
                                    inputs_embeds=inputs_embeds, labels=None, use_cache=use_cache,
                                    logits_to_keep=logits_to_keep, **kwargs)
        model_kwargs = dict(kwargs)
        num_items_in_batch = model_kwargs.pop("num_items_in_batch", None)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, use_cache=use_cache,
                             **model_kwargs)
        hidden_states = outputs.last_hidden_state
        loss = chunked_causal_lm_loss(
            hidden_states, self.lm_head, labels, self.config.vocab_size,
            chunk_size=chunk_size, num_items_in_batch=num_items_in_batch,
            input_token_lengths=(attention_mask.sum(dim=-1).detach().cpu().tolist()
                                 if attention_mask is not None else None),
        )
        # `logits=None` prevents Accelerate's convert_to_fp32 wrapper from copying
        # a sequence-by-vocabulary tensor. Trainer only needs the scalar loss here.
        return CausalLMOutputWithPast(loss=loss, logits=None,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)

    base_model.forward = MethodType(forward_with_chunked_loss, base_model)
    # Keep a reference for diagnostics and restore/debugging.
    base_model._lifemate_original_forward = original_forward
    base_model._lifemate_loss_chunk_size = chunk_size

