"""Memory-bounded causal LM loss for decoder models with a linear LM head."""
from __future__ import annotations

from types import MethodType

_DIAGNOSTIC_REPORTED = False
LAST_LOSS_DIAGNOSTIC = None


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

    total = None
    for start in range(0, sequence_length, chunk_size):
        stop = min(start + chunk_size, sequence_length)
        hidden_chunk = hidden_states[:, start:stop, :]
        label_chunk = labels[:, start:stop]

        def chunk_sum(hidden, targets):
            nonlocal first_logits_shape
            logits = lm_head(hidden).float()
            if first_logits_shape is None:
                first_logits_shape = tuple(logits.shape)
            return F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1),
                                   ignore_index=ignore_index, reduction="sum")

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

