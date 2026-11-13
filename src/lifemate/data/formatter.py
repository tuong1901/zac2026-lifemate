from __future__ import annotations


def encode_assistant_only(messages: list[dict[str, str]], tokenizer) -> dict[str, list[int]]:
    """Apply model chat template and supervise assistant target plus its terminator."""
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("model tokenizer has no chat_template")
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("tokenized assistant prefix is not a prefix of full conversation")
    if len(full_ids) == len(prompt_ids):
        raise ValueError("assistant target was lost by chat template")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
            "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]}
