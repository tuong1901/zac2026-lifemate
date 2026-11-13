from __future__ import annotations

class PolicyContextOverflow(RuntimeError):
    pass


class QwenPolicy:
    def __init__(self, model, tokenizer, *, max_new_tokens: int = 512, temperature: float = 0.0,
                 max_input_tokens: int = 8192):
        self.model, self.tokenizer = model, tokenizer
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        self.max_input_tokens = max_input_tokens

    def __call__(self, messages: list[dict[str, str]]) -> str:
        import torch
        batch = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_tensors="pt", return_dict=True)
        if batch["input_ids"].shape[-1] > self.max_input_tokens:
            raise PolicyContextOverflow(f"prompt has {batch['input_ids'].shape[-1]} tokens; configured limit is {self.max_input_tokens}")
        device = self.model.get_input_embeddings().weight.device
        batch = {k: v.to(device) for k, v in batch.items()}
        kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": self.temperature > 0,
                  "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                  "eos_token_id": self.tokenizer.eos_token_id}
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature
        with torch.inference_mode():
            generated = self.model.generate(**batch, **kwargs)
        new_tokens = generated[0, batch["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
