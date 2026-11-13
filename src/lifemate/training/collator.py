from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AssistantOnlyCollator:
    tokenizer: object
    pad_to_multiple_of: int | None = 8

    def __call__(self, features):
        import torch
        max_len = max(len(row["input_ids"]) for row in features)
        if self.pad_to_multiple_of:
            n = self.pad_to_multiple_of
            max_len = ((max_len + n - 1) // n) * n
        pad_id = self.tokenizer.pad_token_id
        input_ids, masks, labels = [], [], []
        for row in features:
            n = max_len - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_id] * n)
            masks.append(row["attention_mask"] + [0] * n)
            labels.append(row["labels"] + [-100] * n)
        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(masks), "labels": torch.tensor(labels)}
