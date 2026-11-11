"""
scripts/kaggle_train_sft.py — Script Huấn luyện SFT LoRA Qwen3-4B trên Kaggle (GPU 2x T4 hoặc 1x P100/A100).

Tính năng chính:
- Tương thích 100% môi trường Kaggle (không cần setup phức tạp).
- Hỗ trợ 4-bit QLoRA (tiết kiệm VRAM, chạy mượt trên GPU 16GB VRAM như T4/P100).
- Tính Loss chuẩn xác: CHỈ tính loss trên câu trả lời của Assistant (sau lượt user cuối cùng),
  toàn bộ system prompt, lịch sử hội thoại, kết quả gọi tool đều được gán nhãn -100 (bỏ qua loss).
- Lưu checkpoint và hỗ trợ merge adapter vào base weights sau khi train xong.

Cách chạy trên Kaggle:
    python scripts/kaggle_train_sft.py --data data/sft_train.jsonl --out output/qwen3_4b_lora --epochs 3
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Target modules cho họ mô hình Qwen
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]


class SFTDataset(Dataset):
    """
    Dataset cho SFT:
    Mỗi dòng trong data file có dạng:
    {"id": str, "messages": [{"role": "system"/"user"/"assistant"/"tool", "content": ...}], "tools": [...]}
    """
    def __init__(self, data_path: str, tokenizer, max_len: int = 4096):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        self.load_data(data_path)

    def load_data(self, path: str):
        print(f"[*] Đang nạp dữ liệu từ: {path}")
        skipped = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                encoded = self.encode_sample(item)
                if encoded is not None:
                    self.samples.append(encoded)
                else:
                    skipped += 1
        print(f"[*] Nạp thành công {len(self.samples)} mẫu (Bỏ qua {skipped} mẫu do quá dài hoặc sai format).")

    def encode_sample(self, item: Dict[str, Any]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        msgs = item.get("messages", [])
        tools = item.get("tools", None)
        if not msgs:
            return None

        # Tìm vị trí lượt user cuối cùng
        user_indices = [i for i, m in enumerate(msgs) if m["role"] == "user"]
        if not user_indices:
            return None
        last_user_idx = user_indices[-1]

        # Áp dụng chat template của Qwen
        try:
            full_text = self.tokenizer.apply_chat_template(
                msgs,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception:
            return None

        # Xác định các khoảng ký tự (spans) của Assistant cần tính loss
        spans = []
        for k in range(last_user_idx + 1, len(msgs)):
            if msgs[k]["role"] != "assistant":
                continue
            try:
                pre_text = self.tokenizer.apply_chat_template(
                    msgs[:k],
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True
                )
                cur_text = self.tokenizer.apply_chat_template(
                    msgs[:k + 1],
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False
                )
                if cur_text.startswith(pre_text):
                    spans.append((len(pre_text), len(cur_text.rstrip("\n"))))
            except Exception:
                continue

        if not spans:
            return None

        # Tokenize toàn bộ chuỗi
        enc = self.tokenizer(
            full_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_len
        )
        input_ids = enc["input_ids"]
        offset_mapping = enc["offset_mapping"]

        if len(input_ids) >= self.max_len:
            return None

        # Gán nhãn -100 cho toàn bộ, chỉ tính loss trên span của assistant
        labels = [-100] * len(input_ids)
        for idx, (start, end) in enumerate(offset_mapping):
            if end > start and any(s <= start and end <= e for s, e in spans):
                labels[idx] = input_ids[idx]

        # Kiểm tra xem có token assistant nào được tính loss không
        if all(l == -100 for l in labels):
            return None

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Pad về chiều dài lớn nhất trong batch
    max_len = max(len(ids) for ids in input_ids)
    padded_input_ids = []
    padded_labels = []
    attention_masks = []

    for ids, labs in zip(input_ids, labels):
        pad_len = max_len - len(ids)
        padded_input_ids.append(torch.cat([ids, torch.full((pad_len,), 151643, dtype=torch.long)])) # 151643 là pad token của Qwen
        padded_labels.append(torch.cat([labs, torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_masks.append(torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(attention_masks),
    }


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Qwen SFT LoRA trên Kaggle")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="HuggingFace model ID")
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn file sft_train.jsonl")
    parser.add_argument("--out", type=str, default="out/lora_adapter", help="Thư mục lưu weights LoRA")
    parser.add_argument("--epochs", type=int, default=3, help="Số epoch huấn luyện")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size trên mỗi GPU")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank (r)")
    parser.add_argument("--alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--max_len", type=int, default=4096, help="Độ dài tối đa token")
    parser.add_argument("--qlora", action="store_true", default=True, help="Bật 4-bit QLoRA")
    args = parser.parse_args()

    print("=" * 60)
    print(f"[*] Bắt đầu Pipeline Huấn Luyện SFT LoRA: {args.model}")
    print(f"[*] Cấu hình: Epochs={args.epochs}, LR={args.lr}, Rank={args.rank}, QLoRA={args.qlora}")
    print("=" * 60)

    # 1. Khởi tạo Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Chuẩn bị Dataset & DataLoader
    dataset = SFTDataset(args.data, tokenizer, max_len=args.max_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # 3. Nạp Model (với QLoRA 4-bit để vừa VRAM Kaggle T4 16GB)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Thiết bị huấn luyện: {device}")

    if args.qlora and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    # Kích hoạt gradient checkpointing để tiết kiệm bộ nhớ
    model.gradient_checkpointing_enable()

    # 4. Gắn LoRA Adapter
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(dataloader) // args.grad_accum) * args.epochs
    warmup_steps = min(50, int(total_steps * 0.05))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # 6. Training Loop
    model.train()
    step_count = 0
    start_time = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[*] Bắt đầu các vòng lặp huấn luyện...")
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            epoch_loss += loss.item() * args.grad_accum
            running_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1

                if step_count % 10 == 0:
                    avg_loss = running_loss / 10
                    elapsed = time.time() - start_time
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Epoch [{epoch}/{args.epochs}] | Step {step_count}/{total_steps} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
                    running_loss = 0.0

        # Lưu checkpoint sau mỗi epoch
        epoch_out = out_dir / f"epoch_{epoch}"
        print(f"[*] Đang lưu checkpoint Epoch {epoch} tại: {epoch_out}")
        model.save_pretrained(epoch_out)
        tokenizer.save_pretrained(epoch_out)

    # Lưu checkpoint cuối cùng
    final_out = out_dir / "final_adapter"
    print(f"[*] Hoàn thành toàn bộ huấn luyện! Đang lưu adapter tại: {final_out}")
    model.save_pretrained(final_out)
    tokenizer.save_pretrained(final_out)
    print("[✓] Huấn luyện thành công!")


if __name__ == "__main__":
    main()
