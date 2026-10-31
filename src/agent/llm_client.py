"""
llm_client.py — Bộ điều khiển kết nối mô hình ngôn ngữ lớn (LLM Client).
Tuân thủ nghiêm ngặt quy chế ZAC 2026:
- Mô hình: Qwen/Qwen2.5-3B-Instruct (< 5B tham số).
- Sinh ngẫu nhiên: Deterministic decoding (temperature = 0.0, do_sample = False).
- Cố định seed: 42.
- Sẵn sàng chạy offline trong môi trường Docker không có mạng (--network none).
"""

import os
import random
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

SEED = 42
random.seed(SEED)


class BaseLLMClient(ABC):
    """Giao diện trừu tượng cho mọi LLM Client."""
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Sinh chuỗi phản hồi từ system_prompt và user_prompt."""
        pass


class LocalHuggingFaceClient(BaseLLMClient):
    """
    Client chạy trực tiếp mô hình local qua thư viện HuggingFace Transformers.
    Tối ưu cho Qwen/Qwen2.5-3B-Instruct:
    - Tự động nhận diện GPU (CUDA) hoặc CPU.
    - Greedy decoding chuẩn xác (temperature=0.0, do_sample=False).
    - Chat template chuẩn Qwen2.5.
    """
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 512
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        print(f"[LLM] Đang nạp mô hình: {model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )

        if device is None:
            if torch.cuda.is_available():
                device_map = "auto"
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                device_map = "cpu"
                dtype = torch.float32
        else:
            device_map = device
            dtype = torch.float32 if device == "cpu" else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        self.max_new_tokens = max_new_tokens
        print(f"[LLM] Đã nạp thành công mô hình trên thiết bị: {self.model.device}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)
        
        # Greedy decoding (temperature=0.0, do_sample=False) theo luật BTC
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        )
        
        # Chỉ lấy phần tokens mới sinh ra
        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()


class OpenAICompatibleClient(BaseLLMClient):
    """
    Client kết nối tới vLLM, Ollama hoặc server tương thích OpenAI API (local hoặc dev).
    Ví dụ:
    - vLLM chạy local: http://localhost:8000/v1
    - Ollama chạy local: http://localhost:11434/v1
    """
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen/Qwen2.5-3B-Instruct",
        max_tokens: int = 512
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Vui lòng cài đặt thư viện openai: pip install openai")

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers={"ngrok-skip-browser-warning": "true"}
        )
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=self.max_tokens,
            seed=SEED
        )
        return response.choices[0].message.content or ""


def get_llm_client(client_type: str = "hf", **kwargs) -> BaseLLMClient:
    """
    Hàm khởi tạo nhanh LLM Client:
    - client_type="hf": Nạp trực tiếp mô hình HuggingFace Transformers
    - client_type="openai": Kết nối tới local vLLM / server tương thích
    """
    if client_type == "hf":
        return LocalHuggingFaceClient(**kwargs)
    elif client_type in ("openai", "vllm"):
        return OpenAICompatibleClient(**kwargs)
    else:
        raise ValueError(f"Không hỗ trợ client_type: {client_type}")
