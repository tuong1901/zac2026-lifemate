"""
llm_client.py — Bộ điều khiển kết nối mô hình ngôn ngữ lớn (LLM Client).
Tuân thủ nghiêm ngặt quy chế ZAC 2026:
- Mô hình: Qwen/Qwen2.5-3B-Instruct (< 5B tham số).
- Sinh ngẫu nhiên: Deterministic decoding (temperature = 0.0, do_sample = False).
- Cố định seed: 42.
- Sẵn sàng chạy offline trong môi trường Docker không có mạng (--network none).
- Hỗ trợ kết nối đồng thời nhiều luồng (Concurrency & Continuous Batching).
- Tự động nạp cấu hình từ .env nếu không truyền tham số.
"""

import os
import time
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
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 512
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = model_name_or_path or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        print(f"[LLM] Đang nạp mô hình HuggingFace: {model_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
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
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        self.max_new_tokens = max_new_tokens
        print(f"[LLM] Đã nạp thành công mô hình trên thiết bị: {self.model.device}", flush=True)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        t0 = time.time()
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
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        )
        
        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        elapsed = time.time() - t0
        return response.strip()


class OpenAICompatibleClient(BaseLLMClient):
    """
    Client kết nối tới vLLM, Ollama hoặc server tương thích OpenAI API (local hoặc dev).
    Hỗ trợ gọi đồng thời nhiều luồng (Concurrency & Continuous Batching).
    Tích hợp cơ chế tự động thử lại (Retry with exponential backoff) chống đứt kết nối ngrok.
    """
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Vui lòng cài đặt thư viện openai: pip install openai")

        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")).strip()
        if "<" in self.base_url or ">" in self.base_url or "ngrok_của_bạn" in self.base_url.lower():
            raise ValueError(
                f"\n❌ [LỖI CẤU HÌNH BASE_URL]: URL endpoint hiện tại là '{self.base_url}'\n"
                f"   Bạn đang để nguyên chuỗi mẫu placeholder '<URL_NGROK_CỦA_BẠN>'.\n"
                f"👉 Hãy thay thế bằng URL ngrok thực tế của bạn (hoặc bỏ cờ --base_url nếu đã điền trong file .env).\n"
                f"   Ví dụ: --base_url https://5b4d-34-16-234-74.ngrok-free.app/v1"
            )

        self.api_key = api_key or os.getenv("LLM_API_KEY", "EMPTY")
        self.model = model or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("LLM_MAX_TOKENS", "512"))
        self.timeout = timeout if timeout is not None else float(os.getenv("LLM_TIMEOUT", "60.0"))
        self.max_retries = max_retries

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers={"ngrok-skip-browser-warning": "true"},
            timeout=self.timeout,
            max_retries=self.max_retries
        )
        print(f"[LLM Client] Sẵn sàng kết nối tới endpoint: {self.base_url} (Model: {self.model}, Timeout: {self.timeout}s)", flush=True)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        t0 = time.time()
        for attempt in range(self.max_retries):
            try:
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
                content = response.choices[0].message.content or ""
                return content.strip()
            except Exception as e:
                elapsed = time.time() - t0
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"      ⚠️ [LLM Retry {attempt+1}/{self.max_retries}] Lỗi ({elapsed:.1f}s): {e}. Thử lại sau {wait_time}s...", flush=True)
                    time.sleep(wait_time)
                else:
                    print(f"      ❌ [LLM Error] Thất bại sau {self.max_retries} lần thử ({elapsed:.2f}s): {e}", flush=True)
                    raise e


def get_llm_client(client_type: Optional[str] = None, **kwargs) -> BaseLLMClient:
    """
    Hàm khởi tạo nhanh LLM Client:
    - client_type="hf": Nạp trực tiếp mô hình HuggingFace Transformers
    - client_type="openai" / "server" / "vllm": Kết nối tới server API
    Tự động đọc các biến môi trường LLM_MODE, LLM_BASE_URL, LLM_MODEL nếu không truyền vào.
    """
    if client_type is None:
        client_type = os.getenv("LLM_MODE", "server").lower()
    else:
        client_type = client_type.lower()

    if client_type == "hf":
        return LocalHuggingFaceClient(**kwargs)
    elif client_type in ("openai", "vllm", "server"):
        return OpenAICompatibleClient(**kwargs)
    else:
        raise ValueError(f"Không hỗ trợ client_type: {client_type}")
