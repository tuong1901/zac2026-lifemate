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
import time
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

SEED = 42
random.seed(SEED)


class ContextBudgetExceeded(ValueError):
    """The complete grounding/schema context does not fit the configured model."""


def drop_reference_examples(prompt):
    """Drop optional demonstrations as one section, preserving actual observations."""
    start = prompt.find("[CLOSEST REFERENCE EXEMPLAR]")
    end = prompt.find("[USER UTTERANCE]", start)
    if start >= 0 and end > start:
        return prompt[:start] + prompt[end:]
    return prompt


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
        max_new_tokens: int = 512,
        context_window: int = 16384,
    ):
        if not 0 < max_new_tokens < context_window <= 16384:
            raise ValueError("Invalid context/output token budget")
        import torch
        import numpy as np
        from transformers import AutoModelForCausalLM, AutoTokenizer

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        print(f"[LLM] Đang nạp mô hình: {model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=True,
            trust_remote_code=False
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
            local_files_only=True,
            trust_remote_code=False
        )
        if self.model.num_parameters() >= 5_000_000_000:
            raise ValueError("Competition inference requires an LLM with fewer than 5B parameters")
        self.context_window = min(context_window, getattr(self.model.config, "max_position_embeddings", context_window))
        if self.context_window <= max_new_tokens:
            raise ValueError("Model context is too small for the output token budget")
        self.max_new_tokens = max_new_tokens
        self.model.eval()
        print(f"[LLM] Đã nạp thành công mô hình trên thiết bị: {self.model.device}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.generate_with_budget(system_prompt, user_prompt, 28.0)

    def generate_with_budget(self, system_prompt: str, user_prompt: str, seconds: float) -> str:
        import torch
        deadline = time.monotonic() + seconds

        def encode(prompt):
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return self.tokenizer([text], return_tensors="pt", add_special_tokens=False).to(self.model.device)

        inputs = encode(user_prompt)
        if inputs.input_ids.shape[-1] + self.max_new_tokens > self.context_window:
            inputs = encode(drop_reference_examples(user_prompt))
        if inputs.input_ids.shape[-1] + self.max_new_tokens > self.context_window:
            raise ContextBudgetExceeded("Schema, observations and request exceed context budget; no silent truncation")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            from src.agent.execution import RunLimitReached
            raise RunLimitReached("Time budget exhausted while tokenizing")
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, max_time=remaining,
                do_sample=False, temperature=None, top_p=None,
                pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                              else self.tokenizer.eos_token_id),
            )
        generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


class OpenAICompatibleClient(BaseLLMClient):
    """
    Client kết nối tới vLLM, Ollama hoặc server tương thích OpenAI API (local hoặc dev).
    Ví dụ:
    - vLLM chạy local: http://localhost:8000/v1
    - Ollama chạy local: http://localhost:11434/v1
    """
    def __init__(
        self,
        base_url: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Vui lòng cài đặt thư viện openai: pip install openai")

        resolved_base_url = (base_url or endpoint or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")).strip()
        if "<" in resolved_base_url or ">" in resolved_base_url or "ngrok_của_bạn" in resolved_base_url.lower():
            raise ValueError(
                f"\n❌ [LỖI CẤU HÌNH BASE_URL]: URL endpoint hiện tại là '{resolved_base_url}'\n"
                f"   Bạn đang để nguyên chuỗi mẫu placeholder '<URL_NGROK_CỦA_BẠN>'.\n"
                f"👉 Hãy thay thế bằng URL ngrok thực tế của bạn (hoặc bỏ cờ --base_url nếu đã điền trong file .env).\n"
                f"   Ví dụ: --base_url https://5b4d-34-16-234-74.ngrok-free.app/v1"
            )

        resolved_api_key = api_key or os.getenv("LLM_API_KEY", "EMPTY")
        resolved_model = model or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        resolved_max_tokens = max_tokens if max_tokens is not None else int(os.getenv("LLM_MAX_TOKENS", "256"))
        resolved_timeout = timeout if timeout is not None else float(os.getenv("LLM_TIMEOUT", "25.0"))

        self.client = OpenAI(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            default_headers={"ngrok-skip-browser-warning": "true"},
            timeout=resolved_timeout,
            max_retries=0,
        )
        self.model = resolved_model
        self.max_tokens = resolved_max_tokens
        self.timeout = resolved_timeout
        self.last_error = None
        self.last_error_kind = None

    def generate(self, system_prompt: str = "", user_prompt: str = "", prompt: Optional[str] = None, **kwargs) -> str:
        actual_user = prompt if prompt is not None else user_prompt
        return self.generate_with_budget(system_prompt=system_prompt, user_prompt=actual_user, seconds=self.timeout)

    def generate_with_budget(self, system_prompt: str = "", user_prompt: str = "", seconds: Optional[float] = None, prompt: Optional[str] = None, **kwargs) -> str:
        actual_user = prompt if prompt is not None else user_prompt
        actual_seconds = seconds if seconds is not None else self.timeout
        if actual_seconds <= 0:
            from src.agent.execution import RunLimitReached
            raise RunLimitReached("No time remaining for generation")
        from openai import APIConnectionError, APITimeoutError, APIStatusError
        from src.agent.execution import RunLimitReached
        self.last_error = None
        self.last_error_kind = None
        try:
            response = self.client.with_options(timeout=min(actual_seconds, self.timeout), max_retries=0).chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": actual_user}
                ],
                temperature=0.0,
                max_tokens=self.max_tokens,
                seed=SEED
            )
        except APIStatusError as error:
            # Tự động khắc phục khi server microbatch Colab giới hạn max_tokens <= 256
            if error.status_code == 422 and "max_tokens" in str(error) and self.max_tokens > 256:
                self.max_tokens = 256
                return self.generate_with_budget(system_prompt, actual_user, actual_seconds)
            detail = str(error)
            if len(detail) > 240:
                detail = detail[:237] + "..."
            self.last_error = detail
            self.last_error_kind = (
                "endpoint_unavailable"
                if error.status_code in {404, 502, 503, 504}
                else "http_error"
            )
            raise RunLimitReached(
                f"LLM endpoint returned HTTP {error.status_code}: {detail}; no additional tools will run"
            ) from error
        except (APITimeoutError, TimeoutError) as error:
            # Treat a remote timeout as a per-sample inference failure so a
            # batch evaluation can record the sample and proceed.
            self.last_error = str(error)
            self.last_error_kind = "timeout"
            raise RunLimitReached("LLM request timed out; no additional tools will run") from error
        except APIConnectionError as error:
            self.last_error = str(error)
            self.last_error_kind = "endpoint_unavailable"
            raise RunLimitReached(
                "LLM endpoint connection failed; no additional tools will run"
            ) from error
        return response.choices[0].message.content or ""


def get_llm_client(client_type: Optional[str] = None, **kwargs) -> BaseLLMClient:
    """
    Hàm khởi tạo nhanh LLM Client:
    - client_type="hf": Nạp trực tiếp mô hình HuggingFace Transformers
    - client_type="openai" / "vllm" / "server": Kết nối tới local vLLM / server tương thích
    Tự động đọc cấu hình từ .env nếu không chỉ định.
    """
    if client_type is None:
        client_type = os.getenv("LLM_MODE", "server").lower()
    else:
        client_type = client_type.lower()

    if "endpoint" in kwargs and "base_url" not in kwargs:
        kwargs["base_url"] = kwargs.pop("endpoint")

    if client_type == "hf":
        return LocalHuggingFaceClient(**kwargs)
    elif client_type in ("openai", "vllm", "server"):
        return OpenAICompatibleClient(**kwargs)
    else:
        raise ValueError(f"Không hỗ trợ client_type: {client_type}")
