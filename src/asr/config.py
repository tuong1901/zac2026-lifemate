"""
config.py — Cấu hình tập trung cho module ASR, Post-Processing, LLM Agent & Concurrency (ZAC 2026 LifeMate).
Tự động nạp cấu hình từ file .env hoặc biến môi trường hệ thống.
Tương thích 100% môi trường Kaggle (GPU T4/P100), Docker offline và Local.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List


def load_env_file(env_path: Optional[Path] = None) -> None:
    """
    Tự nạp biến môi trường từ file .env.
    Dùng python-dotenv nếu có, hoặc parser nội bộ thuần túy để không phụ thuộc thư viện ngoài.
    """
    if env_path is None:
        # Tìm .env từ thư mục hiện tại lên tới project root
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            candidate = parent / ".env"
            if candidate.exists():
                env_path = candidate
                break

    if env_path and env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=env_path, override=False)
        except ImportError:
            # Fallback parser thuần Python khi chưa cài python-dotenv
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v


# Nạp .env ngay khi module được import
load_env_file()


def _get_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on", "t")


def _get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


@dataclass
class ASRConfig:
    """Cấu hình toàn diện cho ASR và Post-Processing pipeline."""

    # 1. Cấu hình Model ASR
    # Options: 'phowhisper', 'phoasr', 'faster_whisper', 'whisper'
    asr_engine: str = field(default_factory=lambda: os.getenv("ASR_ENGINE", "phowhisper").lower())
    
    # Model name hoặc local path (HuggingFace id hoặc folder path)
    # Gợi ý: 'vinai/phowhisper-small', 'vinai/phowhisper-medium', 'vinai/phowhisper-large',
    # hoặc 'vinai/phoasr-base', 'openai/whisper-large-v3-turbo'
    model_name_or_path: str = field(
        default_factory=lambda: os.getenv("ASR_MODEL_NAME_OR_PATH", "vinai/phowhisper-large")
    )
    
    # Thiết bị & kiểu dữ liệu GPU
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "auto"))
    torch_dtype: str = field(default_factory=lambda: os.getenv("TORCH_DTYPE", "float16"))
    compute_type: str = field(default_factory=lambda: os.getenv("COMPUTE_TYPE", "float16"))

    # 2. Tối ưu Batch & Xử lý nhanh trên GPU (Kaggle T4 / P100 / A100)
    batch_size: int = field(default_factory=lambda: _get_int("ASR_BATCH_SIZE", 16))
    num_workers: int = field(default_factory=lambda: _get_int("NUM_WORKERS", 2))
    sampling_rate: int = field(default_factory=lambda: _get_int("SAMPLING_RATE", 16000))
    chunk_length_s: int = field(default_factory=lambda: _get_int("CHUNK_LENGTH_S", 30))
    
    # Tham số giải mã Deterministic theo luật ZAC 2026
    beam_size: int = field(default_factory=lambda: _get_int("BEAM_SIZE", 1))
    temperature: float = field(default_factory=lambda: _get_float("TEMPERATURE", 0.0))

    # 3. Post-Processing Tầng 1: Từ điển nghiệp vụ & Chuẩn hóa Rule-based
    enable_lexicon_correction: bool = field(
        default_factory=lambda: _get_bool("ENABLE_LEXICON_CORRECTION", True)
    )

    # 4. Post-Processing Tầng 2: Sửa lỗi ngữ cảnh (ViT5 Contextual Correction)
    enable_contextual_correction: bool = field(
        default_factory=lambda: _get_bool("ENABLE_CONTEXTUAL_CORRECTION", False)
    )
    context_model_name_or_path: str = field(
        default_factory=lambda: os.getenv("CONTEXT_MODEL_NAME_OR_PATH", "VietAI/vit5-base")
    )
    context_batch_size: int = field(default_factory=lambda: _get_int("CONTEXT_BATCH_SIZE", 16))

    # 5. Post-Processing Tầng 3: Phục hồi dấu câu & Viết hoa (ViBERT-capu)
    enable_punctuation_restoration: bool = field(
        default_factory=lambda: _get_bool("ENABLE_PUNCTUATION_RESTORATION", False)
    )
    punctuation_model_name_or_path: str = field(
        default_factory=lambda: os.getenv("PUNCTUATION_MODEL_NAME_OR_PATH", "fptai/vibert-capu")
    )
    punctuation_batch_size: int = field(default_factory=lambda: _get_int("PUNCTUATION_BATCH_SIZE", 16))

    # 6. Đường dẫn dữ liệu
    data_root: str = field(
        default_factory=lambda: os.getenv(
            "DATA_ROOT", "/kaggle/input/datasets/n107hoangtuong/zalo-competion/train+public_test"
        )
    )
    output_dir: str = field(
        default_factory=lambda: os.getenv("OUTPUT_DIR", "data/MultiASR_Results")
    )
    benchmark_file: str = field(
        default_factory=lambda: os.getenv("BENCHMARK_FILE", "data/val_benchmark_200.jsonl")
    )

    def summary(self) -> str:
        """Xuất thông tin cấu hình dưới dạng chuỗi bảng tóm tắt."""
        lines = [
            "=" * 60,
            "CẤU HÌNH PIPELINE ASR & POST-PROCESSING (ZAC 2026 LIFEMATE)",
            "=" * 60,
            f"  - ASR Engine:                  {self.asr_engine}",
            f"  - ASR Model:                   {self.model_name_or_path}",
            f"  - Device:                      {self.device} (dtype: {self.torch_dtype})",
            f"  - Batch Size:                  {self.batch_size}",
            f"  - Decoding Mode:               Greedy (temp={self.temperature}, beam={self.beam_size})",
            "-" * 60,
            f"  - [Tầng 1] Lexicon/Rule:       {'BẬT (ON)' if self.enable_lexicon_correction else 'TẮT (OFF)'}",
            f"  - [Tầng 2] ViT5 Contextual:    {'BẬT (ON)' if self.enable_contextual_correction else 'TẮT (OFF)'} [{self.context_model_name_or_path}]",
            f"  - [Tầng 3] ViBERT-capu Punct:  {'BẬT (ON)' if self.enable_punctuation_restoration else 'TẮT (OFF)'} [{self.punctuation_model_name_or_path}]",
            "-" * 60,
            f"  - Data Root:                   {self.data_root}",
            f"  - Output Dir:                  {self.output_dir}",
            "=" * 60,
        ]
        return "\n".join(lines)


def get_asr_config() -> ASRConfig:
    """Trả về instance cấu hình ASR hiện tại."""
    return ASRConfig()


@dataclass
class LLMConfig:
    """Cấu hình toàn diện cho LLM Client, Batching & Concurrency."""
    mode: str = field(default_factory=lambda: os.getenv("LLM_MODE", "server").lower())
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"))
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "EMPTY"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    max_tokens: int = field(default_factory=lambda: _get_int("LLM_MAX_TOKENS", 512))
    timeout: float = field(default_factory=lambda: _get_float("LLM_TIMEOUT", 60.0))
    concurrency: int = field(default_factory=lambda: _get_int("LLM_CONCURRENCY", _get_int("NUM_WORKERS", 8)))
    batch_asr: bool = field(default_factory=lambda: _get_bool("BATCH_ASR", True))
    eval_streams: str = field(default_factory=lambda: os.getenv("EVAL_STREAMS", "all").lower())
    postproc_mode: str = field(default_factory=lambda: os.getenv("POSTPROC_MODE", "aggressive").lower())

    def summary(self) -> str:
        """Xuất thông tin cấu hình LLM & Concurrency."""
        lines = [
            "=" * 60,
            "CẤU HÌNH LLM AGENT & CONCURRENCY (ZAC 2026 LIFEMATE)",
            "=" * 60,
            f"  - LLM Mode:                    {self.mode}",
            f"  - LLM Model:                   {self.model}",
            f"  - Base URL:                    {self.base_url}",
            f"  - Concurrency (Workers):       {self.concurrency} threads",
            f"  - Batch ASR Pre-pass:          {'BẬT (ON)' if self.batch_asr else 'TẮT (OFF)'}",
            f"  - Eval Streams:                {self.eval_streams.upper()}",
            f"  - PostProc Mode:               {self.postproc_mode.upper()}",
            "=" * 60,
        ]
        return "\n".join(lines)


def get_llm_config() -> LLMConfig:
    """Trả về instance cấu hình LLM hiện tại."""
    return LLMConfig()
