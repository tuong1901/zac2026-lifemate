"""
transcribe.py — Bộ chuyển đổi giọng nói thành văn bản (Speech-to-Text) hỗ trợ PhoWhisper, PhoASR và Faster-Whisper.
Tối ưu hóa xử lý theo LÔ (BATCH PROCESSING) trên GPU Kaggle (NVIDIA T4 / P100 / A100).
Tuân thủ nghiêm ngặt quy chế ZAC 2026:
- Tổng tham số < 1B cho module ASR.
- Giải mã tất định (Deterministic decoding: temperature = 0.0, do_sample = False).
- Cố định random seed 42.
- Tuyệt đối không đọc file state.json trong <split>_environment/.
"""

import os
import sys
import random
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np

# Cố định random seed cho khả năng tái lập (Reproducibility)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    torch = None

from src.asr.config import ASRConfig, get_asr_config


def normalize_hf_model_id(model_name: str, engine_type: str) -> str:
    """Chuẩn hóa tên repository ID trên HuggingFace để tránh lỗi 403 / 404."""
    if Path(model_name).exists():
        return model_name

    lower = model_name.lower().strip()
    if engine_type in ("phowhisper", "whisper") or "phowhisper" in lower:
        if "small" in lower:
            return "vinai/PhoWhisper-small"
        elif "large" in lower:
            return "vinai/PhoWhisper-large"
        elif "base" in lower:
            return "vinai/PhoWhisper-base"
        elif "tiny" in lower:
            return "vinai/PhoWhisper-tiny"
        else:
            return "vinai/PhoWhisper-medium"

    if engine_type in ("phoasr", "wav2vec2") or "phoasr" in lower:
        if "wav2vec2" in lower or "nguyenvulebinh" in lower:
            return "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
        return "vinai/phoasr-base"

    return model_name


def load_audio_file(audio_path: Union[str, Path], target_sr: int = 16000) -> np.ndarray:
    """
    Nạp file âm thanh (.wav) và chuyển đổi về dạng đơn kênh (mono) tần số lấy mẫu target_sr (mặc định 16kHz).
    """
    audio_path_str = str(audio_path)
    if not os.path.exists(audio_path_str):
        raise FileNotFoundError(f"Không tìm thấy file audio: {audio_path_str}")

    try:
        import soundfile as sf
        speech, sr = sf.read(audio_path_str)
        if len(speech.shape) > 1:
            speech = np.mean(speech, axis=1)
        if sr != target_sr:
            try:
                import scipy.signal
                num_samples = int(len(speech) * target_sr / sr)
                speech = scipy.signal.resample(speech, num_samples)
            except Exception:
                pass
        return speech.astype(np.float32)
    except Exception:
        try:
            import librosa
            speech, _ = librosa.load(audio_path_str, sr=target_sr, mono=True)
            return speech.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Lỗi đọc file audio {audio_path_str}: {e}")


class PhoWhisperEngine:
    """
    Engine ASR sử dụng PhoWhisper (vinai/PhoWhisper-small, medium, large) hoặc Whisper
    thông qua HuggingFace Transformers Pipeline với khả năng xử lý BATCH trên GPU cực nhanh.
    """
    def __init__(
        self,
        model_name_or_path: str = "vinai/PhoWhisper-medium",
        device: str = "auto",
        torch_dtype: str = "float16",
        batch_size: int = 16,
        chunk_length_s: int = 30
    ):
        from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch

        # Chuẩn hóa tên model HuggingFace ID
        model_name_or_path = normalize_hf_model_id(model_name_or_path, "phowhisper")
        self.model_name = model_name_or_path
        self.batch_size = batch_size
        self.chunk_length_s = chunk_length_s

        if device == "auto":
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = device

        self.dtype = torch.float16 if self.device_str == "cuda" and torch_dtype == "float16" else torch.float32

        print(f"[ASR PhoWhisper] Đang nạp mô hình '{model_name_or_path}' trên {self.device_str} (dtype={self.dtype})...")
        
        # Nạp weights: PhoWhisper dùng định dạng pytorch_model.bin, KHÔNG dùng use_safetensors=True
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name_or_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True
        ).to(self.device_str)

        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        # Sử dụng HuggingFace Pipeline hỗ trợ batching tự động
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            torch_dtype=self.dtype,
            device=self.device_str
        )
        print(f"[ASR PhoWhisper] Nạp thành công pipeline PhoWhisper! Sẵn sàng batch_size={batch_size}.")

    def transcribe(self, audio_path: Union[str, Path], context_hints: Optional[List[str]] = None) -> str:
        """Gỡ băng 1 file audio."""
        return self.transcribe_batch([audio_path], [context_hints] if context_hints else None)[0]

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        context_hints_list: Optional[List[Optional[List[str]]]] = None,
        batch_size: Optional[int] = None
    ) -> List[str]:
        """
        Gỡ băng theo LÔ (BATCH) danh sách các file audio. Tận dụng tối đa song song của GPU.
        """
        if not audio_paths:
            return []

        bs = batch_size or self.batch_size
        str_paths = [str(p) for p in audio_paths]

        generate_kwargs = {
            "language": "vi",
            "task": "transcribe",
            "do_sample": False,
            "temperature": 0.0,
            "num_beams": 1,
        }

        # Gọi pipeline theo batch
        results = self.pipe(
            str_paths,
            batch_size=bs,
            generate_kwargs=generate_kwargs,
            return_timestamps=False
        )

        outputs = []
        if isinstance(results, dict):
            results = [results]

        for r in results:
            text = r.get("text", "").strip()
            outputs.append(text)

        return outputs


class PhoASREngine:
    """
    Engine ASR sử dụng PhoASR / Wav2Vec2 CTC (vinai/phoasr-base hoặc nguyenvulebinh/wav2vec2-base-vietnamese-250h).
    Kiến trúc CTC với tốc độ suy luận cực nhanh trên GPU.
    """
    def __init__(
        self,
        model_name_or_path: str = "vinai/phoasr-base",
        device: str = "auto",
        torch_dtype: str = "float16",
        batch_size: int = 16
    ):
        from transformers import AutoProcessor, AutoModelForCTC
        import torch

        model_name_or_path = normalize_hf_model_id(model_name_or_path, "phoasr")
        self.model_name = model_name_or_path
        self.batch_size = batch_size

        if device == "auto":
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = device

        self.dtype = torch.float16 if self.device_str == "cuda" and torch_dtype == "float16" else torch.float32

        print(f"[ASR PhoASR-CTC] Đang nạp mô hình '{model_name_or_path}' trên {self.device_str}...")
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModelForCTC.from_pretrained(
            model_name_or_path,
            torch_dtype=self.dtype
        ).to(self.device_str)
        self.model.eval()
        print(f"[ASR PhoASR-CTC] Nạp thành công mô hình PhoASR CTC! Sẵn sàng batch_size={batch_size}.")

    def transcribe(self, audio_path: Union[str, Path], context_hints: Optional[List[str]] = None) -> str:
        return self.transcribe_batch([audio_path], [context_hints] if context_hints else None)[0]

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        context_hints_list: Optional[List[Optional[List[str]]]] = None,
        batch_size: Optional[int] = None
    ) -> List[str]:
        import torch

        bs = batch_size or self.batch_size
        results = []

        for i in range(0, len(audio_paths), bs):
            chunk_paths = audio_paths[i : i + bs]
            raw_audios = [load_audio_file(p, target_sr=16000) for p in chunk_paths]

            inputs = self.processor(
                raw_audios,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            ).to(self.device_str)

            with torch.no_grad():
                logits = self.model(inputs.input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = self.processor.batch_decode(predicted_ids)

            for t in transcriptions:
                results.append(t.strip())

        return results


class FasterWhisperEngine:
    """
    Engine ASR sử dụng faster-whisper (CTranslate2).
    """
    def __init__(
        self,
        model_size_or_path: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "float16"
    ):
        from faster_whisper import WhisperModel
        print(f"[ASR Faster-Whisper] Đang nạp model: {model_size_or_path}...")
        self.model = WhisperModel(model_size_or_path, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: Union[str, Path], context_hints: Optional[List[str]] = None) -> str:
        initial_prompt = "Các khẩu lệnh tiếng Việt điều khiển ứng dụng: tin nhắn, lịch, báo thức, đèn, điều hòa, ngân hàng."
        if context_hints:
            initial_prompt += f" Ngữ cảnh: {', '.join(context_hints)}."

        segments, _ = self.model.transcribe(
            str(audio_path),
            language="vi",
            initial_prompt=initial_prompt,
            beam_size=1,
            temperature=0.0,
            vad_filter=True
        )
        return " ".join([s.text.strip() for s in segments]).strip()

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        context_hints_list: Optional[List[Optional[List[str]]]] = None,
        batch_size: Optional[int] = None
    ) -> List[str]:
        if context_hints_list is None:
            context_hints_list = [None] * len(audio_paths)
        return [self.transcribe(p, h) for p, h in zip(audio_paths, context_hints_list)]


class UnifiedASRTranscriber:
    """
    Giao diện ASR thống nhất: tự động nạp PhoWhisper, PhoASR hoặc Faster-Whisper
    dựa trên cấu hình trong .env hoặc tham số truyền vào.
    """
    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or get_asr_config()
        self.engine = None
        self._init_engine()

    def _init_engine(self):
        engine_type = self.config.asr_engine.lower()
        model_name = self.config.model_name_or_path

        try:
            if engine_type in ("phowhisper", "whisper"):
                self.engine = PhoWhisperEngine(
                    model_name_or_path=model_name,
                    device=self.config.device,
                    torch_dtype=self.config.torch_dtype,
                    batch_size=self.config.batch_size,
                    chunk_length_s=self.config.chunk_length_s
                )
            elif engine_type in ("phoasr", "wav2vec2"):
                self.engine = PhoASREngine(
                    model_name_or_path=model_name,
                    device=self.config.device,
                    torch_dtype=self.config.torch_dtype,
                    batch_size=self.config.batch_size
                )
            elif engine_type in ("faster_whisper", "fasterwhisper"):
                self.engine = FasterWhisperEngine(
                    model_size_or_path=model_name if model_name != "vinai/PhoWhisper-medium" else "large-v3-turbo",
                    device=self.config.device,
                    compute_type=self.config.compute_type
                )
            else:
                print(f"[Cảnh báo] Engine '{engine_type}' không xác định. Mặc định dùng PhoWhisper.")
                self.engine = PhoWhisperEngine(
                    model_name_or_path=model_name,
                    device=self.config.device,
                    torch_dtype=self.config.torch_dtype,
                    batch_size=self.config.batch_size
                )
        except Exception as e:
            import traceback
            print(f"\n[ASR LỖI] Không thể nạp ASR Engine ({engine_type} / {model_name}): {e}")
            traceback.print_exc()
            raise RuntimeError(f"Khởi tạo ASR Engine thất bại: {e}")

    def transcribe(self, audio_path: Union[str, Path], context_hints: Optional[List[str]] = None) -> str:
        return self.engine.transcribe(audio_path, context_hints=context_hints)

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        context_hints_list: Optional[List[Optional[List[str]]]] = None,
        batch_size: Optional[int] = None
    ) -> List[str]:
        return self.engine.transcribe_batch(
            audio_paths, context_hints_list=context_hints_list, batch_size=batch_size
        )


# Tương thích ngược với interface cũ của nhóm
SpeechTranscriber = UnifiedASRTranscriber
