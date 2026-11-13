from __future__ import annotations

from pathlib import Path


class LocalWhisperASR:
    """Offline local Whisper-family ASR; model path must be attached/provided explicitly."""
    def __init__(self, model_dir: str | Path, device: int = 0):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        import torch
        path = str(Path(model_dir).resolve())
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(path, local_files_only=True, torch_dtype=dtype,
                                                         low_cpu_mem_usage=True, use_safetensors=True)
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.pipeline = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                                 feature_extractor=processor.feature_extractor,
                                 device=device if torch.cuda.is_available() else -1, torch_dtype=dtype)

    def transcribe(self, audio_path: str | Path) -> str:
        result = self.pipeline(str(audio_path), generate_kwargs={"language": "vietnamese", "task": "transcribe"})
        return str(result.get("text", "")).strip()
