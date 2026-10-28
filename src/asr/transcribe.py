import os
from typing import List, Optional

class SpeechTranscriber:
    """
    Module phụ trách chuyển đổi âm thanh thành văn bản tiếng Việt bằng Faster-Whisper.
    Tích hợp kỹ thuật Dynamic Hotwords (nhồi tên danh bạ, thiết bị vào initial_prompt).
    """
    def __init__(self, model_size: str = "large-v3-turbo", device: str = "auto", compute_type: str = "default"):
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except ImportError:
            print("[Cảnh báo] faster-whisper chưa được cài đặt. Chạy ở chế độ Mock/Fallback.")
            self.model = None

    def transcribe(self, audio_path: str, hotwords: Optional[List[str]] = None) -> str:
        """
        Nhận vào đường dẫn audio .wav và danh sách từ khóa gợi ý (hotwords),
        trả về văn bản tiếng Việt.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Không tìm thấy file audio: {audio_path}")

        if self.model is None:
            # Fallback nếu chưa tải model ASR
            return "Bật điều hòa phòng khách"

        # Tận dụng initial_prompt để định hướng ASR không nghe sai tên riêng
        initial_prompt = "Các khẩu lệnh tiếng Việt điều khiển nhà thông minh, tin nhắn, tài khoản."
        if hotwords:
            initial_prompt += f" Danh bạ và thiết bị: {', '.join(hotwords[:30])}."

        segments, _ = self.model.transcribe(
            audio_path,
            language="vi",
            initial_prompt=initial_prompt,
            beam_size=5,
            vad_filter=True
        )
        text = " ".join([seg.text.strip() for seg in segments])
        return text.strip()
