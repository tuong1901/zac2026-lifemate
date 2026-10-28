import os
from typing import List, Optional

class SpeechTranscriber:
    """
    Module phụ trách chuyển đổi âm thanh thành văn bản tiếng Việt bằng Faster-Whisper.
    Tuân thủ quy chế ZAC 2026:
    - Kích thước < 1B params (Large-v3-turbo ~809M params)
    - Deterministic decoding: temperature = 0.0, beam_size = 5
    - Tuyệt đối KHÔNG đọc file state.json trong <split>_environment/
    """
    def __init__(self, model_size: str = "large-v3-turbo", device: str = "auto", compute_type: str = "default"):
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except ImportError:
            print("[Cảnh báo] faster-whisper chưa được cài đặt. Chạy ở chế độ Mock/Fallback.")
            self.model = None

    def transcribe(self, audio_path: str, context_hints: Optional[List[str]] = None) -> str:
        """
        Nhận vào đường dẫn audio .wav.
        context_hints: Danh sách từ khóa lấy từ context hợp lệ trong jsonl (tên người dùng, tên thành phố).
        Tuyệt đối không lấy từ state.json!
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Không tìm thấy file audio: {audio_path}")

        if self.model is None:
            return "Bật điều hòa phòng khách"

        # Prompt định hướng chung về khẩu lệnh tiếng Việt điều khiển ứng dụng
        initial_prompt = "Các khẩu lệnh tiếng Việt điều khiển ứng dụng: tin nhắn, lịch, báo thức, đèn, điều hòa, ngân hàng."
        if context_hints:
            initial_prompt += f" Ngữ cảnh: {', '.join(context_hints)}."

        # Cấu hình deterministic decoding (temperature=0.0) theo luật BTC
        segments, _ = self.model.transcribe(
            audio_path,
            language="vi",
            initial_prompt=initial_prompt,
            beam_size=5,
            temperature=0.0,
            vad_filter=True
        )
        text = " ".join([seg.text.strip() for seg in segments])
        return text.strip()
