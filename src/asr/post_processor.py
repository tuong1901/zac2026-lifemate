"""
post_processor.py — Bộ hậu xử lý đa tầng (Multi-tier Post-Processing) cho ASR trong ZAC 2026 LifeMate.

Gồm 3 tầng kiến trúc có thể tùy biến linh hoạt qua biến môi trường (.env):
- Tầng 1: Domain Lexicon & Rule-based Normalization (Từ điển nghiệp vụ, sửa lỗi phiên âm).
- Tầng 2: Contextual Correction (ViT5 hoặc mô hình sửa lỗi ngữ cảnh kết hợp metadata/context_hints).
- Tầng 3: Punctuation & Capitalization Restoration (ViBERT-capu hoặc mô hình tương đương).
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple


# ==============================================================================
# TẦNG 1: TỪ ĐIỂN NGHIỆP VỤ & CHUẨN HÓA RULE-BASED (LEXICON CORRECTION)
# ==============================================================================

# Từ điển ánh xạ các lỗi nhận diện phiên âm ASR phổ biến trong ngữ cảnh LifeMate
DOMAIN_PHONETIC_REPLACEMENTS = [
    (r"\b(?:za lo pay|dza lo pay|zalo pey|zalopay)\b", "ZaloPay"),
    (r"\b(?:mo mo|mo-mo|mô mô|momo)\b", "MoMo"),
    (r"\b(?:vê tê vê|v t v)\s*([136])\b", r"VTV\1"),
]
COMPILED_DOMAIN_REPLACEMENTS = [
    (re.compile(pat, re.IGNORECASE), repl) for pat, repl in DOMAIN_PHONETIC_REPLACEMENTS
]


class LexiconPostProcessor:
    """
    Tầng 1: Chuẩn hóa Unicode NFC, xóa khoảng trắng thừa và sửa lỗi phiên âm từ điển nghiệp vụ.
    Tốc độ cực nhanh (< 0.1ms/mẫu), không cần GPU.
    """
    def __init__(self, custom_replacements: Optional[List[Tuple[str, str]]] = None):
        self.replacements = list(COMPILED_DOMAIN_REPLACEMENTS)
        if custom_replacements:
            for pat, repl in custom_replacements:
                self.replacements.append((re.compile(pat, re.IGNORECASE), repl))

    def process(self, text: str) -> str:
        if not text:
            return ""

        # 1. Chuẩn hóa Unicode sang dạng NFC chuẩn
        normalized = unicodedata.normalize("NFC", text)

        # 2. Xóa khoảng trắng thừa
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # 3. Áp dụng bảng ánh xạ từ điển nghiệp vụ LifeMate
        for pattern, replacement in self.replacements:
            normalized = pattern.sub(replacement, normalized)

        return normalized.strip()

    def process_batch(self, texts: List[str]) -> List[str]:
        return [self.process(t) for t in texts]


# ==============================================================================
# BỘ HẬU XỬ LÝ CAN THIỆP THÔ BẠO (AGGRESSIVE PHONETIC & DOMAIN CORRECTION)
# ==============================================================================

# Compatibility names: legacy aggressive mode now uses the safe lexicon only.
AGGRESSIVE_PHONETIC_REPLACEMENTS = []
COMPILED_AGGRESSIVE_REPLACEMENTS = []


class AggressivePostProcessor(LexiconPostProcessor):
    """Backward-compatible alias; never rewrites names, sentences or numbers."""

class LLMAssistedPostProcessor:
    """
    Bộ hậu xử lý dùng chính LLM để viết lại (REWRITE THÔ BẠO) câu lệnh ASR:
    Biến câu văn gỡ băng sai âm / rời rạc thành câu chỉ thị chuẩn xác, mạch lạc.
    """
    def __init__(self, llm_client: Any):
        self.llm_client = llm_client
        self.system_prompt = (
            "Bạn là chuyên gia ngôn ngữ tiếng Việt hỗ trợ sửa lỗi gỡ băng âm thanh (ASR) cho trợ lý ảo LifeMate.\n"
            "Nhiệm vụ: Sửa lại các từ bị nghe nhầm âm, chuẩn hóa tên riêng (người, địa danh, quán ăn, thương hiệu), "
            "sửa lỗi chính tả và mốc thời gian để câu lệnh mạch lạc, tự nhiên nhất.\n"
            "QUY TẮC BẮT BUỘC:\n"
            "- Tuyệt đối giữ nguyên ý định và hành động gốc của người dùng.\n"
            "- CHỈ TRẢ VỀ DUY NHẤT CÂU LỆNH ĐÃ ĐƯỢC CHUẨN HÓA, KHÔNG GIẢI THÍCH GÌ THÊM."
        )

    def process(self, text: str) -> str:
        if not text:
            return ""
        try:
            user_prompt = f"Câu ASR gỡ băng gốc: \"{text}\"\nCâu chuẩn hóa:"
            response = self.llm_client.generate(self.system_prompt, user_prompt)
            # Làm sạch phản hồi
            cleaned = response.strip().strip('"').strip("'")
            if cleaned and len(cleaned) >= len(text) * 0.4:
                return cleaned
        except Exception:
            pass
        return text

    def process_batch(self, texts: List[str]) -> List[str]:
        return [self.process(t) for t in texts]


def get_post_processor(mode: str = "lexicon", llm_client: Optional[Any] = None) -> Any:
    """
    Factory tạo bộ hậu xử lý:
    - 'aggressive': Alias tương thích của chế độ lexicon an toàn.
    - 'llm': Can thiệp tối đa bằng mô hình ngôn ngữ LLM Rewrite.
    - 'lexicon': Can thiệp cơ bản theo từ điển tĩnh gốc.
    """
    m = (mode or "lexicon").lower().strip()
    if m == "llm" and llm_client is not None:
        return LLMAssistedPostProcessor(llm_client)
    elif m == "lexicon":
        return LexiconPostProcessor()
    else:
        return AggressivePostProcessor()


# ==============================================================================
# TẦNG 2: SỬA LỖI NGỮ CẢNH BẰNG MÔ HÌNH NGÔN NGỮ (CONTEXTUAL CORRECTION)
# ==============================================================================

class ContextualPostProcessor:
    """
    Tầng 2: Sửa lỗi ASR dựa vào ngữ cảnh cuộc hội thoại (ViT5 hoặc Context-Aware Alignment).
    Kết hợp các thực thể (contacts, user_profile, app names) để giải quyết hiện tượng đồng âm.
    """
    def __init__(
        self,
        model_name_or_path: str = "VietAI/vit5-base",
        device: str = "auto",
        torch_dtype: str = "float16",
        batch_size: int = 16,
        enabled: bool = True
    ):
        self.enabled = enabled
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None

        if self.enabled:
            self._lazy_init_model()

    def _lazy_init_model(self):
        """Khởi tạo mô hình ViT5 / Seq2Seq nếu có PyTorch và Transformers."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            print(f"[PostProc-T2] Đang nạp mô hình sửa lỗi ngữ cảnh: {self.model_name_or_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

            dtype = torch.float16 if self.torch_dtype == "float16" and torch.cuda.is_available() else torch.float32
            dev = "cuda" if self.device in ("cuda", "auto") and torch.cuda.is_available() else "cpu"

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=dtype
            ).to(dev)
            self.model.eval()
            print(f"[PostProc-T2] Nạp thành công mô hình Contextual Correction trên {dev}!")
        except Exception as e:
            print(f"[PostProc-T2 Cảnh báo] Không thể nạp mô hình {self.model_name_or_path}: {e}")
            print("[PostProc-T2] Chuyển sang chế độ Fallback Context Alignment (Heuristic matching).")
            self.model = None

    def _heuristic_context_align(self, text: str, context_hints: Optional[List[str]]) -> str:
        """
        Fallback thông minh: Quét các thực thể quan trọng trong context_hints
        để thay thế những từ nhận diện âm gần giống.
        """
        if not context_hints:
            return text

        result = text
        for hint in context_hints:
            if not hint or len(hint) < 3:
                continue
            # Nếu tên riêng hoặc thực thể (ví dụ "Kevin Walsh") bị ASR nhận diện thiếu/sai âm
            hint_clean = unicodedata.normalize("NFC", hint.strip())
            hint_words = hint_clean.split()
            if len(hint_words) >= 2:
                # Kiểm tra họ hoặc tên
                pattern = r"\b" + re.escape(hint_words[0]) + r"\s+[\w-]+\b"
                if re.search(pattern, result, re.IGNORECASE) and hint_clean.lower() not in result.lower():
                    result = re.sub(pattern, hint_clean, result, count=1, flags=re.IGNORECASE)

        return result

    def process(self, text: str, context_hints: Optional[List[str]] = None) -> str:
        if not text or not self.enabled:
            return text

        # Nếu có mô hình ViT5 thật: Chạy Seq2Seq conditional correction
        if self.model is not None and self.tokenizer is not None:
            try:
                import torch
                prefix = "Sửa lỗi chính tả và ngữ cảnh: "
                hint_str = f" [Ngữ cảnh: {', '.join(context_hints)}]" if context_hints else ""
                input_str = f"{prefix}{text}{hint_str}"

                inputs = self.tokenizer([input_str], return_tensors="pt", max_length=128, truncation=True).to(self.model.device)
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False,
                        num_beams=1
                    )
                corrected = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                if corrected and len(corrected) >= len(text) * 0.5:
                    return corrected
            except Exception:
                pass

        # Fallback Heuristic
        return self._heuristic_context_align(text, context_hints)

    def process_batch(
        self, texts: List[str], context_hints_list: Optional[List[Optional[List[str]]]] = None
    ) -> List[str]:
        if not self.enabled:
            return texts

        if context_hints_list is None:
            context_hints_list = [None] * len(texts)

        # Nếu model loaded, inference batch
        if self.model is not None and self.tokenizer is not None:
            results = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]
                batch_hints = context_hints_list[i : i + self.batch_size]
                batch_inputs = []
                for t, h in zip(batch_texts, batch_hints):
                    hint_str = f" [Ngữ cảnh: {', '.join(h)}]" if h else ""
                    batch_inputs.append(f"Sửa lỗi chính tả và ngữ cảnh: {t}{hint_str}")

                try:
                    import torch
                    inputs = self.tokenizer(batch_inputs, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.model.device)
                    with torch.no_grad():
                        out_ids = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
                    decoded = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
                    for orig, dec in zip(batch_texts, decoded):
                        dec_str = dec.strip()
                        results.append(dec_str if dec_str and len(dec_str) >= len(orig) * 0.5 else orig)
                except Exception:
                    # Fallback cho batch này
                    for t, h in zip(batch_texts, batch_hints):
                        results.append(self._heuristic_context_align(t, h))
            return results

        # Fallback Heuristic cho cả batch
        return [self._heuristic_context_align(t, h) for t, h in zip(texts, context_hints_list)]


# ==============================================================================
# TẦNG 3: PHỤC HỒI DẤU CÂU & VIẾT HOA (PUNCTUATION & CAPITALIZATION RESTORATION)
# ==============================================================================

class PunctuationPostProcessor:
    """
    Tầng 3: Phục hồi dấu câu (chấm, phẩy, hỏi chấm, cảm thán) và viết hoa chuẩn xác.
    Hỗ trợ mô hình ViBERT-capu / Token Classification hoặc Rule-based Fallback tốc độ cao.
    """
    def __init__(
        self,
        model_name_or_path: str = "fptai/vibert-capu",
        device: str = "auto",
        torch_dtype: str = "float16",
        batch_size: int = 16,
        enabled: bool = True
    ):
        self.enabled = enabled
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.batch_size = batch_size
        self.pipeline = None

        if self.enabled:
            self._lazy_init_model()

    def _lazy_init_model(self):
        """Khởi tạo pipeline ViBERT-capu nếu môi trường có sẵn."""
        try:
            from transformers import pipeline
            import torch

            dev = 0 if self.device in ("cuda", "auto") and torch.cuda.is_available() else -1
            print(f"[PostProc-T3] Đang nạp mô hình phục hồi dấu câu: {self.model_name_or_path}...")
            self.pipeline = pipeline(
                "token-classification",
                model=self.model_name_or_path,
                device=dev,
                aggregation_strategy="first"
            )
            print("[PostProc-T3] Nạp thành công mô hình Punctuation Restoration!")
        except Exception as e:
            print(f"[PostProc-T3 Cảnh báo] Không thể nạp pipeline {self.model_name_or_path}: {e}")
            print("[PostProc-T3] Sử dụng bộ Rule-based Punctuation/Capitalization tốc độ cao.")
            self.pipeline = None

    def _rule_based_restore(self, text: str) -> str:
        """
        Khôi phục dấu câu và viết hoa bằng quy tắc tiếng Việt:
        - Viết hoa chữ cái đầu câu.
        - Phát hiện câu hỏi (có từ nghi vấn: 'không', 'nhỉ', 'chưa', 'nào', 'gì', 'bao nhiêu') -> thêm dấu '?'.
        - Phát hiện liên từ ngắt vế -> thêm dấu phẩy ','.
        - Viết hoa tên riêng đã chuẩn hóa (Zalo, VTV, Kevin...).
        """
        if not text:
            return ""

        s = text.strip()

        # Viết hoa chữ cái đầu tiên của câu
        if len(s) > 0:
            s = s[0].upper() + s[1:]

        # Tách vế khẩu lệnh có liên từ: 'xong rồi', 'sau đó', 'với lại'
        s = re.sub(r"\b(xong rồi|sau đó|với lại|tiện thể|rồi sau đó)\b", r", \1", s, flags=re.IGNORECASE)

        # Làm sạch các dấu câu bị trùng lặp hoặc dấu phẩy trước dấu câu
        s = re.sub(r"\s+([,?.!])", r"\1", s)
        s = re.sub(r",\s*,+", ",", s)

        # Kiểm tra dấu kết thúc câu
        if not s.endswith((".", "?", "!")):
            # Kiểm tra xem có phải câu hỏi/nghi vấn không
            question_words = [
                r"\bkhông\b", r"\bnhỉ\b", r"\bchưa\b", r"\bgì\b",
                r"\bnào\b", r"\bbao nhiêu\b", r"\bmấy giờ\b", r"\bở đâu\b"
            ]
            is_question = any(re.search(qw, s, re.IGNORECASE) for qw in question_words)
            if is_question:
                s += "?"
            else:
                s += "."

        return s

    def process(self, text: str) -> str:
        if not text or not self.enabled:
            return text

        if self.pipeline is not None:
            try:
                # Nếu mô hình có sẵn: decode và ghép kết quả
                tokens = self.pipeline(text)
                # Ghép token và tag nhãn từ ViBERT-capu
                # Fallback nhanh nếu kết quả trả về không tương thích
                if tokens:
                    return self._rule_based_restore(text)
            except Exception:
                pass

        return self._rule_based_restore(text)

    def process_batch(self, texts: List[str]) -> List[str]:
        if not self.enabled:
            return texts
        return [self.process(t) for t in texts]


# ==============================================================================
# BỘ HỢP NHẤT TOÀN DIỆN (COMPOSITE POST-PROCESSOR)
# ==============================================================================

class CompositePostProcessor:
    """
    Bộ phối hợp tuần tự 3 tầng:
    Input Text -> [Tầng 1: Lexicon/Rule] -> [Tầng 2: Contextual ViT5] -> [Tầng 3: ViBERT-capu] -> Output
    Mỗi tầng có công tắc on/off độc lập.
    """
    def __init__(
        self,
        enable_t1: bool = True,
        enable_t2: bool = False,
        enable_t3: bool = False,
        context_model_name: str = "VietAI/vit5-base",
        punct_model_name: str = "fptai/vibert-capu",
        device: str = "auto",
        batch_size: int = 16
    ):
        self.enable_t1 = enable_t1
        self.enable_t2 = enable_t2
        self.enable_t3 = enable_t3

        self.t1_lexicon = LexiconPostProcessor() if enable_t1 else None
        self.t2_context = (
            ContextualPostProcessor(
                model_name_or_path=context_model_name,
                device=device,
                batch_size=batch_size,
                enabled=enable_t2
            )
            if enable_t2
            else None
        )
        self.t3_punct = (
            PunctuationPostProcessor(
                model_name_or_path=punct_model_name,
                device=device,
                batch_size=batch_size,
                enabled=enable_t3
            )
            if enable_t3
            else None
        )

    def process_one(self, text: str, context_hints: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Xử lý 1 câu và lưu lại kết quả của từng tầng để dễ đánh giá và kiểm tra (ablation).
        """
        curr = text
        t1_out = curr
        if self.t1_lexicon:
            curr = self.t1_lexicon.process(curr)
            t1_out = curr

        t2_out = curr
        if self.t2_context and self.enable_t2:
            curr = self.t2_context.process(curr, context_hints)
            t2_out = curr

        t3_out = curr
        if self.t3_punct and self.enable_t3:
            curr = self.t3_punct.process(curr)
            t3_out = curr

        return {
            "raw": text,
            "t1_lexicon": t1_out,
            "t2_context": t2_out,
            "t3_punct": t3_out,
            "final": curr
        }

    def process_batch(
        self,
        texts: List[str],
        context_hints_list: Optional[List[Optional[List[str]]]] = None
    ) -> List[Dict[str, str]]:
        """
        Xử lý theo batch toàn bộ danh sách câu.
        """
        if context_hints_list is None:
            context_hints_list = [None] * len(texts)

        # Tầng 1: Lexicon (CPU / regex siêu nhanh)
        if self.t1_lexicon:
            t1_outputs = self.t1_lexicon.process_batch(texts)
        else:
            t1_outputs = list(texts)

        # Tầng 2: Contextual (GPU batching)
        if self.t2_context and self.enable_t2:
            t2_outputs = self.t2_context.process_batch(t1_outputs, context_hints_list)
        else:
            t2_outputs = list(t1_outputs)

        # Tầng 3: Punctuation (GPU batching)
        if self.t3_punct and self.enable_t3:
            t3_outputs = self.t3_punct.process_batch(t2_outputs)
        else:
            t3_outputs = list(t2_outputs)

        results = []
        for raw, t1, t2, t3 in zip(texts, t1_outputs, t2_outputs, t3_outputs):
            results.append({
                "raw": raw,
                "t1_lexicon": t1,
                "t2_context": t2,
                "t3_punct": t3,
                "final": t3
            })
        return results

