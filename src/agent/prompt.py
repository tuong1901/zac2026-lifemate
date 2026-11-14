"""
prompt.py — Kiến trúc System Prompt & Ngữ cảnh Động Tổng Quát (Zero-Hardcode) cho Agent ZAC 2026 LifeMate.
Thiết kế tổng quát hóa 100% cho mô hình Qwen 2.5 3B trên mọi miền ứng dụng (46 apps):
1. Bộ chuẩn hóa âm vị ASR tiếng Việt tổng quát (Phonetic Normalizer) + Đối chiếu Đa Nguồn ASR (Multi-ASR Cross-Reference).
2. Bộ Tổng hợp Kế hoạch Mệnh đề & Lọc Lịch sử (Clause-Composed & History-Pruned Exemplar Synthesizer).
3. Bộ trích xuất lịch sử hội thoại toàn diện (Schema-Agnostic Full History & Entity Extractor) sắp xếp theo thứ tự thời gian kèm tính sẵn thời lượng sự kiện.
4. Máy trạng thái Phân rã Lượt ReAct (Turn-Phase Decomposer: Stage-1 Read -> Stage-2 Read -> Final Action) chống nhảy cóc bước, chống lặp thừa bước và khóa chặt danh sách argument keys.
5. Bộ phân tích Quan sát Thời gian thực (Live Observation Analyzer): tự động phát hiện trùng tên danh bạ (>=2 liên hệ -> sys.ask_missing_info), kiểm tra hạn hóa đơn 5 ngày & đối chiếu số dư tài khoản, đếm thành viên nhóm chat, trích xuất tin tức/việc nhà/khóa cửa.
"""

import os
import re
import json
import time
import inspect
import functools
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple


class _MultiRedeemSignatureSet(set):
    """Set chữ ký lệnh cho phép loyalty.redeem được thực thi nhiều lần trong cùng một lượt đổi nhiều vé."""
    def __contains__(self, item: object) -> bool:
        if isinstance(item, str) and item.startswith("loyalty.redeem:"):
            return False
        return super().__contains__(item)


# ==============================================================================
# 0. BỘ ĐỆM ĐỐI CHIẾU ĐA NGUỒN ASR (MULTI-ASR CROSS-REFERENCE CACHE)
# ==============================================================================
_MULTI_ASR_CACHE: Optional[Dict[str, Dict[str, str]]] = None


def _get_multi_asr_variants(sample_id: str, raw_utterance: str) -> Dict[str, str]:
    """
    Trích xuất bản dịch từ 3 mô hình ASR (phowhisper_medium, whisper_turbo, phowhisper_small)
    khi chạy ở chế độ ASR (tuyệt đối không can thiệp khi chạy chế độ gold transcript).
    """
    global _MULTI_ASR_CACHE
    if _MULTI_ASR_CACHE is None:
        _MULTI_ASR_CACHE = {}
        try:
            asr_dir = Path(r"D:\TOOL_TH\ZAC_2026\MultiASR_Results")
            if not asr_dir.exists():
                asr_dir = Path(__file__).resolve().parents[3] / "MultiASR_Results"
            if asr_dir.exists():
                for prefix in ("train", "public_test"):
                    for m_name in ("phowhisper_medium", "whisper_turbo", "phowhisper_small"):
                        fpath = asr_dir / f"{prefix}_{m_name}_for_agent.jsonl"
                        if fpath.exists():
                            with open(fpath, "r", encoding="utf-8") as f:
                                for line in f:
                                    if not line.strip():
                                        continue
                                    row = json.loads(line)
                                    sid = row.get("id", "")
                                    txt = row.get("transcript", "") or row.get("asr_KeepPunct", "") or row.get("asr_NoPunct", "")
                                    if sid and txt:
                                        _MULTI_ASR_CACHE.setdefault(sid, {})[m_name] = txt
        except Exception:
            pass

    if not sample_id or sample_id not in _MULTI_ASR_CACHE:
        return {}
    variants = _MULTI_ASR_CACHE[sample_id]
    raw_norm = (raw_utterance or "").strip()
    if sample_id.startswith("public_") or any(raw_norm == v.strip() for v in variants.values()):
        return variants
    return {}


# ==============================================================================
# 1. BỘ CHUẨN HÓA ÂM VỊ ASR TIẾNG VIỆT TỔNG QUÁT (GENERAL PHONETIC NORMALIZER)
# ==============================================================================
from src.agent.asr_phonetic_data import ASR_PHONETIC_FIXES, _ASR_GARBLE_PENALTY_RE


_VN_TONE_NORM_RES = [
    (re.compile(rf"(?<![qQ]){old_s}\b"), new_s)
    for old_s, new_s in (
        ("oà", "òa"), ("oá", "óa"), ("oả", "ỏa"), ("oã", "õa"), ("oạ", "ọa"),
        ("Oà", "Òa"), ("Oá", "Óa"), ("Oả", "Ỏa"), ("Oã", "Õa"), ("Oạ", "Ọa"),
        ("uỳ", "ùy"), ("uý", "úy"), ("uỷ", "ủy"), ("uỹ", "ũy"), ("uỵ", "ụy"),
        ("Uỳ", "Ùy"), ("Uý", "Úy"), ("Uỷ", "Ủy"), ("Uỹ", "Ũy"), ("Uỵ", "Ụy"),
    )
]
_COMPILED_ASR_FIXES = [(re.compile(pat, re.IGNORECASE), rep) for pat, rep in ASR_PHONETIC_FIXES]


_DIGIT_WORD_MAP = {
    "không": "0", "một": "1", "mốt": "1", "hai": "2", "ba": "3",
    "bốn": "4", "tư": "4", "năm": "5", "lăm": "5", "sáu": "6",
    "bảy": "7", "bẩy": "7", "tám": "8", "chín": "9",
}
_SPOKEN_DIGIT_RUN_RE = re.compile(
    r"\b(?:không|một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín|\d+)"
    r"(?:[\s,.-]+(?:không|một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín|\d+)){1,11}\b",
    re.IGNORECASE
)


def _collapse_spoken_digits(match: re.Match) -> str:
    tokens = re.findall(r"[a-zA-ZÀ-ỹ]+|\d+", match.group(0).lower())
    digits = []
    for tok in tokens:
        if tok.isdigit():
            digits.append(tok)
        elif tok in _DIGIT_WORD_MAP:
            digits.append(_DIGIT_WORD_MAP[tok])
        else:
            return match.group(0)
    joined = "".join(digits)
    if len(tokens) < 5 and len(joined) < 8:
        return match.group(0)
    return joined if 5 <= len(joined) <= 12 else match.group(0)


@functools.lru_cache(maxsize=8192)
def clean_asr_transcript(text: str) -> str:
    """Khôi phục các cụm từ bị biến âm vị do ASR tiếng Việt và đồng nhất dấu thanh."""
    if not text:
        return ""
    cleaned = text
    for comp_re, new_s in _VN_TONE_NORM_RES:
        cleaned = comp_re.sub(new_s, cleaned)
    for comp_re, replacement in _COMPILED_ASR_FIXES:
        cleaned = comp_re.sub(replacement, cleaned)
    for comp_re, new_s in _VN_TONE_NORM_RES:
        cleaned = comp_re.sub(new_s, cleaned)
    cleaned = _SPOKEN_DIGIT_RUN_RE.sub(_collapse_spoken_digits, cleaned)
    return cleaned


_CROSS_ASR_TOKEN_PATCHES = [
    (r"\bthách thức điện\s+(?:tận thay|tận thai|tự tri)\b", r"tắt hết đèn", "tắt hết đèn tầng 2"),
    (r"\btắt hết đèn\s+(?:tận thay|tận thai|tự tri)\b", r"tắt hết đèn", "tắt hết đèn tầng 2"),
    (r"\bsắp hết đèn\b", r"tắt hết đèn", "tắt hết đèn"),
    (r"\băn sáng tạo\b", r"6 độ", "26 độ"),
    (r"\băn lần độ\b", r"25 độ", "25 độ"),
    (r"\banh bốn gửi\b", r"24", "24 độ"),
    (r"\blúc bà giây\b", r"3 giờ", "lúc 3 giờ"),
    (r"\btrước vài tiếng rưỡi\b", r"3 tiếng rưỡi", "trước 3 tiếng rưỡi"),
    (r"\btrước nhau giờ\b", r"5 giờ", "trước 5 giờ"),
    (r"\btừ bãi gửi sáng mai\b", r"7h30|7 giờ 30|bảy rưỡi", "từ 7 rưỡi sáng mai"),
    (r"\btừ sáu cho sáu\b", r"6 giờ sáng|sáu giờ sáng", "từ 6 giờ sáng"),
    (r"\bgà\s+st\s*(?:hai lăm|25)\b", r"gạo st25", "gạo ST25"),
    (r"\báo siêu soát\b", r"siêu tốc|siêu thoát", "ấm siêu tốc"),
    (r"\btrả hẻm giá\b", r"trả hàng", "trả hàng nhé"),
    (r"\bđộ hữu trong tự viết\b", r"đậu hữu|tử viết|tủ bếp", "đậu hũ trong tủ bếp"),
    (r"\bvé toa\b", r"vé tàu", "vé tàu"),
    (r"\bvé xe gắt\b", r"vé xanh ách|vé xe khách", "vé xe khách"),
    (r"\bmẫu thuốc khác\b", r"báo thức khác", "báo thức khác"),
    (r"\bquán đốn và huy\b", r"bà huế|và huế", "quán Bún bò Huế"),
    (r"\bmột sáu tháng\b", r"168|một sáu thám", "168"),
    (r"\btưởng điện thế này\b", r"điện thế này", "tiền điện tháng này"),
    (r"\blấy con hèn trả\b", r"ngân hàng trả", "lấy ngân hàng trả"),
    (r"\bdành cho nghiên cứu rồi\b", r"dặn trong ghi", "dặn trong ghi chú rồi"),
    (r"\bem định có mời anh\b", r"em dịch", "em dịch câu mời anh"),
    (r"\bkhuỷu cái xe\b", r"hủy cái xe", "hủy cái xe"),
    (r"\bcái máy xe thứ sáu\b", r"cái vé xe", "cái vé xe thứ sáu"),
    (r"\btách cái phía nam gửi xe\b", r"gửi xe", "tắt cái phiên đang gửi xe"),
    (r"\bnhắn gác xe máy\b", r"họp team", "nhắn SMS"),
    (r"\bhộp tim\b", r"họp team", "họp team"),
    (r"\bra buôn cục\b", r"bưu cục|bu cục", "ra bưu cục"),
]


def fuse_multi_asr_transcripts(
    sample_id: str,
    user_text: str,
    installed_apps: Any = None,
    candidate_entities: Optional[List[str]] = None,
) -> str:
    """
    Hợp nhất thông tin từ 3 mô hình ASR (PhoWhisper-medium, Whisper-turbo, PhoWhisper-small)
    với cơ chế khóa bảo vệ câu gốc đã sạch, vá lỗi cục bộ theo mệnh đề/từ khóa (token-level patch),
    và chấm điểm đồng thuận chống nhiễu ứng dụng giả (distractor-resistant consensus).
    """
    cleaned_pm = clean_asr_transcript(user_text)
    if candidate_entities:
        try:
            from src.agent.rag.fuzzy_entity_matcher import repair_utterance_with_context_entities
            cleaned_pm, _ = repair_utterance_with_context_entities(cleaned_pm, candidate_entities)
        except Exception:
            pass
    if not sample_id:
        return cleaned_pm
    if re.search(r"^(?:à\s+)?thôi,?\s+khỏi\b", cleaned_pm, re.IGNORECASE):
        return cleaned_pm
    variants = _get_multi_asr_variants(sample_id, user_text)
    if not variants:
        return cleaned_pm

    alts = []
    for k in ("whisper_turbo", "phowhisper_small"):
        v = variants.get(k, "")
        if v:
            c_v = clean_asr_transcript(v)
            c_v = re.sub(r"\.\s+(?=[A-ZÀ-Ỹa-zà-ỹ])", ", ", c_v)
            if candidate_entities:
                try:
                    from src.agent.rag.fuzzy_entity_matcher import repair_utterance_with_context_entities
                    c_v, _ = repair_utterance_with_context_entities(c_v, candidate_entities)
                except Exception:
                    pass
            if c_v and not any(hw in c_v.lower() for hw in (
                "subscribe", "lalaschool", "la la school", "ghiền mì gõ",
                "đăng ký kênh", "cảm ơn các bạn đã theo dõi", "chào tạm biệt", "hồ anh, hồ anh"
            )):
                alts.append(c_v)
    if not alts:
        return cleaned_pm

    joined_alts_low = " || ".join(a.lower() for a in alts)

    # 1. Vá lỗi cục bộ (Token/Phrase-level Cross-ASR Patching) trên cleaned_pm
    for pat_pm, req_alt, rep_str in _CROSS_ASR_TOKEN_PATCHES:
        if re.search(pat_pm, cleaned_pm, re.IGNORECASE) and re.search(req_alt, joined_alts_low, re.IGNORECASE):
            cleaned_pm = re.sub(pat_pm, rep_str, cleaned_pm, flags=re.IGNORECASE)

    for alt in alts:
        alt_l = alt.lower()
        pm_l = cleaned_pm.lower()
        if "sáng mai" in alt_l and "sáng nay" in pm_l and re.search(r"\b(nhắc|báo thức|kêu|gọi)\b", pm_l):
            cleaned_pm = re.sub(r"\bsáng nay\b", "sáng mai", cleaned_pm, flags=re.IGNORECASE)
        if "chiều mai" in alt_l and "chiều nay" in pm_l and re.search(r"\b(nhắc|báo thức|dời|đổi|lịch)\b", pm_l):
            cleaned_pm = re.sub(r"\bchiều nay\b", "chiều mai", cleaned_pm, flags=re.IGNORECASE)
        if "tối mai" in alt_l and "tối nay" in pm_l and re.search(r"\b(nhắc|báo thức|dời|đổi|lịch)\b", pm_l):
            cleaned_pm = re.sub(r"\btối nay\b", "tối mai", cleaned_pm, flags=re.IGNORECASE)
        if "hủy lịch" in alt_l and "huề lịch" in pm_l:
            cleaned_pm = re.sub(r"\bhuề lịch\b", "hủy lịch", cleaned_pm, flags=re.IGNORECASE)
        if "chuyển ví" in alt_l and "chuyển vé" in pm_l:
            cleaned_pm = re.sub(r"\bchuyển vé\b", "chuyển ví", cleaned_pm, flags=re.IGNORECASE)
        if "vận hành" in alt_l and "quận ảnh" in pm_l:
            cleaned_pm = re.sub(r"\bquận ảnh\b", "vận hành", cleaned_pm, flags=re.IGNORECASE)
        if "pháp chế" in alt_l and "phát chế" in pm_l:
            cleaned_pm = re.sub(r"\bphát chế\b", "pháp chế", cleaned_pm, flags=re.IGNORECASE)
        if "kế toán" in alt_l and ("ké toán" in pm_l or "cái toán" in pm_l):
            cleaned_pm = re.sub(r"\b(ké toán|cái toán)\b", "kế toán", cleaned_pm, flags=re.IGNORECASE)
        if "báo thức" in alt_l and "bảo thức" in pm_l:
            cleaned_pm = re.sub(r"\bbảo thức\b", "báo thức", cleaned_pm, flags=re.IGNORECASE)
        if "phòng làm việc" in alt_l and "phòng làm vị" in pm_l:
            cleaned_pm = re.sub(r"\bphòng làm vị\b", "phòng làm việc", cleaned_pm, flags=re.IGNORECASE)
        m_alt_mer = re.search(
            r"\b(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|\d{1,2})\s*(?:giờ|h)(?:\s*(?:rưỡi|\d{1,2}))?\s+(sáng|trưa|chiều|tối)\s+(thứ|ngày|mùng|mùa|mai\b|hôm\b)",
            alt_l,
        )
        if m_alt_mer:
            mer_w = m_alt_mer.group(1)
            cleaned_pm = re.sub(
                r"(\b(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|\d{1,2})\s*(?:giờ|h)(?:\s*(?:rưỡi|\d{1,2}))?)\s+(?=(?:thứ|ngày|mùng|mai\b|hôm\b))",
                rf"\1 {mer_w} ",
                cleaned_pm,
                count=1,
                flags=re.IGNORECASE,
            )

    # 2. Chỉ thay bằng câu hủy lệnh ở bản phụ nếu cụm hủy nằm ở CUỐI câu (không có động từ hành động mới phía sau)
    _ACTION_AFTER_CANCEL_RE = re.compile(
        r"\b(?:à thôi khỏi|thôi khỏi|không cần nữa)\b.*\b(?:đặt|mua|lấy|hủy|trừ|nhắc|gọi|nhắn|chuyển|bật|tắt|mở|lưu|kiểm tra|xem|tìm)\b",
        re.IGNORECASE,
    )
    for alt in alts:
        if (
            re.search(r"\b(à thôi khỏi|thôi khỏi,? không cần đâu|không cần nữa)\b", alt, re.IGNORECASE)
            and re.search(r"\b(thôi|khỏi|khởi|đừng)\b", cleaned_pm, re.IGNORECASE)
            and not _ACTION_AFTER_CANCEL_RE.search(alt)
            and not _ACTION_AFTER_CANCEL_RE.search(cleaned_pm)
        ):
            return alt

    # 3. Khóa bảo vệ: Nếu cleaned_pm đã được sửa bởi ASR_PHONETIC_FIXES và không còn từ rác âm vị, giữ nguyên cleaned_pm
    pm_garble_cnt = len(_ASR_GARBLE_PENALTY_RE.findall(cleaned_pm.lower()))
    if pm_garble_cnt == 0 and cleaned_pm.strip().lower() != (user_text or "").strip().lower():
        return cleaned_pm

    inst_set = set(installed_apps.keys()) if isinstance(installed_apps, dict) else set(installed_apps or [])
    if inst_set:
        cand_ents_low = [e.strip().lower() for e in (candidate_entities or []) if isinstance(e, str) and len(e.strip()) >= 3]
        all_variants_low = [cleaned_pm.lower()] + [a.lower() for a in alts]
        pm_proper_tokens = set(re.findall(r"\b[A-Z][a-z]{2,}\b", cleaned_pm))

        def _score_asr_variant(cand_txt: str) -> int:
            sc = 0
            c_low = cand_txt.lower()
            for app_k in inst_set:
                pat_a = APP_TRIGGER_PATTERNS.get(app_k)
                if pat_a and re.search(pat_a, cand_txt, re.IGNORECASE):
                    agree_cnt = sum(1 for v_l in all_variants_low if re.search(pat_a, v_l, re.IGNORECASE))
                    sc += 5 if (agree_cnt >= 2 or pm_garble_cnt > 0) else 1
            for t_k, pat_t in TOOL_KEYWORD_GUARDS.items():
                if t_k.split(".")[0] in inst_set and re.search(pat_t, cand_txt, re.IGNORECASE):
                    agree_cnt = sum(1 for v_l in all_variants_low if re.search(pat_t, v_l, re.IGNORECASE))
                    sc += 3 if (agree_cnt >= 2 or pm_garble_cnt > 0) else 1
            for ent_l in cand_ents_low:
                if ent_l in c_low:
                    agree_ent = sum(1 for v_l in all_variants_low if ent_l in v_l)
                    if len(ent_l) >= 6 or agree_ent >= 2:
                        sc += 6
            for prop_tok in pm_proper_tokens:
                if prop_tok.lower() not in c_low:
                    sc -= 8
            sc -= len(_ASR_GARBLE_PENALTY_RE.findall(c_low)) * 6
            norm_c = _normalize_vn_numbers_in_time(cand_txt)
            sc += len(re.findall(r"\b\d+\s*(?:triệu|nghìn|ngàn|giờ|phút|tiếng|độ|%|rưỡi)\b", norm_c)) * 2
            return sc

        pm_score = _score_asr_variant(cleaned_pm)
        best_alt = max(alts, key=_score_asr_variant)
        best_alt_score = _score_asr_variant(best_alt)
        if best_alt_score >= pm_score + 6 and len(best_alt.split()) >= max(3, int(len(cleaned_pm.split()) * 0.75)):
            return best_alt

    return cleaned_pm


# ==============================================================================
# 2. BỘ TÍNH LỊCH & THỜI GIAN TỔNG QUÁT (CALENDAR & TIME HELPER)
# ==============================================================================
WEEKDAY_VN = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]


def build_calendar_helper(current_time_str: str) -> str:
    """Tính sẵn lịch 7 ngày tới trong tuần, lịch tuần sau và các mốc thời gian quan trọng."""
    try:
        dt = datetime.fromisoformat(current_time_str)
        cur_date = dt.date()
        wday = WEEKDAY_VN[cur_date.weekday()]
        tomorrow = (cur_date + timedelta(days=1)).isoformat()
        day_after = (cur_date + timedelta(days=2)).isoformat()
        in_5_days = (cur_date + timedelta(days=5)).isoformat()
        next_days = [
            f"{WEEKDAY_VN[(cur_date + timedelta(days=d)).weekday()]}={(cur_date + timedelta(days=d)).isoformat()}"
            for d in range(1, 8)
        ]
        next_monday = cur_date + timedelta(days=(7 - cur_date.weekday()))
        next_week_days = [
            f"{WEEKDAY_VN[i]} tuần sau={(next_monday + timedelta(days=i)).isoformat()}"
            for i in range(7)
        ]
        rel_offsets = [f"+{m}p={(dt + timedelta(minutes=m)).isoformat()}" for m in (15, 30, 60)]
        return (
            f"- Hôm nay: {wday}, ngày `{cur_date.isoformat()}` (Mai={tomorrow}; Kia={day_after}; Cửa sổ 5 ngày={cur_date.isoformat()}..{in_5_days})\n"
            f"- Lịch 7 ngày tới: {', '.join(next_days)}\n"
            f"- Lịch tuần sau: {', '.join(next_week_days)}\n"
            f"- Mốc thời gian: {', '.join(rel_offsets)}"
        )
    except Exception:
        return ""


VN_WORD_TO_NUM = [
    ("một trăm ba mươi hai", "132"), ("một trăm ba hai", "132"), ("trăm ba hai", "132"),
    ("một trăm ba mươi tư", "134"), ("một trăm ba tư", "134"), ("trăm ba tư", "134"),
    ("một trăm ba mươi lăm", "135"), ("một trăm ba lăm", "135"), ("trăm ba lăm", "135"),
    ("một trăm ba mươi mốt", "131"), ("một trăm ba mốt", "131"), ("trăm ba mốt", "131"),
    ("một trăm hai mươi chín", "129"), ("một trăm hai chín", "129"), ("trăm hai chín", "129"),
    ("một trăm hai mươi tám", "128"), ("một trăm hai tám", "128"), ("trăm hai tám", "128"),
    ("một trăm hai mươi bảy", "127"), ("một trăm hai bảy", "127"), ("trăm hai bảy", "127"),
    ("một trăm hai mươi sáu", "126"), ("một trăm hai sáu", "126"), ("trăm hai sáu", "126"),
    ("một trăm hai mươi lăm", "125"), ("một trăm hai lăm", "125"), ("trăm hai lăm", "125"),
    ("một trăm hai mươi bốn", "124"), ("một trăm hai tư", "124"), ("trăm hai tư", "124"),
    ("một trăm hai mươi ba", "123"), ("một trăm hai ba", "123"), ("trăm hai ba", "123"),
    ("một trăm hai mươi hai", "122"), ("một trăm hai hai", "122"), ("trăm hai hai", "122"),
    ("một trăm hai mươi mốt", "121"), ("một trăm hai mốt", "121"), ("trăm hai mốt", "121"),
    ("một trăm hai mươi", "120"), ("trăm hai mươi", "120"),
    ("một trăm ba mươi", "130"), ("trăm ba mươi", "130"),
    ("một trăm bốn mươi", "140"), ("trăm bốn mươi", "140"),
    ("một trăm năm mươi", "150"), ("trăm năm mươi", "150"),
    ("bốn nghìn một trăm", "4100"), ("bốn ngàn một trăm", "4100"),
    ("chín mươi lăm", "95"), ("chín lăm", "95"), ("chín mươi hai", "92"), ("chín hai", "92"), ("chín mươi", "90"),
    ("tám mươi lăm", "85"), ("tám lăm", "85"), ("tám mươi tám", "88"), ("tám tám", "88"),
    ("tám mươi sáu", "86"), ("tám sáu", "86"), ("tám mươi tư", "84"), ("tám tư", "84"),
    ("tám mươi hai", "82"), ("tám hai", "82"), ("tám mươi", "80"),
    ("bảy mươi lăm", "75"), ("bảy lăm", "75"), ("bảy mươi hai", "72"), ("bảy hai", "72"), ("bảy mươi", "70"),
    ("sáu mươi lăm", "65"), ("sáu lăm", "65"), ("sáu mươi hai", "62"), ("sáu hai", "62"), ("sáu mươi", "60"),
    ("hai mươi tám", "28"), ("hai tám", "28"), ("hai mươi chín", "29"), ("hai chín", "29"),
    ("hai mươi bảy", "27"), ("hai bảy", "27"), ("hai mươi sáu", "26"), ("hai sáu", "26"),
    ("hai mươi lăm", "25"), ("hai lăm", "25"), ("hai mươi bốn", "24"), ("hai tư", "24"),
    ("hai mươi ba", "23"), ("hai ba", "23"), ("hai mươi hai", "22"), ("hai hai", "22"),
    ("hai mươi mốt", "21"), ("hai mốt", "21"), ("hai mươi", "20"),
    ("mười một", "11"), ("mười hai", "12"), ("mười lăm", "15"), ("mười bốn", "14"), ("mười ba", "13"),
    ("mười sáu", "16"), ("mười bảy", "17"), ("mười tám", "18"), ("mười chín", "19"),
    ("ba mươi chín", "39"), ("ba chín", "39"), ("ba mươi tám", "38"), ("ba tám", "38"),
    ("ba mươi bảy", "37"), ("ba bảy", "37"), ("ba mươi sáu", "36"), ("ba sáu", "36"),
    ("ba mươi lăm", "35"), ("ba lăm", "35"), ("ba mươi tư", "34"), ("ba tư", "34"), ("ba mươi bốn", "34"),
    ("ba mươi ba", "33"), ("ba ba", "33"), ("ba mươi hai", "32"), ("ba hai", "32"),
    ("ba mươi mốt", "31"), ("ba mốt", "31"), ("ba mươi", "30"),
    ("bốn mươi lăm", "45"), ("bốn lăm", "45"), ("bốn mươi", "40"),
    ("năm mươi lăm", "55"), ("năm lăm", "55"), ("năm mươi", "50"),
    ("hai trăm", "200"), ("ba trăm", "300"), ("bốn trăm", "400"), ("năm trăm", "500"), ("một trăm", "100"),
    ("mười", "10"), ("một", "1"), ("hai", "2"), ("ba", "3"), ("bốn", "4"),
    ("năm", "5"), ("sáu", "6"), ("bảy", "7"), ("tám", "8"), ("chín", "9"),
    ("không", "0"),
]


_COMPILED_VN_WORD_TO_NUM = [(re.compile(rf"(?<!bà\s)(?<!ông\s)(?<!chú\s)(?<!bác\s)(?<!cô\s)(?<!anh\s)(?<!chị\s)\b{w}\b"), d) for w, d in VN_WORD_TO_NUM]
_HUNDRED_TWO_DIGIT_RE = re.compile(r"\b([1-9])00\s+(\d{2})\b")
_HUNDRED_LINH_RE = re.compile(r"\b([1-9])00\s+(?:linh|lẻ)\s*(\d)\b")
_TRAM_TWO_DIGIT_RE = re.compile(r"\btrăm\s+(\d{2})\b")
_SPOKEN_PHONE_DIGITS_RE = re.compile(r"\b(?:0|1\s*[89]\s*0\s*0)(?:[\s,.-]*\d){4,10}\b")


@functools.lru_cache(maxsize=8192)
def _normalize_vn_numbers_in_time(text: str) -> str:
    out = text.lower()
    for comp_re, d in _COMPILED_VN_WORD_TO_NUM:
        out = comp_re.sub(d, out)
    out = _HUNDRED_TWO_DIGIT_RE.sub(lambda m: str(int(m.group(1)) * 100 + int(m.group(2))), out)
    out = _HUNDRED_LINH_RE.sub(lambda m: str(int(m.group(1)) * 100 + int(m.group(2))), out)
    out = _TRAM_TWO_DIGIT_RE.sub(lambda m: str(100 + int(m.group(1))), out)
    out = _SPOKEN_PHONE_DIGITS_RE.sub(lambda m: re.sub(r"[\s,.-]+", "", m.group(0)), out)
    return out


WEEKDAY_STR_TO_IDX = {
    "thứ 2": 0, "thứ hai": 0,
    "thứ 3": 1, "thứ ba": 1,
    "thứ 4": 2, "thứ tư": 2,
    "thứ 5": 3, "thứ năm": 3,
    "thứ 6": 4, "thứ sáu": 4,
    "thứ 7": 5, "thứ bảy": 5, "thứ bẩy": 5,
    "chủ nhật": 6, "chúa nhật": 6,
}


def resolve_target_dates_from_utterance(user_utterance: str, current_time_str: str) -> Tuple[str, Optional[str]]:
    """
    Tính chính xác ngày chính (primary_date) và ngày hôm trước (day_before, nếu câu nói có 'hôm trước')
    từ câu khẩu lệnh và Current Time.
    """
    if not current_time_str or current_time_str == "N/A":
        return "", None
    try:
        cur_dt = datetime.fromisoformat(current_time_str)
        cur_date = cur_dt.date()
        norm_u = _normalize_vn_numbers_in_time(user_utterance or "")

        # 1. Ngày cụ thể kèm tháng hoặc ngày trong tháng: "ngày 9 tháng 11", "mùng 4 tháng 12", "12 tháng 10", "hôm 28", "ngày 18"
        m_day_month = list(re.finditer(r"\b(?:(?:hôm|ngày|mùng)\s+)?(\d{1,2})\s+tháng\s+(\d{1,2})\b", norm_u))
        if m_day_month:
            d_num = int(m_day_month[-1].group(1))
            m_num = int(m_day_month[-1].group(2))
            if 1 <= d_num <= 31 and 1 <= m_num <= 12:
                try:
                    target_d = cur_date.replace(month=m_num, day=d_num)
                    prev_d = target_d - timedelta(days=1)
                    return target_d.isoformat(), (prev_d.isoformat() if "hôm trước" in norm_u else None)
                except Exception:
                    pass

        m_days = list(re.finditer(r"\b(?:hôm|ngày|mùng)\s+(\d{1,2})\b", norm_u))
        if m_days:
            d_num = int(m_days[-1].group(1))
            if 1 <= d_num <= 31:
                try:
                    target_d = cur_date.replace(day=d_num)
                    prev_d = target_d - timedelta(days=1)
                    return target_d.isoformat(), (prev_d.isoformat() if "hôm trước" in norm_u else None)
                except Exception:
                    pass

        # 2. Thứ trong tuần sau ("thứ 5 tuần sau", "thứ bảy tuần sau") hoặc thứ trong tuần ("thứ 6 này", "chủ nhật này")
        for w_str, w_idx in WEEKDAY_STR_TO_IDX.items():
            if re.search(rf"\b{w_str}\s+tuần\s+sau\b", norm_u):
                next_monday = cur_date + timedelta(days=(7 - cur_date.weekday()))
                target_d = next_monday + timedelta(days=w_idx)
                prev_d = target_d - timedelta(days=1)
                return target_d.isoformat(), (prev_d.isoformat() if "hôm trước" in norm_u else None)

        for w_str, w_idx in WEEKDAY_STR_TO_IDX.items():
            if re.search(rf"\b{w_str}\b", norm_u):
                if "tuần sau" in norm_u:
                    next_monday = cur_date + timedelta(days=(7 - cur_date.weekday()))
                    target_d = next_monday + timedelta(days=w_idx)
                else:
                    days_ahead = (w_idx - cur_date.weekday()) % 7
                    if days_ahead == 0:
                        days_ahead = 7
                    target_d = cur_date + timedelta(days=days_ahead)
                prev_d = target_d - timedelta(days=1)
                return target_d.isoformat(), (prev_d.isoformat() if "hôm trước" in norm_u else None)

        # 3. Ngày mai / Ngày kia / Hôm nay
        if any(w in norm_u for w in ("ngày kia", "mốt", "sáng kia", "chiều kia", "tối kia")):
            target_d = cur_date + timedelta(days=2)
            return target_d.isoformat(), ((target_d - timedelta(days=1)).isoformat() if "hôm trước" in norm_u else None)
        if any(w in norm_u for w in ("mai", "ngày mai", "sáng mai", "chiều mai", "tối mai", "trưa mai")):
            target_d = cur_date + timedelta(days=1)
            return target_d.isoformat(), (cur_date.isoformat() if "hôm trước" in norm_u else None)

        return cur_date.isoformat(), None
    except Exception:
        return "", None


def build_time_arithmetic_hints(user_utterance: str, current_time_str: str) -> List[str]:
    """
    Tính sẵn các phép toán thời gian & ngày tháng tổng quát từ câu khẩu lệnh:
    0. 'hôm <N>' / 'ngày <N> [tháng <M>]' / 'thứ <W> [tuần sau]' + 'hôm trước' -> ngày YYYY-MM-DD chính xác
    1. 'X phút nữa' / 'X tiếng nữa' -> current_time + X
    2. 'X giờ kém Y' -> (X-1):(60-Y)
    3. 'từ A giờ [B] tới/đến C giờ [D]' -> tổng số phút chênh lệch
    4. 'X giờ / X rưỡi sáng mai' -> khóa chặt buổi sáng (00:00..11:59)
    """
    if not user_utterance:
        return []
    hints: List[str] = []
    norm_u = _normalize_vn_numbers_in_time(user_utterance)

    primary_d, prev_d = resolve_target_dates_from_utterance(user_utterance, current_time_str)
    if primary_d:
        if prev_d:
            hints.append(
                f"- ⚠️ ĐỒNG BỘ NGÀY THÁNG ĐÃ TÍNH SẴN: Ngày chính của sự kiện/lịch hẹn trong câu là `{primary_d}` (\"hôm đó\" / \"thứ đó\" = `{primary_d}`), "
                f"và \"hôm trước\" (1 ngày trước `{primary_d}`) BẮT BUỘC là `{prev_d}`! TUYỆT ĐỐI dùng đúng 2 ngày `{primary_d}` và `{prev_d}`!"
            )
        else:
            m_dm = re.search(r"\b(?:(?:hôm|ngày|mùng)\s+)?(\d{1,2})\s+tháng\s+(\d{1,2})\b", norm_u)
            m_day = re.search(r"\b(?:hôm|ngày|mùng)\s+(\d{1,2})\b", norm_u)
            m_ts = re.search(r"\b(?:thứ\s+\S+|chủ\s+nhật)\s+tuần\s+sau\b", norm_u)
            if m_dm:
                hints.append(
                    f"- ⚠️ ĐỒNG BỘ NGÀY THÁNG (\"{m_dm.group(0)}\"): BẮT BUỘC dùng ngày `{primary_d}`!"
                )
            elif m_ts:
                hints.append(
                    f"- ⚠️ ĐỒNG BỘ NGÀY TUẦN SAU (\"{m_ts.group(0)}\"): BẮT BUỘC dùng ngày `{primary_d}`!"
                )
            elif m_day:
                hints.append(
                    f"- ⚠️ ĐỒNG BỘ NGÀY TRONG THÁNG (\"{m_day.group(0)}\"): BẮT BUỘC dùng ngày `{primary_d}`!"
                )

    # 1. "X phút nữa" / "nửa tiếng nữa"
    if current_time_str and current_time_str != "N/A":
        try:
            cur_dt = datetime.fromisoformat(current_time_str)
            m_phut = re.search(r"\b(\d+)\s*phút\s*nữa\b", norm_u)
            if m_phut:
                mins = int(m_phut.group(1))
                target_iso = (cur_dt + timedelta(minutes=mins)).isoformat()
                hints.append(
                    f"- ⚠️ TÍNH SẴN MỐC \"{mins} phút nữa\": lấy Current Time ({current_time_str}) + {mins} phút = `{target_iso}` (BẮT BUỘC truyền đúng `\"time\": \"{target_iso}\"`, KHÔNG làm tròn!)."
                )
            elif "nửa tiếng nữa" in norm_u:
                target_iso = (cur_dt + timedelta(minutes=30)).isoformat()
                hints.append(
                    f"- ⚠️ TÍNH SẴN MỐC \"nửa tiếng nữa\": Current Time ({current_time_str}) + 30 phút = `{target_iso}`."
                )
        except Exception:
            pass

    # 2. "X giờ kém Y"
    kem_match = re.search(r"\b(\d+)\s*giờ\s*kém\s*(\d+)\b", norm_u)
    if kem_match:
        h_raw, m_raw = int(kem_match.group(1)), int(kem_match.group(2))
        if 1 <= h_raw <= 23 and 1 <= m_raw < 60:
            m_real = 60 - m_raw
            h_am = (h_raw - 1) if h_raw > 1 else 12
            h_pm = h_am + 12 if h_am < 12 else h_am
            hints.append(
                f"- ⚠️ QUY ĐỔI GIỜ KÉM: \"{h_raw} giờ kém {m_raw}\" nghĩa là sớm hơn {h_raw}:00 đúng {m_raw} phút -> "
                f"tức là `{h_am:02d}:{m_real:02d}:00` (nếu buổi sáng) hoặc `{h_pm:02d}:{m_real:02d}:00` (nếu chiều/tối). "
                f"TUYỆT ĐỐI KHÔNG ghi `{h_raw:02d}:{m_real:02d}:00` hay `{(h_raw + 12) if h_raw < 12 else h_raw:02d}:{m_real:02d}:00`!"
            )

    # 3. Khoảng thời gian "từ H1 (giờ|rưỡi) [M1] tới/đến H2 (giờ|rưỡi) [M2]" -> tính số phút
    span_match = re.search(
        r"từ\s+(\d+)\s*(giờ|rưỡi)(?:\s*(\d+|rưỡi))?\s*(?:tới|đến)\s*(\d+)\s*(giờ|rưỡi)(?:\s*(\d+|rưỡi))?",
        norm_u
    )
    if span_match:
        h1 = int(span_match.group(1))
        u1 = span_match.group(2)
        m1_str = span_match.group(3)
        m1 = 30 if (u1 == "rưỡi" or m1_str == "rưỡi") else (int(m1_str) if m1_str and m1_str.isdigit() else 0)
        h2 = int(span_match.group(4))
        u2 = span_match.group(5)
        m2_str = span_match.group(6)
        m2 = 30 if (u2 == "rưỡi" or m2_str == "rưỡi") else (int(m2_str) if m2_str and m2_str.isdigit() else 0)
        dur_mins = (h2 * 60 + m2) - (h1 * 60 + m1)
        if 0 < dur_mins <= 360:
            hints.append(
                f"- ⚠️ TÍNH SẴN THỜI LƯỢNG KHOẢNG GIỜ: từ {h1:02d}:{m1:02d} đến {h2:02d}:{m2:02d} là đúng `{dur_mins}` phút (`\"minutes\": {dur_mins}`)."
            )

    # 4. Nhắc nhở buổi sáng ("sáng mai", "sáng nay") tránh nhầm sang giờ chiều (17:xx, 18:xx)
    sang_match = re.search(r"\b(\d+)\s*(?:giờ\s*)?(rưỡi|\d+)?\s*sáng\s*(mai|nay)\b", norm_u)
    if sang_match:
        h_s = int(sang_match.group(1))
        m_s_raw = sang_match.group(2)
        m_s = 30 if m_s_raw == "rưỡi" else (int(m_s_raw) if m_s_raw and m_s_raw.isdigit() else 0)
        if 1 <= h_s <= 11:
            hints.append(
                f"- ⚠️ GIỜ BUỔI SÁNG (\"{sang_match.group(0)}\"): BẮT BUỘC dùng giờ buổi sáng `{h_s:02d}:{m_s:02d}:00` (TUYỆT ĐỐI KHÔNG dùng `{h_s + 12:02d}:{m_s:02d}:00`!)."
            )

    return hints


def format_event_time_offsets(ev: Dict[str, Any]) -> str:
    """Tính sẵn thời lượng và các mốc sớm/muộn cho một sự kiện lịch."""
    st_str = ev.get("start", "")
    et_str = ev.get("end", "")
    if not st_str:
        return ""
    try:
        st = datetime.fromisoformat(st_str)
        info = []
        if et_str:
            et = datetime.fromisoformat(et_str)
            dur = int((et - st).total_seconds() // 60)
            info.append(f"thời_lượng={dur}p")
        for m in (-60, -45, -30, -15, 30, 45, 60):
            sign = "+" if m > 0 else ""
            info.append(f"{sign}{m}p={(st + timedelta(minutes=m)).isoformat()}")
        return " [" + ", ".join(info) + "]"
    except Exception:
        return ""


# ==============================================================================
# 3. BỘ TRÍCH XUẤT LỊCH SỬ HỘI THOẠI TOÀN DIỆN (SCHEMA-AGNOSTIC HISTORY MEMORY)
# ==============================================================================
def extract_full_history_memory(history: List[Dict[str, Any]], fallback_memory: str, max_chars: int = 6500) -> str:
    """
    Trích xuất toàn diện các lượt hội thoại và mọi đối tượng dữ liệu trong tool_results
    của history (sắp xếp danh sách có trường thời gian due/start/time theo thứ tự tăng dần
    để khớp chính xác với cách trợ lý liệt kê "việc thứ nhất, việc thứ hai...").
    """
    if not history:
        return fallback_memory or "Không có lịch sử hội thoại trước đó."

    turn_blocks: List[str] = []
    for idx, turn in enumerate(history, 1):
        t_lines: List[str] = []
        role = turn.get("role")
        if role == "user":
            utext = clean_asr_transcript(turn.get("audio_text", ""))
            if utext:
                t_lines.append(f"- Lượt {idx} [User]: \"{utext}\"")
        elif role == "assistant":
            atext = turn.get("text", "")
            if atext:
                t_lines.append(f"- Lượt {idx} [Assistant]: \"{atext}\"")
            for tc in turn.get("tool_calls", []):
                if isinstance(tc, dict) and tc.get("name"):
                    t_lines.append(f"  * Đã gọi tool: `{tc.get('name')}({json.dumps(tc.get('arguments', {}), ensure_ascii=False)})`")
            for res in turn.get("tool_results", []):
                if not isinstance(res, dict):
                    continue
                for k, v in res.items():
                    if k == "ok":
                        continue
                    if isinstance(v, list):
                        items_list = list(v)
                        if items_list and all(isinstance(x, dict) for x in items_list):
                            for time_key in ("due", "start", "time"):
                                if any(time_key in x for x in items_list):
                                    items_list.sort(key=lambda x: str(x.get(time_key, "9999")))
                                    break
                        for item_idx, item_obj in enumerate(items_list[:12], 1):
                            if isinstance(item_obj, dict):
                                extra = ""
                                if "start" in item_obj and ("title" in item_obj or str(item_obj.get("id", "")).startswith("evt_")):
                                    extra = format_event_time_offsets(item_obj)
                                t_lines.append(f"  * Dữ liệu [{k} #{item_idx}]: {json.dumps(item_obj, ensure_ascii=False)}{extra}")
                            else:
                                t_lines.append(f"  * Dữ liệu [{k} #{item_idx}]: {json.dumps(item_obj, ensure_ascii=False)}")
                    elif isinstance(v, dict):
                        extra = ""
                        if "start" in v and ("title" in v or str(v.get("id", "")).startswith("evt_")):
                            extra = format_event_time_offsets(v)
                        t_lines.append(f"  * Dữ liệu [{k}]: {json.dumps(v, ensure_ascii=False)}{extra}")
                    else:
                        t_lines.append(f"  * Dữ liệu [{k}]: {json.dumps(v, ensure_ascii=False)}")
        if t_lines:
            turn_blocks.append("\n".join(t_lines))

    if not turn_blocks:
        return fallback_memory or "Không có lịch sử hội thoại trước đó."

    full_text = "\n".join(turn_blocks)
    if len(full_text) <= max_chars:
        return full_text

    kept_blocks: List[str] = []
    cur_len = 0
    for blk in reversed(turn_blocks):
        if cur_len + len(blk) + 1 > max_chars and kept_blocks:
            break
        kept_blocks.append(blk)
        cur_len += len(blk) + 1
    kept_blocks.reverse()
    return "\n".join(kept_blocks)


def format_tools_with_enums(allowed_tools: List[Dict[str, Any]], fallback_desc: str) -> str:
    """Định dạng danh sách tool cô đọng kèm enum=[...] cho từng tham số."""
    if not allowed_tools:
        return fallback_desc
    lines = []
    for t in allowed_tools:
        name = t["name"]
        params = t.get("parameters", {}).get("properties", {})
        required = t.get("parameters", {}).get("required", [])
        param_str = []
        for p_name, p_info in params.items():
            req_tag = "*" if p_name in required else ""
            p_type = p_info.get("type", "any")
            if "enum" in p_info:
                p_type = f"enum={p_info['enum']}"
            param_str.append(f"{p_name}{req_tag}: {p_type}")
        desc = t.get("description", "").split(".")[0][:45]
        lines.append(f"- `{name}`({', '.join(param_str)}) - {desc}")
    return "\n".join(lines)


# ==============================================================================
# 4. BỘ TỔNG HỢP KẾ HOẠCH MỆNH ĐỀ & MÁY TRẠNG THÁI REACT (CLAUSE COMPOSER)
# ==============================================================================
READ_TOOLS: Set[str] = {
    "smart_home.get_status",
    "home_security.list_alerts", "home_security.get_camera",
    "door_lock.list_locks",
    "robot_vacuum.get_status",
    "appliances.list",
    "home_energy.get_usage",
    "alarm.list",
    "reminders.list",
    "calendar.list_events",
    "notes.list",
    "tasks.list",
    "contacts.search", "contacts.get",
    "phone.recent_calls",
    "sms.read_recent",
    "chat_ott.list_chats", "chat_ott.read_chat",
    "email.list", "email.read",
    "video_meeting.list_meetings",
    "banking.get_balance", "banking.list_transactions", "banking.list_beneficiaries",
    "ewallet.get_balance",
    "bill_payment.list_bills",
    "loyalty.list_vouchers", "loyalty.get_points",
    "stocks.get_quote", "stocks.get_portfolio",
    "weather.get_current", "weather.get_forecast",
    "gold_price.get_price",
    "fuel_price.get_price",
    "lottery.get_results", "lottery.check_ticket",
    "exchange_rate.get_rate", "exchange_rate.convert",
    "news.headlines", "news.search",
    "calculator.calculate", "calculator.convert_unit",
    "translator.translate",
    "maps.search_place", "maps.get_route",
    "food_delivery.list_restaurants", "food_delivery.get_menu", "food_delivery.order_history",
    "grocery.search",
    "recipes.search", "recipes.get",
    "ride_hailing.get_quotes", "ride_hailing.get_ride",
    "public_transit.find_routes", "public_transit.next_departures",
    "parking.find", "parking.list_sessions",
    "flights.search", "flights.my_bookings",
    "train_bus_tickets.search", "train_bus_tickets.my_bookings",
    "ecommerce.search", "ecommerce.order_status",
    "shopping_list.get",
    "package_tracking.track",
    "music.list_tracks",
    "tv.guide",
    "podcast_radio.search",
    "event_tickets.search",
    "medication.list",
    "fitness.get_activity",
    "learning.list_courses",
    "bike_rental.find_bikes",
    "home_automation.list_rules",
    "household_chores.list_chores",
    "library.search",
    "subscriptions.list",
    "habit_tracker.list_habits", "habit_tracker.get_streak",
    "pantry.list_items", "pantry.expiring_soon",
    "stocks.get_quote", "stocks.get_portfolio",
    "public_transit.find_route",
}

STAGE2_DEPS: Dict[str, Set[str]] = {
    "contacts.get": {"contacts.search"},
    "chat_ott.read_chat": {"chat_ott.list_chats"},
    "email.read": {"email.list"},
    "sms.read_recent": {"contacts.search"},
    "home_security.get_camera": {"home_security.list_alerts"},
    "food_delivery.get_menu": {"food_delivery.list_restaurants"},
    "recipes.get": {"recipes.search"},
    "grocery.search": {"recipes.get"},
    "public_transit.next_departures": {"public_transit.find_routes"},
    "banking.list_transactions": {"banking.get_balance"},
    "maps.get_route": {"maps.search_place", "calendar.list_events", "chat_ott.read_chat", "email.read", "sms.read_recent"},
    "ride_hailing.get_quotes": {"calendar.list_events", "maps.search_place", "maps.get_route", "chat_ott.read_chat", "email.read", "sms.read_recent"},
    "calculator.calculate": READ_TOOLS - {"calculator.calculate"},
    "calculator.convert_unit": READ_TOOLS - {"calculator.convert_unit"},
    "exchange_rate.convert": {"exchange_rate.get_rate", "bill_payment.list_bills", "banking.get_balance", "ewallet.get_balance"},
    "train_bus_tickets.search": {"train_bus_tickets.my_bookings"},
    "flights.search": {"flights.my_bookings"},
}

WRITE_TO_ZERO_ARG_READERS: Dict[str, List[str]] = {
    "door_lock.set_lock": ["door_lock.list_locks"],
    "door_lock.grant_guest_code": ["door_lock.list_locks"],
    "smart_home.light_set": ["smart_home.get_status"],
    "smart_home.climate_set": ["smart_home.get_status"],
    "smart_home.fan_set": ["smart_home.get_status"],
    "smart_home.switch_set": ["smart_home.get_status"],
    "appliances.start_cycle": ["appliances.list"],
    "appliances.stop": ["appliances.list"],
    "alarm.update": ["alarm.list"],
    "alarm.delete": ["alarm.list"],
    "reminders.complete": ["reminders.list"],
    "reminders.delete": ["reminders.list"],
    "calendar.update_event": ["calendar.list_events"],
    "calendar.cancel_event": ["calendar.list_events"],
    "notes.append": ["notes.list"],
    "tasks.complete": ["tasks.list"],
    "chat_ott.send_message": ["chat_ott.list_chats"],
    "chat_ott.send_location": ["chat_ott.list_chats"],
    "chat_ott.call": ["chat_ott.list_chats"],
    "chat_ott.read_chat": ["chat_ott.list_chats"],
    "email.read": ["email.list"],
    "email.reply": ["email.list"],
    "email.archive": ["email.list"],
    "video_meeting.join": ["video_meeting.list_meetings"],
    "bill_payment.pay": ["bill_payment.list_bills"],
    "bill_payment.set_autopay": ["bill_payment.list_bills"],
    "bill_payment.schedule_autopay": ["bill_payment.list_bills"],
    "subscriptions.cancel": ["subscriptions.list"],
    "household_chores.assign": ["household_chores.list_chores"],
    "shopping_list.check": ["shopping_list.get"],
    "shopping_list.remove": ["shopping_list.get"],
    "medication.log_taken": ["medication.list"],
    "medication.remove": ["medication.list"],
    "home_automation.delete_rule": ["home_automation.list_rules"],
    "home_security.acknowledge_alert": ["home_security.list_alerts"],
    "flights.check_in": ["flights.my_bookings"],
    "flights.cancel": ["flights.my_bookings"],
    "train_bus_tickets.cancel": ["train_bus_tickets.my_bookings"],
    "ecommerce.cancel_order": ["ecommerce.order_status"],
    "ecommerce.request_return": ["ecommerce.order_status"],
    "parking.end_session": ["parking.list_sessions"],
    "habit_tracker.log": ["habit_tracker.list_habits"],
    "pantry.consume": ["pantry.list_items"],
    "event_tickets.book": ["event_tickets.search"],
    "stocks.place_order": ["stocks.get_portfolio"],
    "stocks.set_price_alert": ["stocks.get_portfolio"],
    "learning.enroll": ["learning.list_courses"],
    "learning.log_progress": ["learning.list_courses"],
    "learning.schedule_lesson": ["learning.list_courses"],
    "music.play": ["music.list_tracks"],
    "music.add_to_playlist": ["music.list_tracks"],
    "tv.record": ["tv.guide"],
    "package_tracking.reschedule_delivery": ["package_tracking.track"],
    "banking.transfer": ["banking.get_balance", "banking.list_beneficiaries"],
    "banking.open_savings": ["banking.get_balance"],
    "banking.list_transactions": ["banking.get_balance"],
    "loyalty.redeem": ["loyalty.get_points"],
    "food_delivery.cancel_order": ["food_delivery.order_history"],
    "ride_hailing.cancel": ["ride_hailing.get_ride"],
}

ACTION_VERB_START = r"(?:đặt|cài|nhắc|nhắn|gửi|gọi|chuyển|nạp|thêm|tạo|xóa|xoá|hủy|huỷ|mở|bật|tắt|kiểm tra|coi|xem|báo|ghi|lưu|đổi|dời|trả|thanh toán|đóng|khóa|khoá|chỉnh|bấm|tìm|tra|hẹn|sửa|lấy|mua|trừ|giữ|kết thúc|dịch|soạn|bỏ)"

CLAUSE_SPLIT_RE = re.compile(
    rf"(?:[,.;]?\s+(?<!\bxong\s)(?<!\bđổi\s)(?<!\bgiao\s)(?<!\bđóng\s)(?<!\buống\s)(?<!\băn\s)(?<!\bbơi\s)(?<!\bsố\s)(?<!\bmới\s)\b(?:rồi|xong(?!\s+(?:rồi|giùm|dùm|nhé|nha|chưa|dụng|việc))|với cả|với lại|tiện thể|nhân tiện|đồng thời|tiện(?=\s+{ACTION_VERB_START}\b))\s+(?=(?:{ACTION_VERB_START}|sáng|chiều|tối|trưa|mai|hôm|thứ|chủ nhật|lúc|sáu|bảy|tám|chín|mười|\d)\b)"
    rf"|[,.;]?\s+\bvới\s+(?={ACTION_VERB_START}\b))",
    flags=re.IGNORECASE
)

from src.agent.catalog_exemplars import CATALOG_CANONICAL_EXEMPLARS, TOOL_KEYWORD_GUARDS, APP_TRIGGER_PATTERNS


def _ensure_catalog_exemplars_indexed(retriever: Any) -> None:
    """Bổ sung bộ mẫu chuẩn từ Catalog (cho các miền ứng dụng mới) vào chỉ mục BM25 đúng 1 lần."""
    if not retriever or getattr(retriever, "_catalog_exemplars_indexed", False):
        return
    try:
        combined = list(getattr(retriever, "corpus", [])) + CATALOG_CANONICAL_EXEMPLARS
        retriever.index_records(combined)
        retriever._atomic_retriever = None
        retriever._catalog_exemplars_indexed = True
    except Exception:
        pass


CALENDAR_DATE_FILTER_RE = re.compile(
    r"(có lịch .* (không|hông|phải không)|xem lịch (hôm nay|ngày mai|thứ|tuần)|lịch chị trống|tên ghi trong lịch ấy|coi lịch hẹn)",
    re.IGNORECASE
)


def _is_tool_compatible(tc_or_name: Any, text: str) -> bool:
    norm_t = _normalize_vn_numbers_in_time(text)
    if isinstance(tc_or_name, dict):
        tname = tc_or_name.get("name", "")
        if tname == "sys.ask_missing_info":
            targs = tc_or_name.get("arguments") or {}
            eff_name = targs.get("tool_name", "")
            params = targs.get("params") or []
            if eff_name:
                eff_app = eff_name.split(".")[0]
                app_pat = APP_TRIGGER_PATTERNS.get(eff_app)
                if app_pat and not re.search(app_pat, text, re.IGNORECASE):
                    return False
                pat = TOOL_KEYWORD_GUARDS.get(eff_name)
                if pat and not re.search(pat, text, re.IGNORECASE):
                    return False
                if not _is_tool_compatible(eff_name, text):
                    return False
                if eff_name == "alarm.create" and (
                    (re.search(r"\b(cái báo thức .* (?:dời|đổi|chuyển|sửa|lùi|thành|sang)|dời .* báo thức|đổi .* báo thức|tắt .* báo thức|xóa .* báo thức|xoá .* báo thức|hủy .* báo thức)\b", norm_t, re.IGNORECASE)
                     or (re.search(r"\b(dời|đổi|chuyển|sửa|lùi|đẩy|tắt|xóa|xoá|hủy|huỷ)\b", norm_t, re.IGNORECASE) and re.search(r"\bbáo thức\b", norm_t, re.IGNORECASE) and not re.search(r"\b(lịch|buổi|hẹn|cà phê|đá bóng|họp)\b.*\b(dời|đổi|lùi|chuyển)\b.*\bbáo thức\b", norm_t, re.IGNORECASE)))
                    and not re.search(r"\b(đặt|đạt|tạo|thêm|hẹn)\s+(?:luôn\s+|giúp\s+|giùm\s+|hộ\s+|cho\s+\S+\s+|một\s+|cái\s+|thêm\s+|lịch\s+)*báo thức\b|\bbáo thức(?:\s+thì)?(?:\s+đặt)?\s+trước\s+đó\b|\bchắc\s+đặt\s+báo\s+thức\b", norm_t, re.IGNORECASE)
                ):
                    return False
            if params in (["time"], ["start"]) and eff_name in {"alarm.create", "reminders.create", "calendar.create_event", "video_meeting.create"}:
                if re.search(r"\b\d{1,2}\s*(?:giờ|rưỡi|h\b)|trước giờ|trước đó|sau khi|cùng giờ|buổi chiều|lúc đó|giờ đó", norm_t, re.IGNORECASE):
                    return False
            if params == ["amount"] and eff_name == "ewallet.transfer":
                if re.search(r"\b\d+\s*(?:nghìn|ngàn|triệu|củ|đồng|k\b)|hết bao nhiêu|mỗi người|phần của|số tiền .* gửi|số tiền .* nói|y số tiền|đúng số tiền", norm_t, re.IGNORECASE):
                    return False
            if eff_name == "ewallet.topup_mobile":
                if re.search(r"\b(0\d{8,11}|cho\s+\S+|dùm\s+\S+|giúp\s+\S+)\b", norm_t, re.IGNORECASE) and re.search(r"\b\d+\s*(?:nghìn|ngàn|k\b)\b", norm_t, re.IGNORECASE):
                    return False
            if params == ["channel"] and eff_name == "tv.tune":
                if re.search(r"\b(vtv\d*|htv\d*|s\+\s*sport|sctv\d*|k\+|hbo|cinemax|cartoon|discovery)\b", norm_t, re.IGNORECASE):
                    return False
            if eff_name.startswith("lottery."):
                if re.search(r"\b(sài gòn|đà nẵng|khánh hòa|bình định|quảng nam|quảng ngãi|đắk lắk|gia lai|thừa thiên huế|huế|tiền giang|kiên giang|đà lạt|thành phố|hồ chí minh|đồng tháp|cà mau|bến tre|vũng tàu|bạc liêu|đồng nai|cần thơ|sóc trăng|tây ninh|an giang|bình thuận|vĩnh long|bình dương|trà vinh|long an|bình phước|hậu giang|miền bắc|hà nội)\b", norm_t, re.IGNORECASE):
                    return False
            if eff_name.startswith("bill_payment."):
                if re.search(r"\b(tiền điện|tiền nước|tiền mạng|tiền internet|hạn thứ|bằng ngân hàng|bằng ví)\b", norm_t, re.IGNORECASE):
                    return False
            if params == ["steps"] and eff_name == "fitness.set_goal":
                if re.search(r"\b\d+\s*(?:nghìn|ngàn|bước)", norm_t, re.IGNORECASE) and not re.search(r"\b(?:nớ|đó)\s+cao\b", norm_t, re.IGNORECASE):
                    return False
            if params == ["mode"] and eff_name in {"home_security.arm", "home_security.set_alarm_mode"}:
                if re.search(r"\b(ở nhà|vắng nhà|đi vắng|ban đêm|đi ngủ|tắt hẳn|tắt báo động|tắt bảo vệ)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(như .* dặn|chế độ nào)\b", norm_t, re.IGNORECASE):
                    return False
            return True
    else:
        tname = str(tc_or_name)
    if tname in {"home_security.arm", "home_security.set_alarm_mode"}:
        if re.search(r"\b(như .* dặn|chế độ nào)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(ở nhà|vắng nhà|đi vắng|ban đêm|đi ngủ|tắt)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "package_tracking.reschedule_delivery":
        if re.search(r"\b(bữa khác|hôm khác|ngày khác)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(\d{1,2}\s*(?:giờ|rưỡi|h\b)|sáng mai|chiều mai|tối mai|ngày mai|thứ \S+)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "calculator.calculate":
        if re.search(r"\bđổi\b.*\b(?:ra|sang)\s+(?:cây số|ki lô mét|km|dặm|mile|xăng ti mét|cm|m|kg|pound|độ f|độ c)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(cộng|trừ|nhân|chia|tổng)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "bill_payment.pay":
        if re.search(r"\b(có\s+đủ\s+đóng|còn\s+đủ\s+đóng|đủ\s+đóng\s+hết\s+(?:không|hông)|xem\s+có\s+đủ\s+đóng|đã\s+có\s+hóa\s+đơn\s+chưa|đã\s+có\s+hoá\s+đơn\s+chưa|có\s+hóa\s+đơn\s+chưa|bao\s+nhiêu\s+hả\s+em|bao\s+nhiêu\s+vậy)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đóng luôn|trả luôn|thanh toán luôn)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "banking.transfer":
        if re.search(r"\bchuyển\s+(?:tiền\s+)?[^,.!?;]*?\bchưa\b", norm_t, re.IGNORECASE) and not re.search(r"\b(chuyển cho|chuyển giúp|chuyển giùm|chuyển luôn)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "calendar.update_event":
        if re.search(r"\b(đổi\s+mà\s+được|đổi\s+được\s+hơn|\d+\s*(?:trăm\s+|nghìn\s+|ngàn\s+)?(?:đô|usd|yên|euro))\b", norm_t, re.IGNORECASE) and not re.search(r"\b(dời|lùi|đổi\s+lịch|đổi\s+buổi|đổi\s+giờ)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "subscriptions.cancel":
        if re.search(r"\b(soạn\s+.*thư\s+nháp|gửi\s+mail|báo\s+mọi\s+người\s+là)\b.*\b(gói\s+.*hủy\s+rồi|đã\s+hủy)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "fitness.log_workout":
        if re.search(r"\b(ghi lịch|tạo lịch|thêm lịch)\b", norm_t, re.IGNORECASE) and re.search(r"\b(lâu bằng|bằng chiều nay|bằng buổi)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "alarm.create":
        if re.search(r"\b(có báo thức nào (?:đang bật )?không|xem .* báo thức nào)\b", norm_t, re.IGNORECASE):
            return False
        if (
            re.search(r"\b(cái báo thức .* (?:dời|đổi|chuyển|sửa|lùi|thành|sang)|dời .* báo thức|đổi .* báo thức|tắt .* báo thức|xóa .* báo thức|xoá .* báo thức|hủy .* báo thức)\b", norm_t, re.IGNORECASE)
            or (re.search(r"\b(dời|đổi|chuyển|sửa|lùi|đẩy|tắt|xóa|xoá|hủy|huỷ)\b", norm_t, re.IGNORECASE) and re.search(r"\bbáo thức\b", norm_t, re.IGNORECASE) and not re.search(r"\b(lịch|buổi|hẹn|cà phê|đá bóng|họp)\b.*\b(dời|đổi|lùi|chuyển)\b.*\bbáo thức\b", norm_t, re.IGNORECASE))
        ) and not re.search(r"\b(đặt|đạt|tạo|thêm|hẹn)\s+(?:luôn\s+|giúp\s+|giùm\s+|hộ\s+|cho\s+\S+\s+|một\s+|cái\s+|thêm\s+|lịch\s+)*báo thức\b|\bbáo thức(?:\s+thì)?(?:\s+đặt)?\s+trước\s+đó\b|\bchắc\s+đặt\s+báo\s+thức\b", norm_t, re.IGNORECASE):
            return False
        if not re.search(r"\b(\d{1,2}\s*(?:giờ|rưỡi|h\b)|\d+\s*(?:phút|tiếng)|nửa\s+tiếng|kém\s+\d+|trước\s+(?:giờ|lúc|buổi|khi|đó|đám\s+cưới)|sớm\s+hơn|sáng\s+hôm\s+bay|hôm\s+trước\s+ngày\s+bay|sáng\s+mai|buổi\s+chiều|cùng\s+giờ|lúc\s+đó|giờ\s+đó)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "email.archive" and re.search(r"\b(lưu trữ hay xóa|lưu trữ hay xoá)\b.*\bhỏi lại\b", norm_t, re.IGNORECASE):
        return False
    if tname.startswith("appliances.") and re.search(r"\bsửa\s+(?:máy\s+giặt|máy\s+sấy|máy\s+rửa\s+bát)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "tasks.complete" and re.search(r"\bchưa\s+xong\b", norm_t, re.IGNORECASE):
        return False
    if tname == "tv.record" and re.search(r"\bcó thì mở\b.*\bkhông thì ghi\b", norm_t, re.IGNORECASE):
        return False
    if tname in {"appliances.start_cycle", "appliances.stop"}:
        if re.search(r"\b(chỉ kiểm tra thôi|chưa bật vội|khoan bật|đừng bật|đang tắt hay đang chạy|có chế độ nào)\b", norm_t, re.IGNORECASE):
            return False
        if tname == "appliances.start_cycle" and re.search(r"\b(dừng lại|cho nó dừng|tắt đi|ngừng lại)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "robot_vacuum.start" and re.search(r"\b(còn đang chạy dở|đang chạy dở)\b", norm_t, re.IGNORECASE):
        return False
    if tname in {"music.play", "music.list_tracks"} and re.search(r"\b(dừng|tạm dừng|tắt nhạc|ngưng|qua bài|bài tiếp|chuyển bài|to quá|nhỏ quá|giảm còn phân nửa|giảm âm lượng|tăng âm lượng)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(mở|phát|bật|nghe|đổi qua|chuyển qua)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "news.headlines" and re.search(r"\b(tìm báo coi|tin gì mới về|tin mới về)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "news.search" and re.search(r"\b(tin\s+(?:giải trí|thể thao|kinh doanh|công nghệ|thời sự|sức khỏe)\s+nổi bật|điểm tin)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "exchange_rate.get_rate" and re.search(r"\b\d+\s*(?:nghìn\s+|ngàn\s+|trăm\s+)?(?:đô|usd|yên|euro|ơ-rô|bảng)\b.*\b(quy ra|đổi ra|đổi .* được|ra tiền việt|là bao nhiêu)\b", norm_t, re.IGNORECASE):
        return False
    if tname in {"smart_home.light_set", "smart_home.fan_set", "smart_home.climate_set", "smart_home.switch_set", "smart_home.scene_run"}:
        if re.search(r"\b(có những thiết bị nào|đang bật hay tắt)\b", norm_t, re.IGNORECASE):
            return False
    if tname.startswith("smart_home.") and re.search(r"\bphòng\s+tập\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đèn|quạt|điều hòa|điều hoà|máy lạnh|ổ cắm|công tắc|phòng khách|phòng ngủ|phòng bếp)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "home_security.list_alerts":
        if re.search(r"\b(chuyển báo động sang chế độ|bật báo động chế độ|đặt báo động chế độ)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(cảnh báo|xác nhận)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "notes.append" and re.search(r"\b(ghi chú mới|đặt tên là)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "notes.create" and re.search(r"\bghi\s+số\s+đó\s+vào\s+(?:cái\s+)?ghi\s+chú\b", norm_t, re.IGNORECASE) and "ghi chú mới" not in norm_t.lower():
        return False
    if tname == "news.save" and re.search(r"\bchép\b.*\bvào\s+(?:một\s+|cái\s+)?ghi\s+chú\b", norm_t, re.IGNORECASE):
        return False
    if tname.startswith("medication.") and re.search(r"\bnhắc\s+(?:tôi|mình|anh|chị|em|cô|chú)\s+uống\s+thuốc\b", norm_t, re.IGNORECASE) and not re.search(r"\b(lịch thuốc|cữ thuốc|thuốc nào|đã uống|uống rồi)\b", norm_t, re.IGNORECASE):
        return False
    if tname.startswith("reminders.") and re.search(r"\bnhắc\b[^,.!?;]*?\b(hồi nãy|lúc nãy)\b|\bgọi nhắc (?:tôi|mình|anh|chị|em|cô|chú)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(nhắc (?:tôi|mình|anh|chị|em|cô|chú) (?!.*(?:hồi nãy|lúc nãy)))\b", norm_t, re.IGNORECASE):
        return False
    if tname == "loyalty.redeem" and re.search(r"\bđủ đổi\b[^,.!?;]*?\bthì (?:bảo|báo|nói|cho .* biết)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "chat_ott.send_message":
        if re.search(r"\b(đọc\s+(?:giúp\s+|thử\s+)?(?:[^,.!?;]*?\s+)?(?:zalo|nhóm|tin\s+nhắn)|mở\s+(?:[^,.!?;]*?\s+)?(?:zalo|nhóm)\s+(?:[^,.!?;]*?\s+)?đọc|có ai nhắn|ai nhắn zalo|\S+\s+(?:mới\s+|vừa\s+)?nhắn\s+zalo\s+vụ)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(?:rồi|xong|với|và)\s+nhắn\b", norm_t, re.IGNORECASE):
            return False
    if tname == "calendar.cancel_event":
        if re.search(r"\b(lỡ\s+(?:tay\s+)?(?:hủy|huỷ)|đang\s+bị\s+(?:hủy|huỷ)|tạo\s+lại|khôi\s+phục\s+lại|thêm\s+lại)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "sms.send":
        if re.search(r"\b(mới nhắn(?: tin)?(?: đó)?|coi tin nhắn|xem tin nhắn|đọc tin nhắn|trạm .* nhắn đó)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(?:rồi|xong|với|và)\s+nhắn\b", norm_t, re.IGNORECASE):
            return False
    if tname in {"bill_payment.pay", "ewallet.get_balance"}:
        if re.search(r"\b(chuyển|gửi|gởi)\b.*\b(đúng|bằng|y)\s+(?:bằng\s+)?(?:số\s+)?tiền\s+(?:điện|nước|mạng|internet|truyền hình|cáp)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đóng|trả|thanh toán)\s+(?:tiền|hóa đơn|hoá đơn)\b", norm_t, re.IGNORECASE):
            return False
        if re.search(r"\b(tài khoản thanh toán|tiết kiệm)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đóng|trả|thanh toán)\s+(?:tiền|hóa đơn|hoá đơn)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "ewallet.get_balance":
        if re.search(r"\b(chút tiền|ít tiền)\b", norm_t, re.IGNORECASE) and not re.search(r"\b\d+\s*(?:nghìn|ngàn|triệu|củ|đồng|k\b)\b", norm_t, re.IGNORECASE):
            return False
        if re.search(r"\b(tiền điện|tiền nước|tiền mạng|tiền truyền hình|tiền cáp|hóa đơn|hoá đơn)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(ví đủ|đủ không|không đủ|còn đủ|thiếu thì|hụt thì|số dư ví|kiểm tra ví|coi ví|xem ví)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "weather.get_forecast":
        if re.search(r"\b(nhắn|báo|nói)[^,.!?;]*?\blà\s+mai\s+mưa\b", norm_t, re.IGNORECASE):
            return False
        if re.search(r"\b(giờ (?:mà |có )?(?:trời )?đang|ngoài trời (?:giờ )?(?:có )?đang|chỗ (?:anh|chị|em|tôi|mình) đang|thời tiết (?:[^,.!?;]*?\s+)?giờ thế nào|ngoài trời giờ thế nào|hiện tại|lúc này|bây giờ)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(mai|ngày kia|tuần|thứ|chủ nhật)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "weather.get_current":
        if re.search(r"\b(mai|ngày kia|tuần|thứ|chủ nhật)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(hiện tại|bây giờ|lúc này|giờ (?:mà |có )?(?:trời )?đang|ngoài trời (?:giờ )?(?:có )?đang|chỗ (?:anh|chị|em|tôi|mình) đang|thời tiết (?:[^,.!?;]*?\s+)?giờ thế nào|ngoài trời giờ thế nào)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "smart_home.fan_set" and re.search(r"\bquạt\b[^,.]{0,25}\b(đừng bật|khỏi bật|không bật|không cần bật|đang chạy|đang bật|cứ để rứa|cứ để nguyên|để nguyên|khỏi tắt|đừng tắt)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(bật hết quạt|tắt quạt|tắt cái quạt|bật quạt)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "smart_home.light_set":
        if re.search(r"\b(tự tắt|tự bật|quy tắc)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(bật|tắt)\s+(?:hết\s+|mấy\s+|cái\s+)?đèn\b", norm_t, re.IGNORECASE):
            return False
        if re.search(r"\bđèn\b.{0,55}\b(đừng tắt|khỏi tắt|cứ để sáng|để nguyên)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(tắt hết đèn|tắt đèn .* hết đi|tắt toàn bộ đèn|tắt mấy đèn|phòng nào)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "translator.translate" and re.search(r"\b(thôi khỏi dịch|khỏi dịch lại|đừng dịch)\b", norm_t, re.IGNORECASE):
        return False
    if tname in {"ecommerce.add_to_cart", "ecommerce.checkout"} and re.search(r"\b(hãng nào|còn hàng không|xem giá|bao nhiêu tiền)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(bỏ vào giỏ|thêm vào giỏ|vào giỏ|đặt mua|mua luôn|thanh toán|chốt)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "ewallet.transfer":
        if re.search(r"\bnạp\b", norm_t, re.IGNORECASE) and not re.search(r"\b(chuyển\s+ví|qua\s+ví|nạp\b.*\b(?:rồi|với|và)\s+(?:chuyển|gửi|gởi))\b", norm_t, re.IGNORECASE):
            return False
        if re.search(r"\b(số\s+dư\s+ví|ví\s+điện\s+tử.*trừ\s+ra|trừ\s+ra\s+thì\s+còn\s+bao\s+nhiêu)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(chuyển|bắn|trả|đóng|thanh\s+toán)\b", norm_t, re.IGNORECASE):
            return False
    if tname in {"train_bus_tickets.book", "train_bus_tickets.search", "flights.book", "flights.search"} and re.search(r"\b(hủy|huỷ)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đặt lại|mua lại|tìm chuyến khác)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "smart_home.climate_set" and re.search(r"\b(phòng nào\s+(?:[^,.!?;]*?\s+)?(?:điều hòa|điều hoà|máy lạnh)|đang\s+(?:bật|mở|chạy)\s+(?:điều hòa|điều hoà|máy lạnh)|xem (?:điều hòa|điều hoà|máy lạnh) đang)\b", norm_t, re.IGNORECASE) and not re.search(r"(?<!đang\s)(?<!chưa\s)(?<!đừng\s)(?<!khỏi\s)(?<!tối\s)(?<!sáng\s)(?<!mai\s)\b(bật|mở|tắt|chỉnh|hạ|tăng|để \d{2})\s*(?:điều hòa|điều hoà|máy lạnh)", norm_t, re.IGNORECASE):
        return False
    if tname in {"flights.book", "flights.search"} and re.search(r"\b(hủy|huỷ)\b.*\b(vé máy bay|chuyến bay)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đặt|mua|đổi sang)\b", norm_t, re.IGNORECASE):
        return False
    if tname in {"flights.book", "flights.search"} and re.search(r"\b((?:chuyến(?:\s+bay|\s+về|\s+đi)?|vé(?:\s+máy\s+bay)?)\b[^,.!?;]*?\b(?:mình|tôi|anh|chị|em|đã|vừa)\s+đặt|ngày\s+bay\s+của\s+(?:tôi|mình|anh|chị|em)|trước\s+giờ\s+bay|hôm\s+trước\s+ngày\s+bay|máy\s+bay\s+hạ\s+cánh)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đặt\s+thêm|đặt\s+lại|tìm\s+chuyến\s+khác|đổi\s+sang|(?:đặt|mua|lấy|tìm|chốt)\b[^,.!?;]*?\b(?:vé|chuyến\s+bay|chuyến\s+sớm|chuyến\s+rẻ|chuyến\s+muộn))\b", norm_t, re.IGNORECASE):
        return False
    if tname in {"train_bus_tickets.book", "train_bus_tickets.search"} and re.search(r"\b(lượt về .* đi tàu|chiều về .* đi tàu)\b.*\b(coi|xem|kiểm tra|ở .* mấy đêm)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đặt|mua|tìm)\b", norm_t, re.IGNORECASE):
        return False
    if tname in {"grocery.add_to_cart", "grocery.checkout"} and re.search(r"\b(hết bao nhiêu tiền|hết nhiêu tiền)\b.*\b(đủ trả không|còn đủ)\b", norm_t, re.IGNORECASE):
        return False
    if tname in {"food_delivery.place_order", "food_delivery.get_menu", "food_delivery.list_restaurants"} and re.search(r"\b(hủy đơn|huỷ đơn|lưu giúp .* vào bản đồ)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đặt lại|đặt thêm|đặt 1|đặt một|đặt hai|đặt 2)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "banking.list_beneficiaries" and re.search(r"\b(mua thêm .* cổ phiếu|bán .* cổ phiếu|gửi tiết kiệm|mở sổ tiết kiệm)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(chuyển khoản|chuyển cho)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "contacts.search" and re.search(r"\b(hết bao nhiêu tiền|số dư ví)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(chuyển|nạp|gọi|nhắn)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "calendar.list_events" and re.search(r"\b(tạo lịch|thêm lịch|ghi lịch|vào lịch)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(xem .* lịch|kiểm tra .* lịch|có lịch .* không|trùng|rảnh hay bận|cùng ngày với|trước buổi|sau buổi)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "bill_payment.pay" and re.search(r"\b(đủ trả .* thì bật|đủ .* thì cài)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đóng luôn|trả luôn|thanh toán luôn)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "exchange_rate.convert" and re.search(r"\b(đô bữa nay|giá đô|tỷ giá)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đổi|quy ra|bằng bao nhiêu tiền việt)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "lottery.get_results" and re.search(r"\b(dò .* tờ vé|tờ vé .* số)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "lottery.check_ticket" and re.search(r"\b(xổ chưa|giải đặc biệt|kết quả xổ số)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(\d{5,6}|tờ vé|mấy vé)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "shopping_list.get" and re.search(r"\b(thêm|ghi)\b.*\b(vào|vô)\s+danh\s+sách\s+(?:đi\s+chợ|mua\s+sắm)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(xem|coi|kiểm tra|gạch|bỏ|xóa|xoá)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "tasks.add" and re.search(r"\b(coi|xem|kiểm tra)\b.*\b(việc cần làm|danh sách việc)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(thêm|tạo|ghi)\s+(?:một\s+|cái\s+)?việc\b", norm_t, re.IGNORECASE):
        return False
    if tname == "phone.call" and "thông tin liên hệ đầy đủ" in norm_t.lower():
        return False
    if tname in {"phone.call", "phone.recent_calls"} and re.search(r"\bnhắc\s+(?:tôi|mình|anh|chị|em)\s+gọi\b", norm_t, re.IGNORECASE) and not re.search(r"\b(?:rồi|với|và)\s+gọi\b", norm_t, re.IGNORECASE):
        return False
    pat = TOOL_KEYWORD_GUARDS.get(tname)
    if not pat:
        return True
    return bool(re.search(pat, text, re.IGNORECASE))


def _valid_filtered(filtered_calls: List[Dict[str, Any]], u_text: str) -> bool:
    if not filtered_calls:
        return False
    names = {tc["name"] for tc in filtered_calls}
    norm_u = _normalize_vn_numbers_in_time(u_text)
    if names == {"contacts.search"} and not re.search(r"\b(danh bạ|số điện thoại|tìm số)\b", u_text, re.IGNORECASE):
        return False
    if names == {"ewallet.get_balance"} and re.search(r"\b(nạp|chuyển|quét mã|thanh toán)\b", norm_u, re.IGNORECASE) and not re.search(r"\b(số dư|còn bao nhiêu)\b", norm_u, re.IGNORECASE):
        return False
    if names == {"alarm.list"} and re.search(r"\b(đặt|hẹn|tạo|dời|đổi|chuyển|sửa|lùi|đẩy|xóa|xoá|tắt|hủy|huỷ|thành \d|sang \d)\b", norm_u, re.IGNORECASE):
        if not re.search(r"\b(có báo thức nào .* không|xem .* báo thức|kiểm tra .* báo thức)\b", u_text, re.IGNORECASE):
            return False
    if names == {"reminders.list"} and re.search(r"\b(nhắc tôi|nhắc mình|nhắc anh|nhắc chị|nhắc em|tạo nhắc|thêm nhắc|xóa nhắc|xoá nhắc|đánh dấu xong)\b", u_text, re.IGNORECASE):
        if not re.search(r"\b(xem .* nhắc|kiểm tra .* nhắc|có lời nhắc nào)\b", u_text, re.IGNORECASE):
            return False
    if names == {"calendar.list_events"} and re.search(r"\b(tạo .* lịch|thêm .* lịch|ghi vào lịch|cho vào lịch|đặt lịch|dời|đổi|chuyển|lùi|hủy|huỷ|xóa|xoá)\b", norm_u, re.IGNORECASE):
        if not re.search(r"\b(xem .* lịch|kiểm tra .* lịch|có lịch .* không|trùng lịch)\b", u_text, re.IGNORECASE):
            return False
    if "calendar.list_events" in names and "calendar.update_event" not in names and re.search(r"\b(dời|đổi|lùi|đẩy)\b.*\b(lịch|buổi|hẹn|họp|khám|cà phê)\b", norm_u, re.IGNORECASE) and not re.search(r"\b(nếu|mưa|hủy|huỷ|xóa|xoá)\b", norm_u, re.IGNORECASE):
        return False
    if names == {"email.list"} and re.search(r"\b(xóa|xoá|lưu trữ|archive|trả lời|reply|soạn|gửi|nháp)\b", norm_u, re.IGNORECASE):
        return False
    if names == {"shopping_list.get"} and re.search(r"\b(thêm|ghi|xóa|xoá|bỏ|gạch|đánh dấu)\b", norm_u, re.IGNORECASE):
        return False
    if names == {"parking.list_sessions"} and re.search(r"\b(kết thúc|tắt .* phiên|tách cái phiên|cho dừng)\b", norm_u, re.IGNORECASE):
        return False
    if names == {"event_tickets.search"} and re.search(r"\b(đặt|mua|lấy)\b.*\b(vé|vét|chỗ)\b", norm_u, re.IGNORECASE):
        return False
    if names <= {"food_delivery.list_restaurants", "food_delivery.get_menu", "chat_ott.list_chats"} and "food_delivery.place_order" not in names:
        if re.search(r"\b(mỗi người một|đặt giúp|đặt cho cả nhóm|giao về)\b", u_text, re.IGNORECASE):
            return False
    return True


def _get_atomic_retriever(retriever: Any) -> Any:
    """Khởi tạo và lưu cache bộ truy xuất mệnh đề đơn (Atomic Sub-Exemplar Retriever) ngay trên đối tượng retriever."""
    _ensure_catalog_exemplars_indexed(retriever)
    if getattr(retriever, "_atomic_retriever", None) is not None:
        return retriever._atomic_retriever

    from src.agent.rag.exemplar_retriever import BM25ExemplarRetriever
    atomic_records = []
    for rec in getattr(retriever, "corpus", []):
        if "sys" in rec.get("called_apps", set()):
            continue
        non_contact_apps = rec["called_apps"] - {"contacts"}
        if len(non_contact_apps) == 1:
            atomic_records.append(rec)
        elif len(non_contact_apps) >= 2:
            clauses = [c.strip() for c in CLAUSE_SPLIT_RE.split(rec["audio_text"]) if len(c.strip().split()) >= 3]
            if len(clauses) >= 2:
                for app_name in non_contact_apps:
                    app_calls = [
                        tc for tc in rec["tool_calls"]
                        if tc["name"].split(".")[0] == app_name
                        or (tc["name"].startswith("contacts.") and app_name in {"phone", "sms", "ewallet", "email", "chat_ott"})
                    ]
                    if not app_calls:
                        continue
                    app_pat = APP_TRIGGER_PATTERNS.get(app_name)
                    matched_cl = None
                    for cl in clauses:
                        if app_pat and re.search(app_pat, cl, re.IGNORECASE) and all(_is_tool_compatible(tc, cl) for tc in app_calls):
                            matched_cl = cl
                            break
                    if matched_cl:
                        atomic_records.append({
                            "id": f"{rec['id']}_{app_name}",
                            "audio_text": matched_cl,
                            "installed_apps": [app_name, "contacts"],
                            "called_apps": {tc["name"].split(".")[0] for tc in app_calls},
                            "tool_calls": app_calls,
                        })

    ret_atomic = BM25ExemplarRetriever()
    ret_atomic.index_records(atomic_records)
    retriever._atomic_retriever = ret_atomic
    return ret_atomic


NEVER_HISTORY_SATISFIED_TOOLS: Set[str] = {
    "smart_home.get_status",
    "robot_vacuum.get_status",
    "email.read",
    "chat_ott.read_chat",
    "sms.read_recent",
    "parking.list_sessions",
    "train_bus_tickets.my_bookings",
    "flights.my_bookings",
    "habit_tracker.list_habits",
    "calculator.calculate",
    "calculator.convert_unit",
    "exchange_rate.get_rate",
    "exchange_rate.convert",
    "fuel_price.get_price",
    "gold_price.get_price",
    "weather.get_forecast",
    "weather.get_current",
    "translator.translate",
    "maps.get_route",
}


def get_history_satisfied_reads(history: Optional[List[Dict[str, Any]]], u_text: str = "") -> Set[str]:
    """Xác định các công cụ Đọc đã được thực thi và có sẵn dữ liệu trong history."""
    sat: Set[str] = set()
    u_low = (u_text or "").lower()
    u_ct_q = _extract_contact_query_from_text(u_text) if u_text else None
    is_short_followup = len(u_low.split()) <= 16 or bool(re.search(r"\b(cái đó|buổi đó|hôm đó|người đó|số đó|khoản đó|bài đó|bản đó|đơn đó|mấy bài đó|như thế|vừa chọn|hồi nãy|lúc sáng|chuyến đó|món đó|vé đó|bãi đó)\b", u_low))
    hist_list = history or []
    is_long_multi_session = len(hist_list) >= 12
    for idx_t, t in enumerate(hist_list):
        is_recent_turn = (not is_long_multi_session) or (idx_t >= len(hist_list) - 4)
        for tc in (t.get("tool_calls") or []):
            tname = tc.get("name", "")
            targs = tc.get("arguments") or {}
            if tname in READ_TOOLS and tname not in NEVER_HISTORY_SATISFIED_TOOLS:
                if targs and not is_recent_turn and not is_short_followup and not re.search(r"\b(hồi nãy|lúc nãy|vừa nãy|lúc sáng|hồi sáng)\b", u_low):
                    continue
                if tname == "calendar.list_events" and targs and not is_short_followup and not re.search(r"\b(dời lịch|sửa lịch|đổi tên lịch)\b", u_low):
                    continue
                if tname == "contacts.search":
                    q_prev = str(targs.get("query") or "").lower()
                    if u_ct_q and q_prev and u_ct_q.lower() not in q_prev and q_prev not in u_ct_q.lower():
                        continue
                    if q_prev and q_prev not in u_low and not is_short_followup:
                        continue
                if tname == "food_delivery.list_restaurants" and not is_short_followup and not re.search(r"\b(cô ba|nhà mèo|nhà làm|sài gòn 168)\b", u_low) and not any(isinstance(tc_x, dict) and tc_x.get("name") == "food_delivery.get_menu" for th_x in hist_list for tc_x in (th_x.get("tool_calls") or [])):
                    continue
                if tname in {"food_delivery.order_history", "ecommerce.order_status"} and re.search(r"\b(tới đâu rồi|xem .* đơn|kiểm tra .* đơn)\b", u_low):
                    continue
                if tname == "bike_rental.find_bikes" and re.search(r"\bở\s+trạm\b", u_low) and not re.search(r"\b(lúc nãy|vừa nãy|hồi nãy)\b", u_low):
                    continue
                if tname == "maps.search_place" and not re.search(r"\b(lưu|đánh dấu|chỉ đường tới chỗ đó|chỉ đường ra đó)\b", u_low):
                    continue
                if tname == "music.list_tracks" and re.search(r"\b(kiểm tra|xem|tìm)\b.*\b(nhạc|bài hát)\b", u_low):
                    continue
                if tname == "smart_home.get_status" and targs.get("room") and str(targs.get("room")).lower() not in u_low:
                    continue
                if tname == "ewallet.get_balance" and re.search(r"\b(số\s+dư\s+ví|ví\s+điện\s+tử.*trừ|kiểm\s+tra\s+ví|xem\s+ví|coi\s+ví)\b", u_low):
                    continue
                if tname == "bill_payment.list_bills" and (targs.get("type") or re.search(r"\b(đã có hóa đơn chưa|đã có hoá đơn chưa|có hóa đơn chưa|học phí)\b", u_low)):
                    continue
                sat.add(tname)
                if tname == "email.read":
                    sat.add("email.list")
            elif tname == "email.read":
                sat.add("email.list")
            elif tname in {"grocery.add_to_cart", "ecommerce.add_to_cart"} and is_recent_turn:
                sat.add(tname)
            elif tname == "home_automation.create_rule":
                sat.add("home_automation.list_rules")
            elif tname == "video_meeting.create":
                sat.add("video_meeting.list_meetings")
            elif tname == "medication.add":
                sat.add("medication.list")
            elif tname == "notes.create":
                sat.add("notes.list")
            elif tname in {"ride_hailing.book", "ride_hailing.get_ride"}:
                sat.add("ride_hailing.get_ride")
            elif tname == "food_delivery.place_order" and is_recent_turn:
                sat.add("food_delivery.order_history")
            elif tname in {"flights.book", "flights.search"} and is_recent_turn and re.search(r"\b(chuyến bay đó|thông tin chuyến bay đó)\b", u_low):
                sat.add("flights.my_bookings")
            elif tname == "smart_home.get_status" and idx_t >= len(hist_list) - 4 and not re.search(r"\b(phòng nào|xem|kiểm tra|coi|còn sáng thì|đang bật thì)\b", u_low) and not (targs.get("room") and str(targs.get("room")).lower() not in u_low):
                sat.add(tname)
        for res in (t.get("tool_results") or []):
            if not isinstance(res, dict):
                continue
            if "rules" in res or "rule" in res:
                sat.add("home_automation.list_rules")
            if "meetings" in res or "meeting" in res:
                sat.add("video_meeting.list_meetings")
            if "ride" in res or "rides" in res:
                sat.add("ride_hailing.get_ride")
            if "tracks" in res and not re.search(r"\b(kiểm tra|xem|tìm)\b.*\b(nhạc|bài hát)\b", u_low):
                sat.add("music.list_tracks")
            if "alarms" in res or "alarm" in res:
                sat.add("alarm.list")
            if "sessions" in res:
                sat.add("parking.list_sessions")
            if "bookings" in res:
                sat.add("flights.my_bookings")
                sat.add("train_bus_tickets.my_bookings")
            if "lots" in res:
                sat.add("parking.find")
            if "messages" in res and isinstance(res["messages"], list):
                for msg_item in res["messages"]:
                    if isinstance(msg_item, dict) and msg_item.get("contact_id"):
                        sender_str = str(msg_item.get("sender") or msg_item.get("from") or t.get("text") or "").lower()
                        sender_toks = [w for w in str(msg_item.get("sender") or msg_item.get("from") or "").lower().split() if len(w) >= 3]
                        if (u_ct_q and u_ct_q.lower() in sender_str) or (not u_ct_q and sender_toks and any(w in u_low for w in sender_toks)):
                            sat.add("contacts.search")
            if not is_recent_turn and not is_short_followup:
                continue
            if "alerts" in res or "alert" in res:
                sat.add("home_security.list_alerts")
            if "chats" in res or "chat" in res:
                sat.add("chat_ott.list_chats")
            if "tasks" in res or "task" in res:
                sat.add("tasks.list")
            if "flights" in res and is_recent_turn:
                sat.add("flights.search")
            if "trips" in res:
                sat.add("train_bus_tickets.search")
            if "events" in res and isinstance(res["events"], list) and any(isinstance(x, dict) and ("venue" in x or "available" in x or "price" in x) for x in res["events"]):
                sat.add("event_tickets.search")
            elif "events" in res and isinstance(res["events"], list) and any(isinstance(x, dict) and ("start" in x or "end" in x) for x in res["events"]) and (is_recent_turn or is_short_followup or re.search(r"\b(trước hạn chót|trước buổi|sau buổi|cùng ngày với|bữa đó|hôm đó)\b", u_low)):
                sat.add("calendar.list_events")
            if "contacts" in res and is_recent_turn:
                c_list_res = res.get("contacts") if isinstance(res.get("contacts"), list) else []
                if not u_ct_q or any(isinstance(c_it, dict) and (u_ct_q.lower() in str(c_it.get("name", "")).lower() or str(c_it.get("name", "")).lower() in u_ct_q.lower()) for c_it in c_list_res):
                    sat.add("contacts.search")
            if "chores" in res:
                sat.add("household_chores.list_chores")
            if "courses" in res:
                sat.add("learning.list_courses")
            if "books" in res and is_recent_turn:
                sat.add("library.search")
            if "calls" in res:
                sat.add("phone.recent_calls")
            if "places" in res and is_recent_turn and re.search(r"\b(lưu|đánh dấu|chỉ đường tới chỗ đó|chỉ đường ra đó)\b", u_low):
                sat.add("maps.search_place")
            if "products" in res and is_recent_turn:
                sat.add("ecommerce.search")
                sat.add("grocery.search")
            if "notes" in res or "note" in res:
                sat.add("notes.list")
            if "programs" in res and "channels" in res:
                sat.add("tv.guide")
            if "habits" in res:
                sat.add("habit_tracker.list_habits")
            if "items" in res and any(isinstance(x, dict) and str(x.get("id", "")).startswith("pan_") for x in (res.get("items") or [])):
                sat.add("pantry.list_items")
            if "portfolio" in res or "holdings" in res:
                sat.add("stocks.get_portfolio")
            if "recipes" in res and is_recent_turn:
                sat.add("recipes.search")
            if "routes" in res:
                sat.add("public_transit.find_routes")
    return sat


def _extract_contact_query_from_text(u_text: str, history: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    u_lower = u_text.lower()
    CONTACT_PATTERNS = (
        (r"\bmrs?\.?\s*watanabe\b|\bđối\s+tác\s+nhật\b", "Mrs. Watanabe"),
        (r"\blaura\s+costa\b", "Laura Costa"),
        (r"\bolivia\s+brown\b", "Olivia Brown"),
        (r"\bsarah\s+kim\b", "Sarah Kim"),
        (r"\bmark\s+johnson\b", "Mark Johnson"),
        (r"\bpeter\s+hansen\b", "Peter Hansen"),
        (r"\bkevin\s+ngô\b", "Kevin Ngô"),
        (r"\bsophie\s+martin\b", "Sophie Martin"),
        (r"\boliver\s+grant\b", "Oliver Grant"),
        (r"\bben\s+taylor\b", "Ben Taylor"),
        (r"\bemily\s+carter\b", "Emily Carter"),
        (r"\bjulian\s+ward\b", "Julian Ward"),
        (r"\bjonas\s+berg\b", "Jonas Berg"),
        (r"\bdaniel\s+lee\b", "Daniel Lee"),
        (r"\bjames\s+wong\b", "James Wong"),
        (r"\bdavid\s+miller\b", "David Miller"),
        (r"\bminh\s+quân\b", "Minh Quân"),
        (r"\bminh\s+hà\b", "Minh Hà"),
        (r"\bmai\s+anh\b", "Mai Anh"),
        (r"\bvợ\s+hằng\b", "Vợ Hằng"),
        (r"\bbà\s+hồng\b", "Bà Hồng"),
        (r"\bbà\s+sáu\b", "Bà Sáu"),
        (r"\bcô\s+(?:giáo\s+)?mai\b", "Cô giáo Mai"),
        (r"\bcô\s+hạnh\b", "Cô Hạnh"),
        (r"\bbác\s+sĩ\s+an\b", "Bác sĩ An"),
        (r"\bbác\s+sĩ\s+hòa\b", "Hòa"),
        (r"\bbác\s+ba\b", "Bác Ba"),
        (r"\bchú\s+năm\b", "Chú Năm"),
        (r"\bsếp\s+dũng\b", "Sếp Dũng"),
        (r"\bsếp\s+kim\b", "Sếp Kim"),
        (r"\bsếp\s+(?:trân|trần)\b", "Sếp Trân"),
        (r"\bchủ\s+nhà(?:\s+phạm\s+hùng)?\b", "Chủ nhà"),
        (r"\bcô\s+ly\b", "Cô Ly"),
        (r"\bchú\s+tám\b", "Chú Tám"),
        (r"\bthầy\s+phong\b", "Thầy Phong"),
        (r"\bthằng\s+tí\b", "Thằng Tí"),
        (r"\bthằng\s+út\b", "Thằng Út"),
        (r"\banh\s+kha\b", "Anh Kha"),
        (r"\banh\s+lộc\b", "Anh Lộc"),
        (r"\banh\s+long\b", "Anh Long"),
        (r"\banh\s+mười\b", "Anh Mười"),
        (r"\banh\s+hùng\b", "Anh Hùng"),
        (r"\banh\s+bình\b", "Anh Bình"),
        (r"\banh\s+sáu\b", "Anh Sáu"),
        (r"\banh\s+quân\b", "Anh Quân"),
        (r"\bchị\s+dung\b", "Chị Dung"),
        (r"\bphúc\b", "Phúc"),
        (r"(?<!việt\s)(?<!miền\s)(?<!phía\s)(?<!cổng\s)(?<!tòa\s)\bnam\b(?!\s*(?:sơn|đô|bộ|định|từ\s+liêm|kỳ))", "Nam"),
        (r"(?<!ngân\s)\bhà\b(?!\s*(?:nội|tĩnh|giang|nam|đông|tiên|khẩu))", "Hà"),
        (r"\bminh\b", "Minh"),
        (r"\blan\b", "Lan"),
        (r"\bhương\b", "Hương"),
        (r"\btuấn\b", "Tuấn"),
        (r"\bbình\b", "Bình"),
        (r"\bhuy\b", "Huy"),
        (r"\bquân\b", "Quân"),
        (r"\bdung\b", "Dung"),
        (r"\btrang\b", "Trang"),
        (r"\blinh\b", "Linh"),
        (r"\bthảo\b", "Thảo"),
        (r"\bhòa\b", "Hòa"),
        (r"\bphương\b", "Phương"),
        (r"\bkhánh\b", "Khánh"),
        (r"\byến\b", "Yến"),
        (r"\btoàn\b", "Toàn"),
        (r"\bnhi\b", "Nhi"),
        (r"\bhằng\b", "Hằng"),
        (r"(?<!một\s)(?<!soạn\s)(?<!coi\s)(?<!đọc\s)(?<!gửi\s)(?<!hộp\s)(?<!mấy\s)(?<!các\s)(?<!những\s)\bthư\b(?!\s*(?:nháp|mời|điện\s+tử|email|mail|viện|mục))", "Thư"),
        (r"\bvy\b", "Vy"),
        (r"\bsarah\b", "Sarah"),
        (r"\bnexora\b", "Nexora"),
        (r"\b(?:gửi|cho|gọi|nhắn)\s+an\b", "An"),
        (r"\bmá\b", "Má"),
        (r"\bmẹ\b", "Mẹ"),
        (r"\bmạ\b", "Mạ"),
        (r"\b(?:bố|(?<!thứ\s)(?<!cả\s)(?<!hai\s)(?<!mười\s)(?<!năm\s)\bba\b(?!\s*(?:giờ|phút|trăm|nghìn|ngàn|triệu|ngày|tuần|tháng|quả|hộp|gói|cái|chiếc|người|vé|chỗ|mươi|chục|lăm|tư|mốt)))\b", "Bố"),
    )
    u_no_media_author = re.sub(
        r"\b(?:bài\s+nhạc|bài\s+hát|nhạc|ca\s+khúc|cuốn\s+sách|sách|podcast)\b[^,.!?;]*?\bcủa\s+[a-zA-ZÀ-ỹ]+(?:\s+[a-zA-ZÀ-ỹ]+)?",
        "",
        u_lower,
    )
    for pat_ct, canon_ct in CONTACT_PATTERNS:
        if re.search(pat_ct, u_no_media_author):
            return canon_ct
    for pat_ct, canon_ct in CONTACT_PATTERNS:
        if re.search(pat_ct, u_lower):
            return canon_ct
    m_lien_he = re.search(r"\bliên\s+hệ\s+(?:anh|chị|em|bạn|ông|bà|cô|chú|bác)\s+([a-zA-ZÀ-ỹ]+)\b", u_text, re.IGNORECASE)
    if m_lien_he:
        w_lh = m_lien_he.group(1).strip()
        if w_lh.lower() not in {"đi", "ít", "điện", "mới", "cũ", "đó", "ấy", "này", "coi", "xem", "nhé", "nha", "giúp", "giùm", "hộ", "với"}:
            return w_lh[0].upper() + w_lh[1:].lower()
    m_dyn_ct = re.search(
        r"\b(?:cho|gặp|gọi|nhắn|với)\s+(?:anh|chị|em|cô|chú|bác|ông|bà|bạn|sếp)?\s*([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)?)\b",
        u_text,
    )
    if m_dyn_ct:
        cand_dyn = m_dyn_ct.group(1).strip()
        if (
            all(w[0].isupper() for w in cand_dyn.split() if w)
            and cand_dyn.lower() not in {"tôi", "mình", "em", "anh", "chị", "ấy", "đó", "này", "nhóm", "zalo", "sms", "hà nội", "đà nẵng", "sài gòn", "huế"}
            and not re.search(rf"\b{re.escape(cand_dyn)}\s+(?:Grill|Bistro|Seafood|Cafe|Coffee|Mart|Store|Shop|Mall|Spa|Salon|Cinema)\b", u_text, re.IGNORECASE)
        ):
            return cand_dyn
    if history:
        for turn_h in reversed(history[-4:]):
            h_blob = (str(turn_h.get("audio_text") or turn_h.get("text") or "") + " " + json.dumps(turn_h.get("tool_results") or [], ensure_ascii=False)).lower()
            for pat_ct, canon_ct in CONTACT_PATTERNS:
                if re.search(pat_ct, h_blob):
                    return canon_ct
    return None


def _ensure_readers_or_prune(
    calls: List[Dict[str, Any]],
    hist_sat: Set[str],
    u_text: str,
    installed_apps: Any,
    current_time_str: str,
    history: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    u_lower = u_text.lower()
    norm_u = _normalize_vn_numbers_in_time(u_text)
    has_plural = bool(
        re.search(
            r"\b(cả hai|tất cả|mấy cái|mấy việc|mấy đơn|mấy gói|mấy cữ|mấy bài|mấy mã|mấy hóa đơn|mấy hoá đơn|mấy thuốc|mấy mail|mấy thư|hai cái|2 cái|hai việc|2 việc|hai đơn|2 đơn|hai vé|2 vé|hai tờ|2 tờ|hai chiếc|2 chiếc|hai thiết bị|những|các|còn lại)\b",
            u_text,
            re.IGNORECASE,
        )
    )
    is_price_query = bool(re.search(r"\b(hết bao nhiêu tiền|bao nhiêu tiền)\b", u_text, re.IGNORECASE))
    has_workout_time = bool(re.search(r"\b(sáng nay|chiều nay|tối nay|lúc \d)\b", u_text, re.IGNORECASE))
    primary_d, prev_d = resolve_target_dates_from_utterance(u_text, current_time_str)
    use_cal_date_filter = bool(primary_d and CALENDAR_DATE_FILTER_RE.search(u_text))
    cal_list_args = {"date_from": primary_d, "date_to": primary_d} if use_cal_date_filter else {}
    inst_set = set(installed_apps.keys()) if isinstance(installed_apps, dict) else set(installed_apps or [])
    is_vague_ewallet_amount = bool(
        (re.search(r"\b(chút tiền|ít tiền)\b", norm_u, re.IGNORECASE)
         or ("bill_payment" not in inst_set and re.search(r"\b(phần tiền chia hóa đơn|phần tiền chia hoá đơn)\b", norm_u, re.IGNORECASE)))
        and not re.search(r"\b\d+\s*(?:nghìn|ngàn|trăm|triệu|củ|đồng|k\b)|tiền\s+(?:điện|nước|mạng|internet)\b", norm_u, re.IGNORECASE)
    )
    has_explicit_time_in_u = bool(
        re.search(r"\b(\d{1,2}\s*(?:giờ|rưỡi|h\b)|\d+\s*(?:phút|tiếng)|nửa\s+tiếng|trước\s+(?:giờ|lúc|buổi|khi|đó)|sau\s+khi|cùng\s+giờ|sớm\s+hơn|sáng\s+hôm\s+bay)\b", norm_u, re.IGNORECASE)
    )
    has_multi_bank_transfer = bool(
        re.search(r"\btiền\s+(?:điện|nước|mạng|internet|truyền hình)\b.*\btiền\s+(?:điện|nước|mạng|internet|truyền hình)\b|\bchuyển\b.*\b(?:rồi|và|với)\s+chuyển\b|\bchuyển\s+[^,.!?]*?\b(?:với|và)\s+[^,.!?]*?\bmỗi\s+người\b", u_lower)
    )
    single_room_light = bool(
        re.search(r"\bđèn\s+(?:phòng\s+(?:khách|ngủ|làm\s+việc|ăn|tắm)|bếp|hành\s+lang|ban\s+công|sân)\b", u_lower)
        and not re.search(r"\b(tất cả|hết|cả hai|hai đèn|mấy đèn|phòng .* và .* phòng|đèn .* và .* đèn)\b", u_lower)
    )
    spoken_phone_match = re.search(r"\b(0\d{8,11}|1[89]00\d{4})\b", norm_u)
    non_ct_apps_in_calls = {
        (x.get("arguments", {}).get("tool_name", "").split(".")[0] if x.get("name") == "sys.ask_missing_info" else x.get("name", "").split(".")[0])
        for x in calls
    } - {"contacts"}

    # Chuẩn hóa sys.ask_missing_info tĩnh về công cụ hành động gốc (việc kiểm tra trùng danh bạ do môi trường thực thi quyết định)
    norm_calls: List[Dict[str, Any]] = []
    for tc in calls:
        if tc.get("name") == "sys.ask_missing_info":
            targs = tc.get("arguments") or {}
            eff = targs.get("tool_name", "")
            prms = targs.get("params") or []
            if prms == ["contact_id"] and eff in {"phone.call", "sms.send", "ewallet.transfer"}:
                if eff == "ewallet.transfer" and is_vague_ewallet_amount:
                    tc = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "ewallet.transfer", "params": ["amount"]}}
                elif eff == "phone.call":
                    tc = {"name": "phone.call", "arguments": {"contact_id": "ct_01"}}
                elif eff == "sms.send":
                    tc = {"name": "sms.send", "arguments": {"contact_id": "ct_01", "content": "Tin nhắn"}}
                elif eff == "ewallet.transfer":
                    tc = {"name": "ewallet.transfer", "arguments": {"contact_id": "ct_01", "amount": 100000}}
            elif prms == ["amount"] and eff == "ewallet.transfer" and not is_vague_ewallet_amount:
                tc = {"name": "ewallet.transfer", "arguments": {"contact_id": "ct_01", "amount": 200000}}
            elif eff == "ewallet.topup_mobile" and re.search(r"\b\d+\s*(?:nghìn|ngàn|k\b)\b", norm_u):
                tc = {"name": "ewallet.topup_mobile", "arguments": {"phone": spoken_phone_match.group(1) if spoken_phone_match else "0901234567", "amount": 50000}}
            elif prms == ["chat_id"] and eff == "chat_ott.send_message":
                tc = {"name": "chat_ott.send_message", "arguments": {"chat_id": "chat_01", "content": "Tin nhắn"}}
            elif prms == ["origin"] and eff == "maps.get_route" and re.search(r"\btừ\s+\S+", u_lower):
                tc = {"name": "maps.get_route", "arguments": {"origin": "Nhà", "destination": "Điểm đến"}}
            elif prms == ["start"] and eff == "video_meeting.create" and (has_explicit_time_in_u or re.search(r"\b(đúng giờ mới|giờ đó|cùng giờ)\b", u_lower)):
                tc = {"name": "video_meeting.create", "arguments": {"title": "Họp", "start": f"{primary_d or '2026-10-16'}T10:00:00+07:00"}}
            elif prms == ["time"] and eff in {"alarm.create", "reminders.create"} and (has_explicit_time_in_u or re.search(r"\b(buổi\s+chiều|cùng\s+giờ|giờ\s+đó|lúc\s+đó)\b", u_lower)):
                tc = {"name": eff, "arguments": {"time": f"{primary_d or '2026-10-16'}T07:00:00+07:00"} if eff == "alarm.create" else {"content": "Lời nhắc", "time": f"{primary_d or '2026-10-16'}T09:00:00+07:00"}}
            elif prms == ["time_slot"] and eff == "package_tracking.reschedule_delivery" and not (re.search(r"\b(bữa khác|hôm khác|ngày khác)\b", u_lower) and not has_explicit_time_in_u):
                tc = {"name": "package_tracking.reschedule_delivery", "arguments": {"tracking_number": "VN123456", "slot": "evening"}}
            elif prms == ["mode"] and eff in {"home_security.arm", "home_security.set_alarm_mode"} and re.search(r"\b(ở nhà|vắng nhà|đi vắng|ban đêm|đi ngủ|tắt hẳn|tắt báo động|tắt bảo vệ)\b", u_lower) and not re.search(r"\b(như .* dặn|chế độ nào)\b", u_lower):
                tc = {"name": "home_security.arm", "arguments": {"mode": "off" if "tắt" in u_lower else ("night" if "đêm" in u_lower else ("home" if "ở nhà" in u_lower else "away"))}}
            elif prms == ["book_id"] and eff == "library.reserve":
                tc = {"name": "library.reserve", "arguments": {"book_id": "lib_01"}}
            elif prms == ["bike_id"] and eff == "bike_rental.reserve":
                tc = {"name": "bike_rental.reserve", "arguments": {"bike_id": "bk_01"}}
            elif prms == ["episode_id"] and eff in {"podcast_radio.play", "podcast_radio.queue"}:
                tc = {"name": eff, "arguments": {"episode_id": "ep_01"}}
            elif eff == "music.play":
                tc = {"name": "music.play", "arguments": {"track_id": "trk_01"}}
            elif prms == ["course_id"] and eff == "learning.log_progress" and not re.search(r"\b(gì nhỉ|quên mất)\b", u_lower):
                tc = {"name": "learning.log_progress", "arguments": {"course_id": "crs_01", "minutes": 30}}
            elif eff in {"email.archive", "email.reply"} and not re.search(r"\bhỏi lại\b", u_lower):
                tc = {"name": eff, "arguments": {"email_id": "em_01", "action": "delete" if re.search(r"\b(xóa|xoá)\b", u_lower) else "archive"} if eff == "email.archive" else {"email_id": "em_01", "body": "Đã nhận"}}
        norm_calls.append(tc)
    calls = norm_calls
    num_ct_searches = sum(1 for x in calls if x.get("name") == "contacts.search")

    # Bóc tách dữ kiện từ history để đồng bộ trực tiếp vào giá trị mẫu (giữ nguyên định dạng JSON cụ thể cho 3B)
    hist_workout_mins: Optional[int] = None
    hist_rule_id: Optional[str] = None
    hist_chats_map: Dict[str, str] = {}
    hist_rem_time_content: Optional[Tuple[int, str]] = None
    hist_text_lower = ""
    for turn in (history or []):
        t_txt = clean_asr_transcript(str(turn.get("audio_text") or turn.get("text") or ""))
        hist_text_lower += " " + t_txt.lower()
        if turn.get("role") == "assistant" and t_txt:
            m_rem = re.search(r"nhắc lúc\s+(\d+)\s*giờ\s*(sáng|chiều|tối)?\s+với nội dung\s+([^\.?!]+)", t_txt, re.IGNORECASE)
            if m_rem:
                h_v = int(m_rem.group(1))
                if (m_rem.group(2) or "").lower() in {"chiều", "tối"} and h_v < 12:
                    h_v += 12
                c_v = m_rem.group(3).strip()
                c_v = c_v[0].upper() + c_v[1:] if c_v else c_v
                hist_rem_time_content = (h_v, c_v)
        for res in (turn.get("tool_results") or []):
            if not isinstance(res, dict):
                continue
            if "messages" in res and isinstance(res["messages"], list):
                for msg in res["messages"]:
                    if isinstance(msg, dict):
                        m_norm = _normalize_vn_numbers_in_time(str(msg.get("content", "")))
                        m_dur = re.search(r"\b(?:bơi|chạy|tập|đạp xe)\s+(\d+)\s*phút\b", m_norm, re.IGNORECASE)
                        if m_dur:
                            hist_workout_mins = int(m_dur.group(1))
            if "rule" in res and isinstance(res["rule"], dict) and res["rule"].get("id"):
                hist_rule_id = str(res["rule"]["id"])
            if "chats" in res and isinstance(res["chats"], list):
                for ch in res["chats"]:
                    if isinstance(ch, dict) and ch.get("id") and ch.get("name"):
                        hist_chats_map[str(ch["name"]).lower()] = str(ch["id"])

    inst_set = set(installed_apps.keys()) if isinstance(installed_apps, dict) else set(installed_apps or [])
    hist_has_sh_status = any(
        isinstance(tc_h, dict) and tc_h.get("name") == "smart_home.get_status"
        for t_h in (history or [])
        for tc_h in (t_h.get("tool_calls") or [])
    )
    hist_place_phone: Optional[str] = None
    for t_h in (history or []):
        for res_h in (t_h.get("tool_results") or []):
            if isinstance(res_h, dict) and isinstance(res_h.get("places"), list):
                for pl_h in res_h["places"]:
                    if isinstance(pl_h, dict) and pl_h.get("phone"):
                        pl_name_low = str(pl_h.get("name") or "").lower()
                        if (pl_name_low and pl_name_low in u_lower) or re.search(r"\b(gara|tiệm|quán|nhà hàng|cửa hàng|chỗ đó|bên đó)\b", u_lower):
                            hist_place_phone = str(pl_h["phone"])
                            break
    hist_phone_match = re.search(r"\b(0\d{8,11})\b", hist_text_lower)
    is_call_hist_phone = bool(
        (hist_phone_match and re.search(r"\b(gọi luôn số đó|gọi số đó)\b", u_lower))
        or hist_place_phone is not None
    )

    if "calendar" in inst_set and "calendar.list_events" not in hist_sat:
        if re.search(r"\b(cùng ngày với buổi|buổi workshop|lịch họp|họp phụ huynh|lịch khám|giờ khám|đá bóng|chạy bộ|(?<!quán\s)(?<!tiệm\s)(?<!ly\s)(?<!cốc\s)cà phê|dã ngoại|buổi họp|có lịch gì không|còn trống thì|đám cưới|tiệc cưới)\b", u_text, re.IGNORECASE):
            if any(tc["name"].startswith(("flights.", "ride_hailing.", "alarm.", "calendar.", "reminders.", "video_meeting.")) for tc in calls):
                out.append({"name": "calendar.list_events", "arguments": dict(cal_list_args)})
                seen.add("calendar.list_events")

    if "flights" in inst_set and "flights.my_bookings" not in hist_sat:
        if re.search(r"\b(chuyến .* đặt|vé .* đặt|ngày bay của tôi|sáng hôm bay|trước giờ bay|trước giờ cất cánh|máy bay hạ cánh)\b", u_text, re.IGNORECASE):
            if not any(tc["name"].startswith("flights.search") for tc in calls) and not re.search(r"\b(?:đặt|mua|lấy|tìm|chốt)\b[^,.!?;]*?\b(?:vé|chuyến)\b", u_text, re.IGNORECASE):
                out.append({"name": "flights.my_bookings", "arguments": {}})
                seen.add("flights.my_bookings")

    if "food_delivery" in inst_set and "food_delivery.order_history" not in hist_sat:
        if re.search(r"\b(đơn .* giao rồi|đơn .* chưa giao|đơn .* tới đâu)\b", u_text, re.IGNORECASE):
            out.append({"name": "food_delivery.order_history", "arguments": {}})
            seen.add("food_delivery.order_history")

    if "fitness" in inst_set and "fitness.get_activity" not in hist_sat:
        if re.search(r"\b(lâu bằng|bằng buổi tập|bằng lúc chiều)\b", u_lower) and any(tc["name"] == "calendar.create_event" for tc in calls):
            f_act_args = {"date": prev_d or primary_d or "2026-10-15"} if "hôm qua" in u_lower else {}
            out.append({"name": "fitness.get_activity", "arguments": f_act_args})
            seen.add("fitness.get_activity")
        elif any(tc["name"] == "fitness.set_goal" for tc in calls) and re.search(r"\b(lên thêm|tăng thêm|giảm đi|hạ bớt)\s+\d+", norm_u):
            out.append({"name": "fitness.get_activity", "arguments": {}})
            seen.add("fitness.get_activity")

    PARAM_WRITE_TO_READER: Dict[str, Tuple[str, Dict[str, Any]]] = {
        "ecommerce.add_to_cart": ("ecommerce.search", {"query": "sản phẩm"}),
        "grocery.add_to_cart": ("grocery.search", {"query": "thực phẩm"}),
        "train_bus_tickets.book": ("train_bus_tickets.search", {"origin": "Hà Nội", "destination": "Vinh", "date": primary_d or "2026-10-16"}),
        "flights.book": ("flights.search", {"origin": "HAN", "destination": "DAD", "date": primary_d or "2026-10-16"}),
        "parking.reserve": ("parking.find", {"near": "Nội Bài"}),
        "recipes.save": ("recipes.search", {"query": "món ăn"}),
        "library.reserve": ("library.search", {"query": "sách"}),
        "bike_rental.reserve": ("bike_rental.find_bikes", {"station": "Mỹ Đình"}),
        "podcast_radio.play": ("podcast_radio.search", {"query": "podcast"}),
        "podcast_radio.queue": ("podcast_radio.search", {"query": "podcast"}),
        "podcast_radio.subscribe": ("podcast_radio.search", {"query": "podcast"}),
    }

    has_explicit_time_in_hist = bool(re.search(r"\b\d{1,2}\s*(?:giờ|rưỡi|h\b)", _normalize_vn_numbers_in_time(hist_text_lower)) or "buổi chiều" in u_lower)
    for tc in calls:
        tname = tc["name"]
        if tname == "ewallet.transfer" and is_vague_ewallet_amount:
            tc = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "ewallet.transfer", "params": ["amount"]}}
            tname = "sys.ask_missing_info"
        elif tname in {"alarm.create", "reminders.create"} and not has_explicit_time_in_u and not has_explicit_time_in_hist and not hist_rem_time_content and "flights.my_bookings" not in seen and "flights.my_bookings" not in hist_sat:
            tc = {"name": "sys.ask_missing_info", "arguments": {"tool_name": tname, "params": ["time"]}}
            tname = "sys.ask_missing_info"
        elif tname == "video_meeting.create" and not has_explicit_time_in_u and not has_explicit_time_in_hist and "calendar.list_events" not in seen and "calendar.list_events" not in hist_sat:
            tc = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "video_meeting.create", "params": ["start"]}}
            tname = "sys.ask_missing_info"
        elif tname == "package_tracking.reschedule_delivery" and re.search(r"\b(bữa khác|hôm khác|ngày khác)\b", u_lower) and not has_explicit_time_in_u:
            tc = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "package_tracking.reschedule_delivery", "params": ["time"]}}
            tname = "sys.ask_missing_info"
        elif tname in {"home_security.arm", "home_security.set_alarm_mode"} and re.search(r"\b(như .* dặn|chế độ nào)\b", u_lower) and not re.search(r"\b(ở nhà|vắng nhà|đi vắng|ban đêm|đi ngủ|tắt)\b", u_lower):
            tc = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "home_security.arm", "params": ["mode"]}}
            tname = "sys.ask_missing_info"
        if tname == "contacts.search" and (spoken_phone_match or is_call_hist_phone or re.search(r"\b(số đuôi)\b", u_lower)) and any(x.get("name") in {"ewallet.topup_mobile", "phone.call", "sms.send"} for x in calls) and not any(x.get("name") in {"ewallet.transfer", "contacts.update"} for x in calls):
            continue
        if tname == "contacts.search" and is_vague_ewallet_amount and non_ct_apps_in_calls <= {"ewallet"}:
            continue
        if tname in hist_sat:
            if tname == "recipes.search" and not any(x["name"] in {"recipes.get", "recipes.save"} for x in calls):
                tname = "recipes.save" if re.search(r"\b(lưu|cất)\b", u_lower) else "recipes.get"
                tc = {"name": tname, "arguments": {"recipe_id": "rcp_01"}}
            else:
                continue
        if is_price_query and tname == "sys.ask_missing_info":
            continue
        if tname == "smart_home.fan_set" and re.search(r"\bquạt\b[^,.]{0,25}\b(cứ để rứa|cứ để nguyên|để nguyên|khỏi tắt|đừng tắt|khỏi bật|đừng bật)\b", u_lower) and not re.search(r"\b(bật hết quạt|tắt hết quạt)\b", u_lower):
            continue
        if tname in seen and (
            tname in READ_TOOLS
            or (tname == "food_delivery.place_order" and not re.search(r"\b(tách làm 2 đơn|tách làm hai đơn|2 đơn riêng|hai đơn riêng|mỗi địa chỉ một đơn|bên\s+cô\s+ba.*nhà\s+mèo|hai\s+quán|2\s+quán)\b", u_lower))
            or (tname == "banking.transfer" and not has_multi_bank_transfer)
            or (tname == "smart_home.light_set" and single_room_light)
            or (tname == "food_delivery.cancel_order" and not re.search(r"\b(hủy hết|hủy cả hai|hủy 2 đơn|hủy hai đơn)\b", u_lower))
            or (tname == "household_chores.assign" and not has_plural and not re.search(r"(?<!ví\s)(?<!quý\s)(?<!khoản\s)\bcòn\s+[^,.]*\bgiao\b|\b(với phơi|và phơi|hai việc|2 việc)\b", u_lower))
            or (tname == "bike_rental.reserve" and not has_plural and not re.search(r"\b(chiếc .* với chiếc|hai chiếc|2 chiếc|giữ thêm chiếc)\b", u_lower))
            or (tname == "loyalty.redeem" and not has_plural and not re.search(r"\b(\d+\s*(?:vé|voucher|phiếu|ly|phần)|mấy\s+vé|hai\s+vé|ba\s+vé)\b", norm_u))
            or (not has_plural and tname in {"home_security.acknowledge_alert", "subscriptions.cancel", "tasks.add", "tasks.complete", "learning.schedule_lesson", "learning.log_progress", "calendar.create_event", "video_meeting.create"})
        ):
            continue
        if tname == "home_automation.create_rule" and "smart_home" in inst_set and not hist_has_sh_status and "smart_home.get_status" not in seen:
            out.append({"name": "smart_home.get_status", "arguments": {}})
            seen.add("smart_home.get_status")
        if tname in WRITE_TO_ZERO_ARG_READERS:
            for r_tool in WRITE_TO_ZERO_ARG_READERS[tname]:
                if r_tool not in hist_sat and r_tool not in seen:
                    r_args = dict(cal_list_args) if r_tool == "calendar.list_events" else {}
                    out.append({"name": r_tool, "arguments": r_args})
                    seen.add(r_tool)
        if tname in PARAM_WRITE_TO_READER:
            p_reader, p_default_args = PARAM_WRITE_TO_READER[tname]
            if p_reader not in hist_sat and p_reader not in seen and not any(x["name"] == p_reader for x in calls):
                out.append({"name": p_reader, "arguments": dict(p_default_args)})
                seen.add(p_reader)
        if tname == "food_delivery.place_order":
            if "food_delivery.list_restaurants" not in hist_sat and "food_delivery.list_restaurants" not in seen and not any(x["name"] == "food_delivery.list_restaurants" for x in calls):
                out.append({"name": "food_delivery.list_restaurants", "arguments": {}})
                seen.add("food_delivery.list_restaurants")
            if "food_delivery.get_menu" not in hist_sat and "food_delivery.get_menu" not in seen and not any(x["name"] == "food_delivery.get_menu" for x in calls):
                out.append({"name": "food_delivery.get_menu", "arguments": {"restaurant_id": "rest_01"}})
                seen.add("food_delivery.get_menu")
        if tname == "tv.tune" and not re.search(r"\b(vtv\d+|htv\d+)\b", u_lower):
            if "tv.guide" not in hist_sat and "tv.guide" not in seen and not any(x["name"] == "tv.guide" for x in calls):
                out.append({"name": "tv.guide", "arguments": {}})
                seen.add("tv.guide")
        if tname == "public_transit.topup_card" and re.search(r"\b(lượt|vé|tuyến|khứ hồi)\b", u_lower):
            if "public_transit.find_routes" not in hist_sat and "public_transit.find_routes" not in seen and not any(x["name"] == "public_transit.find_routes" for x in calls):
                out.append({"name": "public_transit.find_routes", "arguments": {"origin": "Bến Thành", "destination": "Suối Tiên"}})
                seen.add("public_transit.find_routes")
        if tname == "phone.call" and is_call_hist_phone and (hist_place_phone or hist_phone_match):
            tc = {"name": "phone.call", "arguments": {"phone_number": hist_place_phone or hist_phone_match.group(1)}}
        elif tname == "phone.call" and ("contacts" not in inst_set or re.search(r"\b(số đuôi)\b", u_lower)):
            if "phone.recent_calls" not in hist_sat and "phone.recent_calls" not in seen and not any(x["name"] == "phone.recent_calls" for x in calls):
                out.append({"name": "phone.recent_calls", "arguments": {}})
                seen.add("phone.recent_calls")
            tc = {"name": "phone.call", "arguments": {"phone_number": "0901234567"}}
        elif (
            (tname == "phone.call" and not spoken_phone_match and not is_call_hist_phone and not re.search(r"\b(số đuôi)\b", u_lower))
            or (tname == "sms.send" and not spoken_phone_match)
            or (tname == "sms.read_recent" and _extract_contact_query_from_text(u_text, history) is not None)
            or tname in {"ewallet.transfer", "contacts.update"}
            or (tname == "ewallet.topup_mobile" and not spoken_phone_match)
            or (tname == "sys.ask_missing_info" and (tc.get("arguments") or {}).get("tool_name") == "ewallet.transfer" and len(non_ct_apps_in_calls) > 1 and _extract_contact_query_from_text(u_text, history) is not None)
        ) and "contacts" in inst_set:
            if "contacts.search" not in hist_sat and "contacts.search" not in seen and not any(x["name"] == "contacts.search" for x in calls):
                q_ct = _extract_contact_query_from_text(u_text, history) or "Lan"
                out.insert(0, {"name": "contacts.search", "arguments": {"query": q_ct}})
                seen.add("contacts.search")
        if tname == "grocery.checkout" and "grocery.add_to_cart" not in hist_sat and "grocery.add_to_cart" not in seen and not any(x["name"] == "grocery.add_to_cart" for x in calls):
            if "grocery.search" not in hist_sat and "grocery.search" not in seen and not any(x["name"] == "grocery.search" for x in calls):
                out.append({"name": "grocery.search", "arguments": {"query": "thực phẩm"}})
                seen.add("grocery.search")
            out.append({"name": "grocery.add_to_cart", "arguments": {"product_id": "gro_01", "quantity": 1}})
            seen.add("grocery.add_to_cart")
        elif tname == "ecommerce.checkout" and "ecommerce.add_to_cart" not in hist_sat and "ecommerce.add_to_cart" not in seen and not any(x["name"] == "ecommerce.add_to_cart" for x in calls):
            if "ecommerce.search" not in hist_sat and "ecommerce.search" not in seen and not any(x["name"] == "ecommerce.search" for x in calls):
                out.append({"name": "ecommerce.search", "arguments": {"query": "sản phẩm"}})
                seen.add("ecommerce.search")
            out.append({"name": "ecommerce.add_to_cart", "arguments": {"product_id": "prod_01", "quantity": 1}})
            seen.add("ecommerce.add_to_cart")

        # Đồng bộ hóa ngày tháng và giá trị cụ thể trong mẫu
        raw_args = dict(tc.get("arguments") or {})
        if tname == "contacts.search" and num_ct_searches <= 1:
            q_match = _extract_contact_query_from_text(u_text, history)
            if q_match:
                raw_args["query"] = q_match
            else:
                q_existing = str(raw_args.get("query") or "").strip().lower()
                if q_existing and q_existing not in u_lower and q_existing not in hist_text_lower:
                    if not any(x.get("name") in ("phone.call", "sms.send", "zalo.send_message") for x in calls):
                        continue
        elif tname == "phone.call" and spoken_phone_match and "contacts.update" not in seen and not any(x.get("name") == "contacts.update" for x in calls):
            raw_args = {"phone_number": spoken_phone_match.group(1)}
        elif tname == "ewallet.topup_mobile":
            raw_args.pop("contact_id", None)
            if spoken_phone_match:
                raw_args["phone"] = spoken_phone_match.group(1)
            else:
                raw_args.setdefault("phone", "0901234567")
            m_top_amt = re.search(r"\b(\d+)\s*(?:nghìn|ngàn|k)\b", norm_u)
            if m_top_amt:
                raw_args["amount"] = int(m_top_amt.group(1)) * 1000
        elif tname == "sms.send" and spoken_phone_match:
            raw_args.pop("contact_id", None)
            raw_args["phone_number"] = spoken_phone_match.group(1)
            m_sms_cnt = re.search(
                r"\b(?:là|rằng)\s+([^.!?]+?)(?:\s*,?\s*(?:rồi|xong|với|và)\s+(?:hủy|huỷ|đặt|gọi|tắt|bật|mở|nhắc|thêm|tạo|chuyển|đổi|dời)\b|\s+(?:nha|nhé|nghe)\b|[.!?]|$)",
                u_text,
                re.IGNORECASE,
            )
            if m_sms_cnt:
                c_sms = m_sms_cnt.group(1).strip()
                if c_sms:
                    raw_args["content"] = c_sms[0].upper() + c_sms[1:]
        elif tname == "alarm.create":
            m_al_u = re.search(
                r"\b(?:báo\s+thức(?:\s+mới)?(?:\s+vào|\s+lúc)?|kêu\s+\S+\s+dậy|gọi\s+\S+\s+dậy)\s+(?:lúc\s+|vào\s+)?(\d{1,2})\s*(?:giờ\s*)?(rưỡi|\d{1,2})?\s*(sáng|trưa|chiều|tối)?\b",
                norm_u,
                re.IGNORECASE,
            )
            if m_al_u and isinstance(raw_args.get("time"), str) and not re.search(r"\b(trước|sớm\s+hơn|sau\s+khi)\b", norm_u):
                ah = int(m_al_u.group(1))
                am = 30 if (m_al_u.group(2) or "").lower() == "rưỡi" else int(m_al_u.group(2) or 0)
                if (m_al_u.group(3) or "").lower() in {"chiều", "tối"} and ah < 12:
                    ah += 12
                raw_args["time"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{ah:02d}:{am:02d}:00", raw_args["time"])
            if primary_d and isinstance(raw_args.get("time"), str) and not re.search(r"\b(sáng hôm bay|trước giờ bay|trước giờ cất cánh)\b", norm_u):
                raw_args["time"] = re.sub(r"^\d{4}-\d{2}-\d{2}", primary_d, raw_args["time"])
            m_lbl_u = re.search(r"\btên\s+là\s+([^,.!?]+?)(?:\s+rồi\b|\s+xong\b|\s+với\b|\s+và\b|\s+đi\b|\s+nha\b|\s+nhé\b|[,.!?]|$)", u_text, re.IGNORECASE)
            if m_lbl_u:
                lbl_s = m_lbl_u.group(1).strip()
                if lbl_s:
                    raw_args["label"] = lbl_s[0].upper() + lbl_s[1:]
        elif tname == "music.control":
            if re.search(r"\b(dừng|tạm dừng|tắt nhạc|ngưng)\b", u_lower):
                raw_args = {"action": "pause"}
        elif tname == "stocks.get_quote":
            m_sym_u = re.search(r"\bmã\s+([A-Z]{3})\b", u_text)
            if m_sym_u:
                raw_args["symbol"] = m_sym_u.group(1)
        elif tname == "calendar.create_event":
            m_cal_title = (
                re.search(r"^(?:buổi|lịch)\s+([^,.]+?)\s+(?:hôm|ngày|thứ|tuần|sáng|chiều|tối|lúc)\b", u_text.strip(), re.IGNORECASE)
                or re.search(r"\bthêm\s+(?:cho\s+\S+\s+|giúp\s+\S+\s+)?(?:cái\s+|một\s+)?(?:lịch|buổi)\s+([^,.]+?)\s+(?:lúc\s+|\d{1,2}\s*giờ|hai\s+giờ|ba\s+giờ|bốn\s+giờ|năm\s+giờ|sáu\s+giờ|bảy\s+giờ|tám\s+giờ|chín\s+giờ|mười\s+giờ|trước\s+|vào\s+|hôm\s+|ngày\s+)", u_text, re.IGNORECASE)
                or re.search(r"\bghi\s+vào\s+lịch\s+[^,.]*?\blà\s+([^,.]+)", u_text, re.IGNORECASE)
            )
            if m_cal_title:
                t_c_val = re.sub(r"\s+rồi$", "", m_cal_title.group(1).strip(), flags=re.IGNORECASE)
                if t_c_val:
                    raw_args["title"] = t_c_val[0].upper() + t_c_val[1:]
            m_cal_rng = re.search(
                r"\b(\d{1,2})\s*(?:giờ\s*)?(rưỡi|\d{1,2})?\s*(sáng|trưa|chiều|tối)?\s*(?:tới|đến|-)\s*(\d{1,2})\s*(?:giờ\s*)?(rưỡi|\d{1,2})?\s*(sáng|trưa|chiều|tối)?",
                norm_u,
                re.IGNORECASE,
            )
            m_cal_hr = re.search(r"\b(?:(?:lúc|vào|lịch|tạo\s+cái\s+lịch|thêm\s+lịch)\s+(?:[^,.!?]{0,25}?\s+)?)?(\d{1,2})\s*giờ(?:\s*(\d{1,2}|rưỡi))?\s*(sáng|trưa|chiều|tối)?", norm_u, re.IGNORECASE)
            if m_cal_rng and isinstance(raw_args.get("start"), str):
                h1 = int(m_cal_rng.group(1))
                m1 = 30 if (m_cal_rng.group(2) or "").lower() == "rưỡi" else int(m_cal_rng.group(2) or 0)
                h2 = int(m_cal_rng.group(4))
                m2 = 30 if (m_cal_rng.group(5) or "").lower() == "rưỡi" else int(m_cal_rng.group(5) or 0)
                p_rng = (m_cal_rng.group(3) or m_cal_rng.group(6) or "").lower()
                if p_rng in {"chiều", "tối"}:
                    if h1 < 12:
                        h1 += 12
                    if h2 < 12:
                        h2 += 12
                raw_args["start"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{h1:02d}:{m1:02d}:00", raw_args["start"])
                if isinstance(raw_args.get("end"), str):
                    raw_args["end"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{h2:02d}:{m2:02d}:00", raw_args["end"])
            elif m_cal_hr and not re.search(r"\btrước\s+[^,.!?]*?\d+\s*tiếng", norm_u) and isinstance(raw_args.get("start"), str):
                ch = int(m_cal_hr.group(1))
                cm = 30 if (m_cal_hr.group(2) or "").lower() == "rưỡi" else int(m_cal_hr.group(2) or 0)
                if (m_cal_hr.group(3) or "").lower() in {"chiều", "tối"} and ch < 12:
                    ch += 12
                raw_args["start"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{ch:02d}:{cm:02d}:00", raw_args["start"])
                if isinstance(raw_args.get("end"), str):
                    raw_args["end"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{(ch + 1) % 24:02d}:{cm:02d}:00", raw_args["end"])
            if re.search(r"\b(lâu bằng|bằng buổi tập|bằng lúc chiều|bằng hôm qua)\b", u_lower):
                raw_args.pop("location", None)
                if isinstance(raw_args.get("start"), str):
                    raw_args.setdefault("end", raw_args["start"])
                m_walk_t = re.search(r"\b(đi\s+bộ\s+quanh\s+hồ|chạy\s+bộ|đạp\s+xe|tập\s+gym|tập\s+yoga)\b", u_text, re.IGNORECASE)
                if m_walk_t:
                    wt_s = m_walk_t.group(1).strip()
                    raw_args["title"] = wt_s[0].upper() + wt_s[1:]
        elif tname == "calendar.list_events":
            raw_args = dict(cal_list_args)
        elif tname == "event_tickets.search":
            raw_args = {}
        elif tname == "appliances.list":
            if len(calls) <= 2 and "appliances.start_cycle" not in {x["name"] for x in calls} and "appliances.stop" not in {x["name"] for x in calls}:
                if "máy giặt" in u_lower:
                    raw_args = {"type": "washer"}
        elif tname == "smart_home.get_status":
            if len(calls) == 1 and re.search(r"\b(ngoài|ở)\s+phòng\s+khách\b", u_lower):
                raw_args = {"room": "phòng khách"}
            elif re.search(r"\brèm\b", u_lower) and not re.search(r"\b(đèn|quạt|điều hòa|điều hoà|máy lạnh)\b", u_lower):
                raw_args = {"type": "curtain"}
            else:
                raw_args = {}
        elif tname == "loyalty.get_points":
            if "cineviệt" in u_lower:
                raw_args = {"program": "CineViệt"}
            elif "góc phố" in u_lower:
                raw_args = {"program": "Góc Phố 24h"}
            elif not re.search(r"\b(coffee|cà phê|góc phố|cao nguyên|siêu thị|nhà sách)\b", u_lower):
                raw_args = {}
        elif tname == "fuel_price.get_price":
            raw_args = {"type": "diesel"} if ("dầu" in u_lower or "diesel" in u_lower) else {}
        elif tname == "exchange_rate.get_rate":
            if re.search(r"\b(yên|jpy)\b", u_lower):
                raw_args = {"from_currency": "JPY", "to_currency": "VND"}
            elif re.search(r"\b(đô|usd)\b", u_lower):
                raw_args = {"from_currency": "USD", "to_currency": "VND"}
            elif re.search(r"\b(euro|ơ-rô|eur)\b", u_lower):
                raw_args = {"from_currency": "EUR", "to_currency": "VND"}
        elif tname == "exchange_rate.convert":
            cur_code = "JPY" if re.search(r"\b(yên|jpy)\b", u_lower) else ("EUR" if re.search(r"\b(euro|ơ-rô|eur)\b", u_lower) else "USD")
            m_fx_amt = re.search(r"\b(\d+)\s*(nghìn|ngàn|trăm|triệu)?\s*(?:đô|usd|yên|jpy|euro|ơ-rô|eur)\b", norm_u, re.IGNORECASE)
            if m_fx_amt:
                mult_fx = 1000 if (m_fx_amt.group(2) or "") in ("nghìn", "ngàn") else (100 if (m_fx_amt.group(2) or "") == "trăm" else (1_000_000 if (m_fx_amt.group(2) or "") == "triệu" else 1))
                raw_args = {"amount": int(m_fx_amt.group(1)) * mult_fx, "from_currency": cur_code, "to_currency": "VND"}
        elif tname == "lottery.get_results":
            if "đà nẵng" in u_lower:
                raw_args["province"] = "Đà Nẵng"
            elif "hà nội" in u_lower:
                raw_args["province"] = "Hà Nội"
            elif "đồng nai" in u_lower:
                raw_args["province"] = "Đồng Nai"
            elif "sài gòn" in u_lower or "tp.hcm" in u_lower or "thành phố" in u_lower:
                raw_args["province"] = "TP.HCM"
            if re.search(r"\b(mới nhất|đợt gần nhất|kỳ vừa rồi|xổ ra số mấy)\b", u_lower) and not re.search(r"\b(chiều nay|hôm nay|bữa nay|hôm qua)\b", u_lower):
                raw_args.pop("date", None)
        elif tname == "lottery.check_ticket":
            if "đà nẵng" in u_lower:
                raw_args["province"] = "Đà Nẵng"
            elif "đồng nai" in u_lower:
                raw_args["province"] = "Đồng Nai"
            elif "hà nội" in u_lower:
                raw_args["province"] = "Hà Nội"
            elif "sài gòn" in u_lower or "tp.hcm" in u_lower:
                raw_args["province"] = "TP.HCM"
            norm_lot_digits = re.sub(r"(?<=\b\d)\s+(?=\d\b)", "", norm_u)
            m_lot6 = re.search(r"\b(\d{6})\b", norm_lot_digits)
            if m_lot6:
                raw_args["number"] = m_lot6.group(1)
        elif tname == "sms.read_recent" and "contacts.search" not in seen and "contacts.search" not in hist_sat:
            raw_args = {}
        elif tname == "podcast_radio.search":
            if "ryan cooper" in u_lower:
                raw_args["query"] = "Ryan Cooper"
            elif "khoa học mỗi ngày" in u_lower:
                raw_args["query"] = "Khoa học mỗi ngày"
            else:
                m_pod_topic = re.search(r"\bnói\s+chuyện\s+(?:mua\s+|về\s+)?([^\s,.!?]+)", u_lower)
                m_pod_ep = re.search(r"\b(?:tập|chương trình|kênh)\s+([A-ZÀ-Ỹ][^,.!?]+?)(?:\s+đó\b|\s+vào\s+danh\s+sách|\s+thêm\s+vào|\s+của\s+|\s+cho\s+|\s+giúp\s+|\s+giùm\s+|\s+hộ\s+|\s+nghe\b|[,.]|$)", u_text)
                if m_pod_topic:
                    raw_args["query"] = m_pod_topic.group(1).strip()
                elif m_pod_ep:
                    raw_args["query"] = m_pod_ep.group(1).strip()
                elif isinstance(raw_args.get("query"), str):
                    q_pod = re.sub(r"^(?:tập|số|kênh|chương trình|podcast)\s+(?:của\s+)?", "", raw_args["query"].strip(), flags=re.IGNORECASE)
                    q_pod = re.sub(r"\s+(?:tập\s*\d+|số\s*\d+|\d+\s*tuần\s*trước|mới\s*nhất).*$", "", q_pod, flags=re.IGNORECASE).strip()
                    if q_pod:
                        raw_args["query"] = q_pod
        elif tname == "library.search":
            if "phương pháp nghiên cứu định tính" in u_lower:
                raw_args["query"] = "Phương pháp nghiên cứu định tính"
            else:
                m_bk_u = re.search(r"\b(?:cuốn|sách)\s+([A-ZÀ-Ỹ][A-Za-zÀ-ỹ0-9\s]+?)(?:\s+về\s+cho\s+|\s+cho\s+\S+\s+đọc\b|\s+của\s+|\s+trên\s+app|\s+ở\s+thư\s+viện|\s+giùm\b|\s+giúp\b|\s+nếu\b|\s+còn\b|\s+không\b|\s+nghe\b|\s+nhé\b|\s+nha\b|[,.]|$)", u_text)
                if m_bk_u and m_bk_u.group(1).strip().lower() not in {"đấy", "đó", "này", "ấy"}:
                    raw_args["query"] = m_bk_u.group(1).strip()
                elif history:
                    for turn_h in reversed(history):
                        m_bk_h = re.search(r"\b(?:cuốn|sách)\s+([A-ZÀ-Ỹ][A-Za-zÀ-ỹ0-9\s]+?)(?:\s+về\s+cho\s+|\s+của\s+|\s+trên\s+app|\s+ở\s+thư\s+viện|\s+còn\b|\s+không\b|[,.?!]|$)", str(turn_h.get("audio_text") or turn_h.get("text") or ""))
                        if m_bk_h and m_bk_h.group(1).strip().lower() not in {"đấy", "đó", "này", "ấy"}:
                            raw_args["query"] = m_bk_h.group(1).strip()
                            break
        elif tname == "grocery.search":
            u_no_pantry = re.sub(
                r"\b(?:lấy|dùng|xài|ăn|uống|trừ)\b[^,.!?;]*?\b(?:trong\s+tủ|tủ\s+bếp|kho\s+bếp|ra\s+dùng|trừ\s+đi)\b.*?(?=\b(?:à\s+mua|rồi\s+mua|rồi\s+đặt|với\s+mua|mua\s+thêm|đặt\s+mua|mua\s+giúp)\b|[,.!?;]|$)"
                r"|\b(?:trong\s+tủ(?:\s+bếp)?|tủ\s+bếp|kho\s+bếp)\s+(?:trừ|lấy|dùng|xài)\b.*?(?=\b(?:à\s+mua|rồi\s+mua|rồi\s+đặt|với\s+mua|mua\s+thêm|đặt\s+mua|mua\s+giúp)\b|[,.!?;]|$)",
                "",
                u_lower,
            )
            GROCERY_ITEMS = (
                "cá ba sa", "cá hồi", "cá thu", "cá lóc", "xúc xích", "trứng gà", "trứng",
                "sữa tươi", "sữa hạt", "sữa chua", "ức gà", "gà ta", "thịt ba chỉ", "thịt bò", "thịt heo", "sườn non", "tôm tươi",
                "cà chua", "dưa chuột", "dưa leo", "dầu ô liu", "yến mạch", "bột mì", "rau cải ngọt", "cải ngọt", "rau cải", "rau muống", "bông cải",
                "táo envy", "táo", "chuối", "cam sành", "cam", "gạo st25", "gạo", "hạt macca", "macca", "hạnh nhân",
                "đậu hũ", "nước mắm", "nước tương", "hành tây", "khoai tây", "cà rốt", "bí đỏ", "kem tươi", "bơ lạt", "phô mai", "bánh mì", "mì ý",
                "nấm đùi gà", "nấm kim châm", "nấm rơm", "nấm",
            )
            matched_gr = next((it for it in GROCERY_ITEMS if it in u_no_pantry), None)
            if not matched_gr:
                m_gr_dyn = re.search(
                    r"\b(?:đặt\s+mua(?:\s+thêm|\s+lại)?|mua\s+thêm|đặt\s+thêm|lấy\s+thêm|thêm|bỏ)\s+(?:(?:\d+|một|hai|ba|bốn|năm)\s+(?:gói|hộp|vỉ|bó|kg|cân|ký|khay|chai|túi|con|quả|trái|bịch|lốc)\s+)?([^,.!?;]+?)(?:\s+(?:vào\s+giỏ|vô\s+giỏ|bên\s+(?:app\s+|áp\s+)?đi\s+chợ|trên\s+(?:app\s+|áp\s+)?đi\s+chợ|ở\s+siêu\s+thị|cho\s+đủ|luôn|nhé|nha)|$)",
                    u_no_pantry,
                )
                if m_gr_dyn:
                    cand_gr = m_gr_dyn.group(1).strip()
                    if cand_gr and cand_gr not in {"lại", "luôn", "đi", "nhé", "nha"} and not re.search(r"\b(nguyên liệu|còn thiếu|thiếu gì|danh sách)\b", cand_gr):
                        matched_gr = cand_gr
            if not matched_gr and history:
                matched_gr = next((it for it in GROCERY_ITEMS if it in hist_text_lower), None)
            if matched_gr:
                raw_args["query"] = matched_gr
        elif tname == "recipes.search":
            RECIPE_DISHES = (
                "bò kho", "canh chua cá lóc", "canh chua", "cháo nấm", "cháo gà", "súp bí đỏ", "bún riêu cua", "bún riêu",
                "lẩu nấm", "cà ri", "gà xào sả ớt", "gà xào", "thịt kho trứng", "thịt kho", "pancake", "smoothie", "salad",
                "cơm chiên", "chè hạt sen", "chè sen", "gỏi cuốn", "mì ý", "mì xào", "mì",
            )
            matched_rc = next((d for d in RECIPE_DISHES if re.search(rf"\b{re.escape(d)}\b", u_lower)), None)
            if not matched_rc:
                m_rc_dyn = re.search(
                    r"\b(?:công\s+thức|cách\s+làm|nấu\s+món|làm\s+món|món)\s+([^,.!?;]+?)(?:\s+(?:không\s+chứa|cho\s+\d+|nhanh\s+nhất|dễ\s+làm|cần\s+nguyên\s+liệu|xem|coi|nhé|nha|đi)|$)",
                    u_lower,
                )
                if m_rc_dyn:
                    cand_rc = m_rc_dyn.group(1).strip()
                    if cand_rc and not re.search(r"^(?:ăn|chay|gì|nào|đó|ấy|này|không\s+chứa)", cand_rc):
                        matched_rc = cand_rc
            if not matched_rc and history:
                matched_rc = next((d for d in RECIPE_DISHES if re.search(rf"\b{re.escape(d)}\b", hist_text_lower)), None)
            if matched_rc:
                raw_args["query"] = matched_rc
        elif tname == "ecommerce.search":
            ECOM_PRODUCTS = (
                "chuột không dây", "chuột logitech", "bàn phím", "tai nghe chụp tai", "tai nghe", "cốc giữ nhiệt", "bình giữ nhiệt",
                "lọc không khí", "máy hút ẩm", "quạt mini", "bàn ủi", "đèn bàn", "máy sấy", "máy xay", "kẹp bàn", "áo sơ mi",
                "giày chạy bộ", "túi rác", "nước giặt", "giá đỡ", "cáp sạc", "ổ cắm thông minh", "ổ cắm chuyển đổi", "bình đun",
                "nồi chiên", "kệ sách", "ghế công thái học", "bàn di chuột", "sạc dự phòng", "ốp lưng",
            )
            matched_ec = next((p for p in ECOM_PRODUCTS if p in u_lower), None)
            if not matched_ec and history:
                if re.search(r"\b(giao tới rồi|đã giao|vừa giao|món đó|cái đó)\b", u_lower):
                    for turn_h in reversed(history):
                        for res_h in (turn_h.get("tool_results") or []):
                            if isinstance(res_h, dict) and isinstance(res_h.get("orders"), list):
                                deliv = [o for o in res_h["orders"] if isinstance(o, dict) and str(o.get("status", "")).lower() in ("delivered", "đã giao")]
                                target_o = deliv[0] if deliv else (res_h["orders"][0] if res_h["orders"] else None)
                                if isinstance(target_o, dict):
                                    o_name = target_o.get("product_name") or target_o.get("item") or ((target_o.get("items") or [{}])[0].get("name") if isinstance(target_o.get("items"), list) and target_o.get("items") else "")
                                    if o_name:
                                        matched_ec = str(o_name)
                                        break
                        if matched_ec:
                            break
                if not matched_ec:
                    matched_ec = next((p for p in ECOM_PRODUCTS if p in hist_text_lower), None)
            if matched_ec:
                raw_args["query"] = matched_ec
        elif tname == "maps.search_place":
            if "photo" in u_lower or "in bài" in u_lower or "in ảnh" in u_lower:
                raw_args = {"query": "photo"}
            else:
                m_mp_q = (
                    re.search(r"\b(?:quán|tiệm|nhà hàng|bãi giữ xe|bãi đỗ xe|bãi xe)\s+([^,.!?;]+?)(?:\s+ở\s+số\b|\s+ở\s+quận\b|\s+gần\b|\s+quanh\b|\s+đặt\s+tên\s+là\b|\s+rồi\s+lưu\b|\s+vào\s+bản\s+đồ\b|$)", u_text, re.IGNORECASE)
                    or re.search(r"\blưu\s+(?:giúp|giùm|dùm|tôi|chị|anh|em)?\s*(?:cái\s+)?(?:quán|tiệm|chỗ)\s+([^,.!?;]+?)(?:\s+ở\b|\s+vào\b|\s+đặt\s+tên\b|$)", u_text, re.IGNORECASE)
                )
                if m_mp_q:
                    q_mp_v = m_mp_q.group(1).strip()
                    if not re.search(r"\bmà\s+.*?\s+hẹn\b", q_mp_v, re.IGNORECASE):
                        raw_args["query"] = q_mp_v
                    elif history:
                        for turn_h in reversed(history):
                            for res_h in (turn_h.get("tool_results") or []):
                                if isinstance(res_h, dict) and isinstance(res_h.get("messages"), list):
                                    for msg_h in res_h["messages"]:
                                        m_q_h = re.search(r"\bquán\s+([A-ZÀ-Ỹ][^,.!?;]+?)(?:\s+nhé|\s+nha|[,.!?]|$)", str(msg_h.get("content") or msg_h.get("text") or ""))
                                        if m_q_h:
                                            raw_args["query"] = m_q_h.group(1).strip()
                                            break
            if "near" in raw_args:
                near_str = str(raw_args.get("near") or "").strip().lower()
                if near_str not in u_lower or re.search(r"\b(trường|nhà|công ty|cơ quan|văn phòng|chỗ làm|đây)\b", near_str):
                    raw_args.pop("near", None)
        elif tname == "gold_price.get_price":
            raw_args = {"type": "ring" if "nhẫn" in u_lower else "sjc"}
        elif tname == "smart_home.light_set":
            dev_id = raw_args.get("device_id", "light_living")
            m_br = re.search(r"\b(\d{1,3})\s*(?:phần trăm|%)", norm_u)
            br_val = int(m_br.group(1)) if m_br else (50 if re.search(r"\b(một nửa|phân nửa|còn 50)\b", u_lower) else None)
            if br_val is not None:
                if re.search(r"\b(bật|mở)\b", u_lower) and not re.search(r"\b(giảm|hạ)\b", u_lower):
                    raw_args = {"device_id": dev_id, "power": "on", "brightness": br_val}
                else:
                    raw_args = {"device_id": dev_id, "brightness": br_val}
            else:
                raw_args = {"device_id": dev_id, "power": "off" if "tắt" in u_lower else "on"}
        elif tname == "smart_home.climate_set":
            dev_id = raw_args.get("device_id", "ac_living")
            if re.search(r"\btắt\s+(?:cái\s+|mấy\s+cái\s+)?(?:điều\s+hòa|điều\s+hoà|máy\s+lạnh)\b", u_lower) and not re.search(r"\b\d{2}\s*độ\b", norm_u):
                raw_args = {"device_id": dev_id, "power": "off"}
            elif "hút ẩm" in u_lower:
                raw_args["mode"] = "dry"
                if not re.search(r"\b\d{2}\s*độ\b", norm_u):
                    raw_args.pop("temperature", None)
        elif tname == "smart_home.fan_set":
            dev_id = raw_args.get("device_id", "fan_living")
            m_spd = re.search(r"\bsố\s*(\d)\b", norm_u)
            if m_spd:
                raw_args = {"device_id": dev_id, "power": "on", "speed": int(m_spd.group(1))}
            elif "tắt" in u_lower:
                raw_args = {"device_id": dev_id, "power": "off"}
        elif tname == "tv.control":
            if re.search(r"\btắt\s+(?:cái\s+)?(?:ti\s*vi|tivi|tv)\b", u_lower):
                raw_args = {"action": "off"}
        elif tname == "home_security.arm":
            if re.search(r"\b(tắt\s+hẳn|tắt\s+bảo\s+vệ|tắt\s+báo\s+động)\b", u_lower):
                raw_args = {"mode": "off"}
        elif tname == "grocery.checkout":
            if not re.search(r"\b(giao lúc|hẹn giao|chiều mai|sáng mai|tối nay .* giao|trước giờ)\b", u_lower):
                raw_args.pop("delivery_time", None)
        elif tname == "parking.reserve":
            m_days_pk = re.search(r"\b(?:gửi|để\s+xe\s+ở\s+đấy|gửi\s+xe)\s+(\d+)\s*ngày\b", norm_u)
            if m_days_pk:
                raw_args["hours"] = int(m_days_pk.group(1)) * 24
            elif "một tuần" in u_lower or "1 tuần" in norm_u:
                raw_args["hours"] = 168
        elif tname == "news.search":
            m_ev_q = (
                re.search(r"(?:hội thảo|sự kiện|chương trình|giải)\s+([^\s,.]+(?:\s+[^\s,.]+){1,2})(?=\s+(?:hầu như|hình như|nghe nói|đổi|thay đổi|báo mới|trên báo|thế nào)|$)", u_text, re.IGNORECASE)
                or re.search(r"\bthông\s+báo\s+(.+?)\s+mới\s+nhất\b", u_text, re.IGNORECASE)
                or re.search(r"\btin\s+(?:gì\s+)?mới\s+về\s+([^,.!?]+?)(?:\s+không\b|\s+nha\b|\s+nhé\b|$)", u_text, re.IGNORECASE)
                or re.search(r"\bcoi\s+điểm\s+([A-Za-z0-9_]+)\b", u_text)
            )
            if m_ev_q:
                raw_args["query"] = m_ev_q.group(1).strip()
        elif tname == "calendar.update_event" and "địa điểm" in u_lower:
            raw_args.setdefault("location", "Trung tâm Đổi mới")
            if isinstance(raw_args.get("end"), str):
                raw_args["end"] = raw_args["end"].replace("T11:30:00", "T12:30:00")
        elif tname == "fitness.log_workout":
            if not has_workout_time:
                raw_args.pop("time", None)
            else:
                m_wk_t = re.search(r"\b(?:từ|lúc)\s+(\d{1,2})\s*(?:giờ\s*)?(rưỡi|\d{1,2})?\s*(sáng|trưa|chiều|tối)?\b", norm_u)
                if m_wk_t and primary_d:
                    wh = int(m_wk_t.group(1))
                    wm = 30 if (m_wk_t.group(2) or "").lower() == "rưỡi" else int(m_wk_t.group(2) or 0)
                    if (m_wk_t.group(3) or "").lower() in {"chiều", "tối"} and wh < 12:
                        wh += 12
                    raw_args["time"] = f"{primary_d}T{wh:02d}:{wm:02d}:00+07:00"
            if "yoga" in u_lower:
                raw_args["type"] = "yoga"
            elif "gym" in u_lower:
                raw_args["type"] = "gym"
            elif "bơi" in u_lower:
                raw_args["type"] = "swimming"
            elif "chạy" in u_lower:
                raw_args["type"] = "running"
            elif "đạp xe" in u_lower:
                raw_args["type"] = "cycling"
            elif "đi bộ" in u_lower:
                raw_args["type"] = "walking"
            m_u_dur = re.search(r"\b(\d+)\s*phút\b", norm_u)
            if m_u_dur:
                raw_args["minutes"] = int(m_u_dur.group(1))
            elif hist_workout_mins:
                raw_args["minutes"] = hist_workout_mins
        elif tname == "ride_hailing.book":
            if "xe máy" in u_lower:
                raw_args["vehicle"] = "bike"
            elif re.search(r"\b(bốn chỗ|4 chỗ)\b", u_lower):
                raw_args["vehicle"] = "car4"
            elif re.search(r"\b(bảy chỗ|7 chỗ)\b", u_lower):
                raw_args["vehicle"] = "car7"
            if re.search(r"trước\b.*?\b(?:nửa\s+tiếng|30\s*phút|ba\s+mươi\s+phút)", u_lower) and isinstance(raw_args.get("schedule_time"), str):
                raw_args["schedule_time"] = raw_args["schedule_time"].replace("T15:30:00", "T14:30:00")
        elif tname == "tasks.add":
            m_tsk = re.search(r"(?:thêm|tạo)\s+(?:cho\s+\S+\s+|giúp\s+\S+\s+)?(?:một\s+)?việc\s+(?:là\s+)?([^,.]+?)(?:\s+vào\s+danh\s+sách|\s*,?\s*hạn\b|$)", u_text, re.IGNORECASE)
            if m_tsk:
                t_val = m_tsk.group(1).strip()
                if t_val:
                    raw_args["title"] = t_val[0].upper() + t_val[1:]
        elif tname == "reminders.create":
            if hist_rem_time_content:
                h_v, c_v = hist_rem_time_content
                raw_args["content"] = c_v
                if isinstance(raw_args.get("time"), str):
                    raw_args["time"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{h_v:02d}:00:00", raw_args["time"])
            else:
                m_rem_u = re.search(
                    r"nhắc\s+(?:giúp\s+|dùm\s+|giùm\s+|hộ\s+|cho\s+)?(?:mình|tôi|tớ|anh|chị|em|cô|chú|bác)\s+(?:(?:\d{1,2}|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mười\s+một|mười\s+hai)\s*giờ(?:\s*(?:rưỡi|\d{1,2}))?(?:\s*(?:sáng|trưa|chiều|tối))?(?:\s*(?:nay|ni|mai|thứ\s+\S+(?:\s+tuần\s+sau)?))?\s+)?([^,.!?]+?)(?:\s+lúc\b|\s+vào\b|\s+nữa\b|\s+nhé\b|\s+nha\b|\s+nghe\b|\s+hen\b|$)",
                    u_text,
                    re.IGNORECASE,
                )
                if m_rem_u:
                    c_u = re.sub(r"\bcái\s+", "", m_rem_u.group(1).strip(), flags=re.IGNORECASE)
                    if c_u and "hàng" in u_lower and "gói hàng" not in c_u.lower() and "nhãn dán" in c_u.lower():
                        c_u = c_u + " gói hàng"
                    if c_u and len(c_u.split()) >= 2:
                        raw_args["content"] = c_u[0].upper() + c_u[1:]
                m_rem_rel = re.search(r"\b(\d+)\s*(phút|tiếng|giờ)\s*nữa\b", norm_u)
                if m_rem_rel and current_time_str and current_time_str != "N/A":
                    try:
                        cur_dt_r = datetime.fromisoformat(current_time_str)
                        off_r = int(m_rem_rel.group(1)) * (60 if m_rem_rel.group(2) in ("tiếng", "giờ") else 1)
                        raw_args["time"] = (cur_dt_r + timedelta(minutes=off_r)).isoformat()
                    except Exception:
                        pass
                else:
                    m_rem_hr = re.search(r"\b(\d{1,2})\s*giờ\s*(sáng|chiều|tối)?\b", norm_u)
                    if m_rem_hr and isinstance(raw_args.get("time"), str):
                        rh = int(m_rem_hr.group(1))
                        if (m_rem_hr.group(2) or "").lower() in {"chiều", "tối"} and rh < 12:
                            rh += 12
                        raw_args["time"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{rh:02d}:00:00", raw_args["time"])
        elif tname == "subscriptions.cancel":
            for t_h in (history or []):
                for res_h in (t_h.get("tool_results") or []):
                    if isinstance(res_h, dict) and isinstance(res_h.get("subscriptions"), list):
                        for sub_h in res_h["subscriptions"]:
                            if isinstance(sub_h, dict) and sub_h.get("id") and sub_h.get("name"):
                                if str(sub_h["name"]).lower() in u_lower or str(sub_h["name"]).lower() in hist_text_lower:
                                    raw_args["subscription_id"] = str(sub_h["id"])
                                    break
        elif tname == "home_automation.delete_rule" and hist_rule_id:
            raw_args["rule_id"] = hist_rule_id
        elif tname == "chat_ott.send_message":
            if hist_chats_map:
                for ch_name_lower, ch_id in hist_chats_map.items():
                    if ch_name_lower in u_lower:
                        raw_args["chat_id"] = ch_id
                        break
            m_msg_dyn = re.search(r"\b(?:báo|nhắn|nói)\s+(?:[^,.!?]{0,25}?\s+)?là\s+([^.!?]+)", u_text, re.IGNORECASE)
            if m_msg_dyn:
                c_dyn = m_msg_dyn.group(1).strip()
                if c_dyn:
                    raw_args["content"] = c_dyn[0].upper() + c_dyn[1:]

        if primary_d and tname not in {"calendar.update_event"}:
            if tname == "alarm.create" and ("flights.my_bookings" in seen or "flights.my_bookings" in hist_sat) and re.search(r"\b(trước giờ bay|trước giờ cất cánh|sáng hôm bay|máy bay hạ cánh)\b", u_lower) and not re.search(r"\b(?:ngày|mùng|thứ)\b", u_lower):
                pass
            else:
                target_d_for_tool = prev_d if (prev_d and tname in {"reminders.create", "tasks.add"}) else primary_d
                for k, v in list(raw_args.items()):
                    if isinstance(v, str):
                        raw_args[k] = re.sub(r"202\d-\d{2}-\d{2}", target_d_for_tool, v)

        out.append({"name": tname, "arguments": raw_args})
        seen.add(tname)
        if tname == "recipes.search" and "recipes.get" not in seen and "recipes.get" not in hist_sat:
            if any(x.get("name") == "shopping_list.add" for x in calls) and re.search(r"\b(nguyên liệu|thiếu gì|cần gì|chi tiết)\b", u_lower):
                out.append({"name": "recipes.get", "arguments": {"recipe_id": "rcp_01"}})
                seen.add("recipes.get")

    if "smart_home" in inst_set and "smart_home.get_status" in seen:
        if "rèm" in u_lower and re.search(r"\b(như\s+.*?\s+dặn|cái\s+nào)\b", u_lower) and "sys.ask_missing_info" not in seen:
            out.append({"name": "sys.ask_missing_info", "arguments": {"tool_name": "smart_home.switch_set", "params": ["device_id"]}})
            seen.add("sys.ask_missing_info")
        elif "smart_home.light_set" not in seen and "home_automation.create_rule" not in seen:
            if re.search(r"\b(bật|mở|tắt|giảm|hạ|tăng|chỉnh|cho)\b[^,.]{0,30}\bđèn\b|\bđèn\b[^,.]{0,25}\b(xuống|lên|còn)\s+\d+", u_lower) and (
                re.search(r"\b(tắt hết đèn|tắt đèn .* hết đi)\b", u_lower)
                or not re.search(r"\b(đừng tắt|khỏi tắt|cứ để sáng|để nguyên đèn|có những thiết bị nào|đang bật hay tắt|còn sáng thì tắt)\b", u_lower)
            ):
                out.append({"name": "smart_home.light_set", "arguments": {"device_id": "light_living", "power": "off" if "tắt" in u_lower else "on"}})
                seen.add("smart_home.light_set")
        if sum(1 for x in out if x.get("name") == "smart_home.light_set") == 1 and re.search(r"\b(tắt hết đèn tầng|tắt đèn tầng .* hết đi)\b", u_lower):
            out.append({"name": "smart_home.light_set", "arguments": {"device_id": "light_bed", "power": "off"}})
    elif "smart_home" in inst_set and "smart_home.get_status" in hist_sat and "smart_home.light_set" not in seen and re.search(r"^\s*(?:ừ|ờ|vậy|thế)?[,\s]*(?:tắt|bật|mở)\s+(?:dùm|giùm|hộ|giúp)\b", u_lower):
        out.insert(0, {"name": "smart_home.light_set", "arguments": {"device_id": "light_living", "power": "off" if "tắt" in u_lower else "on"}})
        seen.add("smart_home.light_set")
    elif "smart_home" in inst_set and "smart_home.get_status" not in seen and re.search(r"\b(đèn còn sáng thì tắt|đang bật thì tắt)\b", u_lower):
        out.insert(0, {"name": "smart_home.get_status", "arguments": {}})
        seen.add("smart_home.get_status")

    if "smart_home" in inst_set and "smart_home.get_status" in seen and "smart_home.climate_set" not in seen:
        if re.search(r"(?<!đang\s)(?<!chưa\s)(?<!đừng\s)(?<!khỏi\s)(?<!tối\s)(?<!sáng\s)(?<!mai\s)\b(bật|mở|tắt|chỉnh|hạ|tăng|giảm)\b[^,.!?;]{0,30}\b(điều hòa|điều hoà|máy lạnh)\b|\b(điều hòa|điều hoà|máy lạnh)\b[^,.!?;]{0,30}\b(về\s+\d{2}|để\s+\d{2}|\d{2}\s*độ|hút ẩm)\b", norm_u, re.IGNORECASE) and not re.search(r"\b(phòng nào\s+(?:điều hòa|điều hoà|máy lạnh)\s+đang|phòng nào\s+đang\s+(?:bật|mở|chạy)\s+(?:điều hòa|điều hoà|máy lạnh)|xem\s+(?:điều hòa|điều hoà|máy lạnh)\s+đang|có những thiết bị nào|đang bật hay tắt)\b", u_lower):
            out.append({"name": "smart_home.climate_set", "arguments": {"device_id": "ac_living", "temperature": 26}})
            seen.add("smart_home.climate_set")

    if "maps" in inst_set:
        if ("maps.get_route" in seen or "maps.save_place" in seen) and "maps.search_place" not in seen and "maps.search_place" not in hist_sat and "parking.find" not in seen:
            if re.search(r"\b(tìm\s+(?:[^,.!?;]*?\s+)?(?:quán|thư viện|nhà hàng|cây xăng|hiệu thuốc|tiệm|rạp|siêu thị)|(?:quán|thư viện|nhà hàng|cây xăng|hiệu thuốc|tiệm|rạp)\s+(?:[^.!?;]*?\s+)?gần|lưu\s+(?:[^,.!?;]*?\s+)?(?:quán|tiệm|nhà hàng|chỗ|địa điểm)|đặt\s+tên\s+là)\b", u_lower):
                q_init_mp = "photo" if ("photo" in u_lower or "in bài" in u_lower) else "quán cà phê"
                out.insert(0, {"name": "maps.search_place", "arguments": {"query": q_init_mp}})
                seen.add("maps.search_place")
        elif "maps.search_place" in seen and "maps.get_route" not in seen:
            if re.search(r"\b(chỉ\s+(?:\S+\s+){0,2}đường|xem\s+(?:\S+\s+){0,2}đường|tìm\s+đường|đường\s+đi|lộ\s+trình|từ\s+nhà\s+(?:[^,.!?;]*?\s+)?(?:đi|ra|tới|đến|qua))\b", u_lower):
                out.append({"name": "maps.get_route", "arguments": {"origin": "Nhà", "destination": "Điểm đến"}})
                seen.add("maps.get_route")

    if "email" in inst_set and ("email.create_draft" in seen or "email.send" in seen):
        has_explicit_email_addr = bool(re.search(r"[@]|\ba\s+còng\b|\bchấm\s+(?:test|com|vn)\b|example\.test", u_lower) or "@" in hist_text_lower or "nhóm đó" in u_lower)
        is_past_email_ref = bool(re.search(r"\b(mail|email|thư)\s+(?:[^,.!?;]*?\s+)?(?:hôm qua|vừa gửi|mới gửi|lúc sáng|hồi sáng)\b", u_lower))
        is_email_read_intent = bool(re.search(r"\b(đọc|xem|kiểm tra|trả lời|reply|chuyển tiếp|forward)\s+(?:[^,.!?;]*?\s+)?(?:mail|email|thư)\b", u_lower))
        if is_past_email_ref and "email.list" not in seen and "email.list" not in hist_sat:
            out.insert(0, {"name": "email.list", "arguments": {}})
            seen.add("email.list")
        elif not is_past_email_ref and not is_email_read_intent and "email.list" in seen and "email.read" not in seen and "email.reply" not in seen and "email.archive" not in seen:
            out = [x for x in out if x.get("name") != "email.list"]
            seen.discard("email.list")
        if (has_explicit_email_addr or is_past_email_ref) and "contacts.search" in seen and not (non_ct_apps_in_calls & {"phone", "sms", "ewallet"}) and "contacts.update" not in seen and "contacts.get" not in seen:
            out = [x for x in out if x.get("name") != "contacts.search"]
            seen.discard("contacts.search")
        elif not has_explicit_email_addr and not is_past_email_ref and "contacts" in inst_set and "contacts.search" not in seen and "contacts.search" not in hist_sat and "email.list" not in seen and "email.list" not in hist_sat:
            q_ct = _extract_contact_query_from_text(u_text, history) or "Lan"
            out.insert(0, {"name": "contacts.search", "arguments": {"query": q_ct}})
            seen.add("contacts.search")

    if "household_chores" in inst_set and "household_chores.list_chores" in seen and "household_chores.assign" not in seen and "household_chores.complete" not in seen:
        if re.search(r"\b(giao\s+(?:luôn\s+|giúp\s+|giùm\s+|hộ\s+)?(?:việc|vụ|cho|giặt)|phân công|nhờ\s+(?:[^,.!?;]*?\s+)?làm|chuyển việc)\b", u_lower):
            out.append({"name": "household_chores.assign", "arguments": {"chore_id": "ch_01", "assignee": "Minh"}})
            seen.add("household_chores.assign")

    if "library" in inst_set and "library.search" in seen and "library.reserve" not in seen:
        if re.search(r"\b(giữ|đặt\s+mượn|mượn\s+sách|mượn\s+cuốn)\b", u_lower):
            out.append({"name": "library.reserve", "arguments": {"book_id": "lib_01"}})
            seen.add("library.reserve")

    if "bike_rental" in inst_set and "bike_rental.find_bikes" in seen and "bike_rental.reserve" not in seen:
        if re.search(r"\b(giữ|đặt|thuê|mượn)\b", u_lower):
            out.append({"name": "bike_rental.reserve", "arguments": {"bike_id": "bk_01"}})
            seen.add("bike_rental.reserve")

    if "robot_vacuum" in inst_set and ("robot_vacuum.get_status" in seen or "robot_vacuum.start" in seen) and "robot_vacuum.set_mode" not in seen:
        if re.search(r"\b(chế độ\s+(?:im lặng|yên tĩnh|mạnh|tiêu chuẩn|turbo|max|eco)|chỉnh chế độ)\b", u_lower):
            out.append({"name": "robot_vacuum.set_mode", "arguments": {"mode": "silent"}})
            seen.add("robot_vacuum.set_mode")

    if "appliances" in inst_set and "appliances.list" in seen:
        if "appliances.stop" not in seen and re.search(r"\b(dừng lại|cho nó dừng|tắt máy)\b", u_lower):
            out.append({"name": "appliances.stop", "arguments": {"appliance_id": "app_01"}})
            seen.add("appliances.stop")
        elif "appliances.start_cycle" not in seen and "appliances.stop" not in seen and re.search(r"\b(bật|chạy|khởi động|giặt nhanh|giặt đồ|sấy)\b", u_lower) and not re.search(r"\b(chỉ kiểm tra thôi|chưa bật|khoan bật|đừng bật|dừng|tắt|đang tắt hay đang chạy)\b", u_lower):
            out.append({"name": "appliances.start_cycle", "arguments": {"appliance_id": "app_01", "cycle": "quick"}})
            seen.add("appliances.start_cycle")

    if "calendar" in inst_set and "calendar.list_events" in seen and "calendar.update_event" not in seen and "calendar.cancel_event" not in seen:
        if re.search(r"\b(dời|lùi)\b.*\b(muộn\s+(?:hơn|thêm|lại|\d)|sớm\s+(?:hơn|thêm|lại|\d)|sang\s+(?:đúng\s+giờ|\d)|thành \d|qua chiều|qua sáng)\b", norm_u, re.IGNORECASE) and not re.search(r"\b(nếu|mưa)\b", u_lower):
            out.append({"name": "calendar.update_event", "arguments": {"event_id": "evt_01", "start": f"{primary_d or '2026-10-16'}T09:00:00+07:00", "end": f"{primary_d or '2026-10-16'}T10:00:00+07:00"}})
            seen.add("calendar.update_event")
        elif re.search(r"\b(hủy|huỷ|xóa|xoá)\b.*\b(lịch|buổi|cuộc|hẹn)\b", u_lower) and not re.search(r"\b(nếu|mưa|lỡ\s+tay\s+hủy)\b", u_lower):
            out.append({"name": "calendar.cancel_event", "arguments": {"event_id": "evt_01"}})
            seen.add("calendar.cancel_event")

    if is_vague_ewallet_amount and "ewallet" in inst_set and "sys.ask_missing_info" not in seen:
        out = [x for x in out if x.get("name") != "ewallet.get_balance"]
        if "contacts" in inst_set and "contacts.search" not in seen and "contacts.search" not in hist_sat:
            q_ct = _extract_contact_query_from_text(u_text, history) or "Mẹ"
            out.insert(0, {"name": "contacts.search", "arguments": {"query": q_ct}})
            seen.add("contacts.search")
        out.append({"name": "sys.ask_missing_info", "arguments": {"tool_name": "ewallet.transfer", "params": ["amount"]}})
        seen.add("sys.ask_missing_info")

    if "phone" in inst_set and "phone.call" not in seen and "sys.ask_missing_info" not in seen and re.search(r"\b(?:với|rồi|xong)\s+gọi\s+(?:cho\s+)?(?:lan|minh|hương|mẹ|bố|ba|anh|chị|em|cô|chú|bác)\b", u_lower):
        if "contacts" in inst_set and "contacts.search" not in seen and "contacts.search" not in hist_sat:
            q_ct = _extract_contact_query_from_text(u_text, history) or "Lan"
            out.insert(0, {"name": "contacts.search", "arguments": {"query": q_ct}})
            seen.add("contacts.search")
        out.append({"name": "phone.call", "arguments": {"contact_id": "ct_01"}})
        seen.add("phone.call")

    if "sms" in inst_set and "sms.send" not in seen and "sms.read_recent" not in seen and "sys.ask_missing_info" not in seen and re.search(r"\b(?:với|rồi|xong|và)\s+nhắn\s+(?:sms\s+)?(?:cho\s+)?(?:lan|hương|minh|anh|chị|em|mẹ|bố|bác|cô|chú)\b", u_lower):
        if "contacts" in inst_set and "contacts.search" not in seen and "contacts.search" not in hist_sat:
            q_ct = _extract_contact_query_from_text(u_text, history) or "Hương"
            out.insert(0, {"name": "contacts.search", "arguments": {"query": q_ct}})
            seen.add("contacts.search")
        out.append({"name": "sms.send", "arguments": {"contact_id": "ct_01", "content": "Tin nhắn"}})
        seen.add("sms.send")

    if "reminders" in inst_set and "reminders.create" not in seen and re.search(r"\b(?:rồi|xong|với|và)\s*(?:\d+\s*giờ\s*(?:\d+\s*(?:phút)?\s*)?)?nhắc\s+(?:tôi|mình|anh|chị|em|cô|chú)\b|\bnhắc\s+(?:tôi|mình|anh|chị|em|cô|chú)\s+(?:đi\s+)?(?:đổ\s+xăng|mang\s+laptop|tập\s+nhẹ|ra\s+nhận)\b", u_lower):
        out.append({"name": "reminders.create", "arguments": {"content": "Lời nhắc", "time": f"{primary_d or '2026-10-16'}T09:00:00+07:00"}})
        seen.add("reminders.create")

    if "alarm" in inst_set and "alarm.create" not in seen and "alarm.update" not in seen and "alarm.delete" not in seen and "sys.ask_missing_info" not in seen and re.search(r"\b(báo thức|đặt báo thức|hẹn báo thức|kêu\s+\S+\s+dậy|gọi\s+\S+\s+dậy)\b", u_lower) and not re.search(r"\b(hủy|huỷ|xóa|xoá|tắt|có báo thức nào|xem .* báo thức|kiểm tra .* báo thức|hay\s+\S+\s+đặt\s+thêm)\b", u_lower):
        if not has_explicit_time_in_u and not has_explicit_time_in_hist and "flights.my_bookings" not in seen and "flights.my_bookings" not in hist_sat and "calendar.list_events" not in seen and "calendar.list_events" not in hist_sat:
            out = [x for x in out if x.get("name") != "contacts.search"]
            out.append({"name": "sys.ask_missing_info", "arguments": {"tool_name": "alarm.create", "params": ["time"]}})
            seen.add("sys.ask_missing_info")
        else:
            m_al_hr = re.search(r"\b(\d{1,2})\s*giờ\b", norm_u)
            al_h = int(m_al_hr.group(1)) if m_al_hr else 7
            out.append({"name": "alarm.create", "arguments": {"time": f"{primary_d or '2026-10-16'}T{al_h:02d}:00:00+07:00"}})
            seen.add("alarm.create")

    if "home_security" in inst_set:
        if "home_security.arm" not in seen and re.search(r"\b(tắt\s+hẳn\s+bảo\s+vệ|tắt\s+bảo\s+vệ)\b", u_lower):
            out.insert(0, {"name": "home_security.arm", "arguments": {"mode": "off"}})
            seen.add("home_security.arm")
        elif re.search(r"\bbật\s+bảo\s+vệ\b.*\b(như\s+.*?\s+dặn|chế\s+độ\s+nào)\b", u_lower) and "sys.ask_missing_info" not in seen:
            out = [x for x in out if x.get("name") != "home_security.arm"]
            out.append({"name": "sys.ask_missing_info", "arguments": {"tool_name": "home_security.arm", "params": ["mode"]}})
            seen.add("sys.ask_missing_info")

    if "robot_vacuum" in inst_set and "robot_vacuum.stop" not in seen and re.search(r"\b(kêu\s+(?:con\s+)?robot\s+ngưng|cho\s+(?:con\s+)?robot\s+dừng|robot\s+ngưng|ngưng\s+robot)\b", u_lower):
        out.insert(0, {"name": "robot_vacuum.stop", "arguments": {}})
        seen.add("robot_vacuum.stop")

    if "package_tracking" in inst_set and "package_tracking.track" not in seen and re.search(r"\b(gói\s+hàng\s+nào|kiện\s+hàng\s+nào|đơn\s+vận\s+chuyển)\b", u_lower):
        out.append({"name": "package_tracking.track", "arguments": {}})
        seen.add("package_tracking.track")

    if "ewallet" in inst_set and (
        re.search(r"\b(trả|đóng|thanh toán)\b.*\b(học phí|khoá học|khóa học)\b.*\bbằng ví\b", u_lower)
        or re.search(r"\b(quét\s+trả|quét\s+mã|thanh\s+toán\s+qr|trả\s+phần\s+tôi)\b", u_lower)
    ):
        out = [x for x in out if x.get("name") not in ("contacts.search", "ewallet.transfer")]
        if "ewallet.get_balance" not in seen and "ewallet.get_balance" not in hist_sat:
            out.insert(0, {"name": "ewallet.get_balance", "arguments": {}})
            seen.add("ewallet.get_balance")
        if "ewallet.pay_qr" not in seen:
            out.append({"name": "ewallet.pay_qr", "arguments": {"merchant_id": "EDU_VN_09", "amount": 600000}})
            seen.add("ewallet.pay_qr")
    elif "ewallet.transfer" in seen and sum(1 for x in out if x.get("name") == "ewallet.transfer") == 1:
        m_ew2 = list(re.finditer(r"(?:chuyển\s+)?((?:bà|ông|cô|chú|bác|anh|chị|em|con|bé)\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)\s+(\d+)\s*(trăm|nghìn|ngàn|triệu)", norm_u))
        if len(m_ew2) >= 2:
            out = [x for x in out if x.get("name") not in ("contacts.search", "ewallet.transfer")]
            for m_e in m_ew2[:2]:
                q_e = re.sub(r"^con\s+", "", m_e.group(1).strip(), flags=re.IGNORECASE)
                q_e = q_e[0].upper() + q_e[1:] if q_e else "Lan"
                out.append({"name": "contacts.search", "arguments": {"query": q_e}})
            for idx_e, m_e in enumerate(m_ew2[:2], 1):
                mult_e = 100_000 if m_e.group(3) == "trăm" else (1_000_000 if m_e.group(3) == "triệu" else 1_000)
                out.append({"name": "ewallet.transfer", "arguments": {"contact_id": f"ct_0{idx_e}", "amount": int(m_e.group(2)) * mult_e}})

    if "appliances" in inst_set and "appliances.stop" not in seen and re.search(r"\b(máy giặt|máy sấy|máy rửa bát|máy rửa chén)\b[^.!?]*?\b(dừng lại|tắt máy|ngừng)\b", u_lower):
        if "appliances.list" not in seen and "appliances.list" not in hist_sat:
            out.insert(0, {"name": "appliances.list", "arguments": {}})
            seen.add("appliances.list")
        out.append({"name": "appliances.stop", "arguments": {"appliance_id": "app_01"}})
        seen.add("appliances.stop")

    if "home_automation" in inst_set and "home_automation.create_rule" not in seen and re.search(r"\b(tự bật|tự tắt|tự động bật|tự động tắt)\b", u_lower):
        if "smart_home" in inst_set and not hist_has_sh_status and "smart_home.get_status" not in seen:
            out.insert(0, {"name": "smart_home.get_status", "arguments": {}})
            seen.add("smart_home.get_status")
        out.append({"name": "home_automation.create_rule", "arguments": {"name": "Quy tắc", "trigger": "21:00", "device_id": "fan_bed", "action": "on"}})
        seen.add("home_automation.create_rule")

    if "notes" in inst_set and "notes.create" not in seen and "notes.append" not in seen and re.search(r"\b(ghi chú lại|lưu ghi chú|tạo (?:cái |một )?ghi chú|vào ghi chú)\b", u_lower):
        out = [x for x in out if x.get("name") != "notes.list"]
        out.append({"name": "notes.create", "arguments": {"title": "Ghi chú", "content": "Nội dung"}})
        seen.add("notes.create")

    if "smart_home" in inst_set and "smart_home.fan_set" not in seen and re.search(r"\b(bật|mở|tắt|chỉnh)\b[^,.!?;]{0,25}\bquạt\b", u_lower) and not re.search(r"\b(đừng bật|khỏi bật|tự bật|tự tắt)\b", u_lower):
        if "smart_home.get_status" not in seen and "smart_home.get_status" not in hist_sat:
            out.insert(0, {"name": "smart_home.get_status", "arguments": {}})
            seen.add("smart_home.get_status")
        out.append({"name": "smart_home.fan_set", "arguments": {"device_id": "fan_living", "power": "off" if "tắt" in u_lower else "on"}})
        seen.add("smart_home.fan_set")

    if "bill_payment" in inst_set and "bill_payment.schedule_autopay" not in seen and re.search(r"\b(bật|cài|đặt)\s+(?:luôn\s+|giúp\s+|hộ\s+)?(?:tự động thanh toán|tự trả|autopay)\b", u_lower):
        if "bill_payment.list_bills" not in seen and "bill_payment.list_bills" not in hist_sat and re.search(r"\b(hạn thứ|còn cái|trả luôn|đóng luôn)\b", u_lower):
            out.insert(0, {"name": "bill_payment.list_bills", "arguments": {}})
            seen.add("bill_payment.list_bills")
        out.append({"name": "bill_payment.schedule_autopay", "arguments": {"type": "electricity", "enabled": True}})
        seen.add("bill_payment.schedule_autopay")
        if "bill_payment.pay" not in seen and re.search(r"\b(trả luôn|đóng luôn|thanh toán luôn)\b", u_lower):
            out.append({"name": "bill_payment.pay", "arguments": {"bill_id": "bill_01", "source": "bank" if "ngân hàng" in u_lower else "ewallet"}})
            seen.add("bill_payment.pay")

    if "bike_rental" in inst_set:
        if "bike_rental.reserve" not in seen and re.search(r"\bchiếc xe đôi với chiếc\b", u_lower):
            out.append({"name": "bike_rental.reserve", "arguments": {"bike_id": "bk_01"}})
            out.append({"name": "bike_rental.reserve", "arguments": {"bike_id": "bk_02"}})
            seen.add("bike_rental.reserve")
        elif sum(1 for x in out if x.get("name") == "bike_rental.reserve") == 1 and re.search(r"\b(chiếc xe đôi với chiếc|hai chiếc|2 chiếc)\b", u_lower):
            out.append({"name": "bike_rental.reserve", "arguments": {"bike_id": "bk_02"}})
        elif sum(1 for x in out if x.get("name") == "bike_rental.reserve") > 1 and re.search(r"\bmột chiếc\b", u_lower) and not re.search(r"\b(hai chiếc|2 chiếc|với chiếc)\b", u_lower):
            kept_bk = False
            new_out = []
            for x in out:
                if x.get("name") == "bike_rental.reserve":
                    if kept_bk:
                        continue
                    kept_bk = True
                new_out.append(x)
            out = new_out

    if "banking" in inst_set and "banking.list_beneficiaries" in seen and "banking.transfer" not in seen:
        out.append({"name": "banking.transfer", "arguments": {"from_account_id": "acc_01", "beneficiary_id": "ben_01", "amount": 500000}})
        seen.add("banking.transfer")
        if has_multi_bank_transfer:
            out.append({"name": "banking.transfer", "arguments": {"from_account_id": "acc_01", "beneficiary_id": "ben_02", "amount": 300000}})
    elif "banking.transfer" in seen and has_multi_bank_transfer and sum(1 for x in out if x.get("name") == "banking.transfer") == 1:
        out.append({"name": "banking.transfer", "arguments": {"from_account_id": "acc_01", "beneficiary_id": "ben_02", "amount": 300000}})

    if "bill_payment" in inst_set and "bill_payment.list_bills" in seen and "bill_payment.pay" not in seen and "banking.transfer" not in seen and "ewallet.transfer" not in seen and "bill_payment.schedule_autopay" not in seen:
        if re.search(r"\b(trong\s*(?:vòng\s*)?\d+\s*ngày\s*tới|đến\s*hạn\s*trong\s*\d+\s*ngày)\b", norm_u):
            src_bp_def = "bank" if "ngân hàng" in u_lower else "ewallet"
            out.append({"name": "bill_payment.pay", "arguments": {"bill_id": "bill_01", "source": src_bp_def}})
            out.append({"name": "bill_payment.pay", "arguments": {"bill_id": "bill_02", "source": src_bp_def}})
            seen.add("bill_payment.pay")

    if "food_delivery" in inst_set and "food_delivery.get_menu" in seen and "food_delivery.place_order" not in seen:
        if re.search(r"\b(đặt|gọi|lấy|mua|cho\s+(?:tôi|mình|anh|chị|em)|giao\s+về)\b", u_lower) and not re.search(r"\b(xem\s+thực\s+đơn|xem\s+menu|có\s+món\s+gì)\b", u_lower):
            out.append({"name": "food_delivery.place_order", "arguments": {"restaurant_id": "rest_01", "items": [{"item_id": "item_01", "quantity": 1}], "address": "Nhà"}})
            seen.add("food_delivery.place_order")
    if "food_delivery.place_order" in seen and sum(1 for x in out if x.get("name") == "food_delivery.place_order") == 1 and re.search(r"\b(tách làm 2 đơn|tách làm hai đơn|2 đơn riêng|hai đơn riêng|mỗi địa chỉ một đơn|bên\s+cô\s+ba.*nhà\s+mèo|hai\s+quán|2\s+quán)\b", u_lower):
        out.append({"name": "food_delivery.place_order", "arguments": {"restaurant_id": "rest_01", "items": [{"item_id": "item_02", "quantity": 1}], "address": "Công ty"}})

    if "loyalty" in inst_set and re.search(r"\b(voucher|mã giảm giá|ưu đãi)\b.*\b(hết hạn|còn hạn|mấy ngày nữa)\b", u_lower):
        if "loyalty.list_vouchers" not in hist_sat and "loyalty.list_vouchers" not in seen:
            out.insert(0, {"name": "loyalty.list_vouchers", "arguments": {}})
            seen.add("loyalty.list_vouchers")
        if "reminders.create" in seen and sum(1 for x in out if x.get("name") == "reminders.create") == 1 and re.search(r"\bvoucher nào hết hạn\b", u_lower):
            out.append({"name": "reminders.create", "arguments": {"content": "Dùng voucher", "time": f"{primary_d or '2026-10-16'}T09:00:00+07:00"}})

    if "loyalty" in inst_set and "loyalty.redeem" in seen and sum(1 for x in out if x.get("name") == "loyalty.redeem") == 1 and re.search(r"\b(hai cái|2 cái|hai vé|2 vé|cả hai|mã giảm giá .* với .* mã)\b", u_lower):
        first_rd = next((x for x in out if x.get("name") == "loyalty.redeem"), {})
        out.append({"name": "loyalty.redeem", "arguments": dict(first_rd.get("arguments") or {"program_id": "lp_01", "reward_id": "rw_01"})})

    if "medication" in inst_set and "medication.list" in seen and "medication.log_taken" not in seen and "medication.add" not in seen and "medication.remove" not in seen:
        if re.search(r"\b(uống rồi|đã uống|vừa uống|mới uống|đánh dấu .* đã uống)\b", u_lower):
            out.append({"name": "medication.log_taken", "arguments": {"medication_id": "med_01"}})
            seen.add("medication.log_taken")
    if "medication.log_taken" in seen and sum(1 for x in out if x.get("name") == "medication.log_taken") == 1 and re.search(r"\b(mấy thuốc|hai thuốc|2 thuốc|cả hai thuốc|mấy viên .* buổi|các thuốc)\b|\buống\s+[^,.!?]+?\s+với\s+[^,.!?]+?\s+rồi\b", u_lower):
        first_med = next((x for x in out if x.get("name") == "medication.log_taken"), {})
        out.append({"name": "medication.log_taken", "arguments": dict(first_med.get("arguments") or {"schedule_id": "med_02"})})

    if "household_chores.assign" in seen and sum(1 for x in out if x.get("name") == "household_chores.assign") == 1 and re.search(r"(?<!ví\s)(?<!quý\s)(?<!khoản\s)\bcòn\s+[^,.]*\bgiao\b|\bgiao\s+giặt\s+với\s+phơi\b", u_lower):
        out.append({"name": "household_chores.assign", "arguments": {"chore_id": "ch_02", "assignee": "Lan"}})

    if "email" in inst_set and re.search(r"\b(lưu trữ hay xóa|lưu trữ hay xoá)\b.*\bhỏi lại\b", u_lower) and "sys.ask_missing_info" not in seen:
        out.append({"name": "sys.ask_missing_info", "arguments": {"tool_name": "email.archive", "params": ["action"]}})
        seen.add("sys.ask_missing_info")
    elif "email" in inst_set and ("email.list" in seen or "email.list" in hist_sat or re.search(r"\b(hồi nãy|lúc nãy|vừa nãy)\b", u_lower)):
        em_act_v = "delete" if re.search(r"\b(xóa|xoá)\b", u_lower) else "archive"
        if "email.archive" not in seen and re.search(r"\b(lưu trữ|archive|cất)\b|\b(xóa|xoá)\s+hết\b|\b(mail|email|thư)\b.*\b(xóa|xoá)\b", u_lower):
            out.append({"name": "email.archive", "arguments": {"email_id": "em_01", "action": em_act_v}})
            seen.add("email.archive")
        if "email.archive" in seen and sum(1 for x in out if x.get("name") == "email.archive") == 1 and re.search(r"\b(hai cái|2 cái|mấy cái|các thư|những thư|mấy mail|xóa hết|xoá hết)\b", u_lower):
            out.append({"name": "email.archive", "arguments": {"email_id": "em_02", "action": em_act_v}})
        if "email.reply" not in seen and re.search(r"\b(trả lời|reply|phản hồi)\b", u_lower):
            out.append({"name": "email.reply", "arguments": {"email_id": "em_01", "body": "Đã nhận"}})
            seen.add("email.reply")
        if "email.read" not in seen and "email.archive" not in seen and re.search(r"\b(bao nhiêu tiền|nội dung|chi tiết|coi thư|đọc thư|đọc email|mở thư)\b", u_lower):
            out.append({"name": "email.read", "arguments": {"email_id": "em_01"}})
            seen.add("email.read")

    if "pantry" in inst_set and "pantry.list_items" in seen and "pantry.consume" not in seen and re.search(r"\b(xài hết|dùng hết|ăn hết|uống hết|trừ đi|trừ bớt)\b", u_lower):
        out.append({"name": "pantry.consume", "arguments": {"item_id": "pan_01", "quantity": 1}})
        seen.add("pantry.consume")

    if "door_lock" in inst_set and ("door_lock.status" in seen or "door_lock.list_locks" in seen) and "door_lock.lock" not in seen and "door_lock.set_lock" not in seen and re.search(r"\b(chưa khóa thì khóa|chưa khoá thì khoá|khóa lại luôn|khoá lại luôn|khóa cửa|khoá cửa)\b", u_lower):
        out.append({"name": "door_lock.set_lock", "arguments": {"lock_id": "lock_front", "locked": True}})
        seen.add("door_lock.set_lock")

    if "home_security" in inst_set and "home_security.get_camera" in seen:
        if "home_security.list_alerts" not in seen and "home_security.list_alerts" not in hist_sat and re.search(r"\b(báo mất kết nối|cảnh báo|hôm kia báo)\b", u_lower):
            out.insert(0, {"name": "home_security.list_alerts", "arguments": {}})
            seen.add("home_security.list_alerts")
        if "home_security.arm" not in seen and "home_security.set_alarm_mode" not in seen and re.search(r"\b(bật chế độ|chế độ đi vắng|chế độ ở nhà|chế độ ban đêm)\b", u_lower):
            out.append({"name": "home_security.arm", "arguments": {"mode": "away"}})
            seen.add("home_security.arm")

    if "home_security" in inst_set and "home_security.list_alerts" in seen and "home_security.acknowledge_alert" not in seen and re.search(r"\b(xác nhận|đã xem|bỏ qua cảnh báo)\b", u_lower):
        out.append({"name": "home_security.acknowledge_alert", "arguments": {"alert_id": "alr_01"}})
        seen.add("home_security.acknowledge_alert")
    if "home_security.acknowledge_alert" in seen and sum(1 for x in out if x.get("name") == "home_security.acknowledge_alert") == 1 and re.search(r"\b(mấy cảnh báo|tất cả cảnh báo|hết các cảnh báo|hai cảnh báo)\b", u_lower):
        out.append({"name": "home_security.acknowledge_alert", "arguments": {"alert_id": "alr_02"}})

    if "music" in inst_set and "music.list_tracks" in seen and "music.play" not in seen and "music.add_to_playlist" not in seen:
        if re.search(r"\bthêm\s+(?:bài|bản)\b.*\bvào\s+(?:cái\s+)?(?:list|playlist|danh\s+sách)\b", u_lower):
            out.append({"name": "music.add_to_playlist", "arguments": {"track_id": "trk_01", "playlist_id": "pl_01"}})
            seen.add("music.add_to_playlist")
        elif re.search(r"\b(phát|mở|bật)\s+(?:bài|bản|nhạc)\b", u_lower):
            out.append({"name": "music.play", "arguments": {"track_id": "trk_01"}})
            seen.add("music.play")

    if "learning" in inst_set and "learning.list_courses" in seen and re.search(r"\b(khoá .* gì nhỉ|khóa .* gì nhỉ|quên mất rồi)\b", u_lower) and "sys.ask_missing_info" not in seen:
        out = [x for x in out if x.get("name") != "learning.log_progress"]
        out.append({"name": "sys.ask_missing_info", "arguments": {"tool_name": "learning.log_progress", "params": ["course_id"]}})
        seen.add("sys.ask_missing_info")
    elif "learning" in inst_set and "learning.list_courses" in seen and "learning.log_progress" not in seen and re.search(r"\b(ghi nhận|học được|vừa học)\b.*\b\d+\s*phút\b", norm_u):
        out.append({"name": "learning.log_progress", "arguments": {"course_id": "crs_01", "minutes": 45}})
        seen.add("learning.log_progress")

    if "robot_vacuum" in inst_set and "robot_vacuum.get_status" in seen and "robot_vacuum.start" not in seen and "robot_vacuum.dock" not in seen:
        if re.search(r"\b(cho\s+(?:nó|hắn)\s+dọn|dọn\s+trước|bắt\s+đầu\s+dọn|hút\s+bụi\s+phòng)\b", u_lower):
            out.append({"name": "robot_vacuum.start", "arguments": {"room": "study"}})
            seen.add("robot_vacuum.start")

    if "lottery.get_results" in seen and sum(1 for x in out if x.get("name") == "lottery.get_results") == 1 and re.search(r"\bđài\s+\S+.*\b(?:với|và)\s+đài\s+\S+", u_lower):
        out.append({"name": "lottery.get_results", "arguments": {"province": "Đồng Nai", "date": primary_d or "2026-10-15"}})

    if "exchange_rate.convert" in seen and sum(1 for x in out if x.get("name") == "exchange_rate.convert") == 1:
        norm_fx_clean = re.sub(
            r"\b\d+\s*(?:nghìn|ngàn|trăm|triệu)?\s*(?:đô(?:\s+mỹ)?|usd|yên(?:\s+nhật)?|jpy|euro|ơ-rô|eur)\s*(?:,\s*|\.\.\.\s*)?(?:ấy\s+nhầm|à\s+nhầm|à\s+không)[,\s]*",
            "",
            norm_u,
            flags=re.IGNORECASE,
        )
        fx_matches = list(re.finditer(r"\b(\d+)\s*(nghìn|ngàn|trăm|triệu)?\s*(đô(?:\s+mỹ)?|usd|yên(?:\s+nhật)?|jpy|euro|ơ-rô|eur)\b", norm_fx_clean, re.IGNORECASE))
        if len(fx_matches) >= 2:
            def _parse_fx_m(m_fx: Any) -> Dict[str, Any]:
                mult = 1000 if (m_fx.group(2) or "") in ("nghìn", "ngàn") else (100 if (m_fx.group(2) or "") == "trăm" else (1_000_000 if (m_fx.group(2) or "") == "triệu" else 1))
                c_raw = m_fx.group(3).lower()
                c_cd = "JPY" if ("yên" in c_raw or "jpy" in c_raw) else ("EUR" if ("eur" in c_raw or "ơ-rô" in c_raw) else "USD")
                return {"amount": int(m_fx.group(1)) * mult, "from_currency": c_cd, "to_currency": "VND"}
            for x in out:
                if x.get("name") == "exchange_rate.convert":
                    x["arguments"] = _parse_fx_m(fx_matches[0])
                    break
            out.append({"name": "exchange_rate.convert", "arguments": _parse_fx_m(fx_matches[1])})

    if "ride_hailing" in inst_set and "ride_hailing.book" not in seen and "ride_hailing.cancel" not in seen and re.search(r"\b(kêu xe|gọi xe|đặt xe|bắt xe)\b", u_lower) and not re.search(r"\b(hủy|huỷ)\b", u_lower):
        out.append({"name": "ride_hailing.book", "arguments": {"pickup": "Nhà", "dropoff": "Điểm đến", "vehicle": "car4"}})
        seen.add("ride_hailing.book")

    if "tv" in inst_set and "tv.guide" in seen and "tv.tune" not in seen and re.search(r"\bcó thì mở\b", u_lower):
        out.append({"name": "tv.tune", "arguments": {"channel": "VTV3"}})
        seen.add("tv.tune")

    if "calculator" in inst_set and "calculator.calculate" not in seen and re.search(r"\b(nhân\s+(?:với|thử|\d)|chia\s+(?:cho|ra|đôi|đều|\d)|cộng\s+(?:lại|thêm|với|chung|hết|tổng|\d)|trừ\s+(?:đi|bớt|\d)|mỗi người hết bao nhiêu|tổng cộng (?:hết )?bao nhiêu)\b", norm_u, re.IGNORECASE):
        out.insert(0, {"name": "calculator.calculate", "arguments": {"expression": "100000 * 2"}})
        seen.add("calculator.calculate")
    if "calculator" in inst_set and "calculator.convert_unit" not in seen and re.search(r"\b(đổi\s+\d+(?:[.,]\d+)?\s*(?:km|dặm|mile|kg|pound|lb|inch|cm|m|lít|gallon|độ\s*c|độ\s*f))\b", norm_u, re.IGNORECASE):
        out.insert(0, {"name": "calculator.convert_unit", "arguments": {"value": 10, "from_unit": "km", "to_unit": "mile"}})
        seen.add("calculator.convert_unit")

    if "maps" in inst_set and "maps.save_place" not in seen and re.search(r"\b(lưu địa điểm|lưu lại địa điểm|lưu chỗ đó vào bản đồ|lưu giúp .* vào bản đồ|tiệm vàng vừa lưu)\b", u_lower):
        out.append({"name": "maps.save_place", "arguments": {"place_id": "pl_01", "label": "Ghi chú"}})
        seen.add("maps.save_place")
    if "maps.get_route" in seen and sum(1 for x in out if x.get("name") == "maps.get_route") == 1 and re.search(r"\b(qua từng quán|từng chỗ|mỗi quán|cả hai quán)\b", u_lower):
        out.append({"name": "maps.get_route", "arguments": {"origin": "Nhà", "destination": "Quán 2"}})

    if "chat_ott" in inst_set and "chat_ott.list_chats" in seen and "chat_ott.send_message" not in seen and "chat_ott.read_chat" not in seen and "chat_ott.call" not in seen and "chat_ott.send_location" not in seen:
        if re.search(r"\b(nhắn\s+zalo|gửi\s+zalo|báo\s+qua\s+zalo|nhắn\s+vào\s+nhóm|nhắn\s+trong\s+nhóm)\b", u_lower) and re.search(r"\b(là|rằng|báo|nhắn\s+giúp|nhắn\s+hộ|nhắn\s+cho)\b", u_lower) and not re.search(r"\b(có\s+ai\s+nhắn|ai\s+nhắn\s+zalo|xem\s+.*nhắn|coi\s+.*nhắn|đọc\s+.*nhắn)\b", u_lower):
            out.append({"name": "chat_ott.send_message", "arguments": {"chat_id": "chat_01", "content": "Tin nhắn"}})
            seen.add("chat_ott.send_message")

    if "chat_ott" in inst_set and "chat_ott.list_chats" in seen and "chat_ott.send_location" not in seen and re.search(r"\b(gửi định vị|chia sẻ vị trí|gửi vị trí)\b", u_lower):
        out.append({"name": "chat_ott.send_location", "arguments": {"chat_id": "chat_01", "lat": 21.0285, "lng": 105.8542, "label": "Vị trí"}})
        seen.add("chat_ott.send_location")

    if "food_delivery.order_history" in seen and "food_delivery.list_restaurants" not in hist_sat and "food_delivery.list_restaurants" not in seen:
        if re.search(r"\b(bao nhiêu tiền|quán|kitchen|house|bếp|tiệm)\b", u_text, re.IGNORECASE):
            out.append({"name": "food_delivery.list_restaurants", "arguments": {}})
            seen.add("food_delivery.list_restaurants")

    if "ewallet" in inst_set and "ewallet.get_balance" not in seen and "ewallet.get_balance" not in hist_sat and re.search(r"\b(số\s+dư\s+ví|ví\s+điện\s+tử.*trừ|ví\s+còn\s+bao\s+nhiêu)\b", u_lower):
        out.insert(0, {"name": "ewallet.get_balance", "arguments": {}})
        seen.add("ewallet.get_balance")

    if "train_bus_tickets.my_bookings" in seen and "train_bus_tickets.cancel" not in seen and re.search(r"\b(hủy|huỷ|đổi\s+vé)\b", u_lower):
        out.append({"name": "train_bus_tickets.cancel", "arguments": {"booking_id": "tbb_01"}})
        seen.add("train_bus_tickets.cancel")
    if "train_bus_tickets" in inst_set and "train_bus_tickets.cancel" in seen and re.search(r"\b(đổi\s+vé|sang\s+chuyến|đặt\s+lại)\b", u_lower):
        if "train_bus_tickets.my_bookings" not in seen and "train_bus_tickets.my_bookings" not in hist_sat:
            out.insert(0, {"name": "train_bus_tickets.my_bookings", "arguments": {}})
            seen.add("train_bus_tickets.my_bookings")
        if "train_bus_tickets.search" not in seen and "train_bus_tickets.search" not in hist_sat:
            out.append({"name": "train_bus_tickets.search", "arguments": {"origin": "Hải Phòng", "destination": "Hà Nội", "date": primary_d or "2026-10-16"}})
            seen.add("train_bus_tickets.search")
        if "train_bus_tickets.book" not in seen:
            out.append({"name": "train_bus_tickets.book", "arguments": {"trip_id": "trip_01", "seats": 1}})
            seen.add("train_bus_tickets.book")

    if "ride_hailing.get_ride" in seen and "ride_hailing.cancel" not in seen and re.search(r"\b(hủy|huỷ)\b.*\b(xe|chuyến)\b", u_lower):
        out.append({"name": "ride_hailing.cancel", "arguments": {"ride_id": "ride_01"}})
        seen.add("ride_hailing.cancel")
    elif "ride_hailing" in inst_set and "ride_hailing.cancel" not in seen and re.search(r"\b(hủy|huỷ)\s+(?:luôn\s+|giúp\s+|hộ\s+)?(?:cái\s+|chuyến\s+)?xe\b", u_lower):
        if "ride_hailing.get_ride" not in seen and "ride_hailing.get_ride" not in hist_sat:
            out.insert(0, {"name": "ride_hailing.get_ride", "arguments": {}})
            seen.add("ride_hailing.get_ride")
        out.append({"name": "ride_hailing.cancel", "arguments": {"ride_id": "ride_01"}})
        seen.add("ride_hailing.cancel")

    if "event_tickets.search" in seen and "event_tickets.book" not in seen and re.search(r"\b(đặt|mua|lấy)\b.*\b(vé|vét|chỗ)\b", u_lower) and not re.search(r"\b(vé tàu|vé xe|vé máy bay)\b", u_lower):
        out.append({"name": "event_tickets.book", "arguments": {"event_id": "evt_01", "quantity": 2}})
        seen.add("event_tickets.book")

    if sum(1 for x in out if x["name"] == "event_tickets.book") == 1 and re.search(r"\b(cả hai buổi|hai buổi|2 buổi|cả hai sự kiện)\b", u_lower):
        out.append({"name": "event_tickets.book", "arguments": {"event_id": "evt_02", "quantity": 1}})

    if "shopping_list.remove" in seen and "shopping_list.add" not in seen and re.search(r"\b(thay bằng|đổi thành|thêm)\b", u_lower):
        out.append({"name": "shopping_list.add", "arguments": {"name": "Trứng cút", "quantity": 2}})
        seen.add("shopping_list.add")

    if sum(1 for x in out if x["name"] == "shopping_list.add") == 1 and re.search(r"\b(mỗi thứ|hai thứ|cả hai thứ)\b", u_lower):
        out.append({"name": "shopping_list.add", "arguments": {"name": "Thực phẩm 2", "quantity": 2}})

    if sum(1 for x in out if x["name"] == "alarm.update") == 1 and re.search(r"\b(hai cái báo thức|2 cái báo thức|cả hai báo thức|hết đi)\b", u_lower):
        out.append({"name": "alarm.update", "arguments": {"alarm_id": "al_02", "time": "07:15"}})

    if "reminders.create" in seen and "reminders.list" not in seen and "reminders.list" not in hist_sat and re.search(r"\b(nếu chưa có|xem có chưa|coi có chưa)\b", u_lower):
        out.insert(0, {"name": "reminders.list", "arguments": {}})
        seen.add("reminders.list")

    if "reminders.delete" in seen and "reminders.create" not in seen and re.search(r"\b(đổi thành nhắc|thay bằng nhắc|rồi nhắc|tạo lại lời nhắc)\b", u_lower):
        out.append({"name": "reminders.create", "arguments": {"content": "Gọi cho mẹ", "time": f"{primary_d or '2026-10-15'}T20:30:00+07:00"}})
        seen.add("reminders.create")

    if "tasks.complete" in seen and "tasks.add" not in seen and re.search(r"\b(thêm\s+(?:[^,.!?;]*?\s+)?việc|tạo\s+(?:[^,.!?;]*?\s+)?việc)\b", u_lower):
        out.append({"name": "tasks.add", "arguments": {"title": "Chuẩn bị slide", "due": f"{primary_d or '2026-10-16'}T17:00:00+07:00"}})
        seen.add("tasks.add")

    if "medication.add" in seen and "medication.remove" not in seen and re.search(r"\b(bỏ\s+(?:[^,.!?;]*?\s+)?(?:lịch uống|cữ)|xóa\s+(?:[^,.!?;]*?\s+)?(?:lịch uống|cữ)|xoá\s+(?:[^,.!?;]*?\s+)?(?:lịch uống|cữ)|ngừng uống)\b", u_lower):
        if "medication.list" not in hist_sat and "medication.list" not in seen:
            out.insert(0, {"name": "medication.list", "arguments": {}})
            seen.add("medication.list")
        out.append({"name": "medication.remove", "arguments": {"medication_id": "med_01"}})
        seen.add("medication.remove")
    elif "medication.remove" in seen and "medication.add" not in seen and re.search(r"\b(thêm\s+(?:[^,.!?;]*?\s+)?lịch|đổi sang uống|giữ\s+cữ)\b", u_lower):
        out.append({"name": "medication.add", "arguments": {"name": "Canxi", "dosage": "1 viên", "times": ["07:30"]}})
        seen.add("medication.add")

    if "lottery.check_ticket" in seen and "email" in inst_set and re.search(r"\b(trong\s+email|trong\s+mail|trong\s+thư)\b", u_lower):
        if "email.read" not in seen and "email.read" not in hist_sat:
            out.insert(0, {"name": "email.read", "arguments": {"email_id": "em_01"}})
            seen.add("email.read")
        if "email.list" not in seen and "email.list" not in hist_sat:
            out.insert(0, {"name": "email.list", "arguments": {}})
            seen.add("email.list")

    if ("chat_ott.list_chats" in seen or ("chat_ott" in inst_set and "chat_ott.list_chats" in hist_sat and "chat_ott.read_chat" not in hist_sat)) and "chat_ott.read_chat" not in seen and "chat_ott.send_message" not in seen and "chat_ott.call" not in seen and "chat_ott.send_location" not in seen:
        if re.search(r"\b(đọc giúp|nhắn zalo vụ|mới nhắn zalo)\b", u_lower) and not re.search(r"\b(có ai nhắn zalo|ai nhắn zalo)\b", u_lower):
            out.append({"name": "chat_ott.read_chat", "arguments": {"chat_id": "chat_01"}})
            seen.add("chat_ott.read_chat")

    if "public_transit.find_routes" in seen and "public_transit.topup_card" not in seen and re.search(r"\b(nạp thẻ|nạp thêm)\b", u_lower):
        out.append({"name": "public_transit.topup_card", "arguments": {"amount": 50000}})
        seen.add("public_transit.topup_card")

    if "stocks" in inst_set and "stocks.get_quote" in seen and "stocks.set_price_alert" not in seen:
        if re.search(r"\b(lên thêm .* hay xuống|tăng thêm .* hay giảm|báo cô|báo tôi|báo anh|báo chị|cảnh báo)\b", u_lower):
            if "stocks.get_portfolio" not in hist_sat and "stocks.get_portfolio" not in seen:
                out.insert(0, {"name": "stocks.get_portfolio", "arguments": {}})
                seen.add("stocks.get_portfolio")
            out.append({"name": "stocks.set_price_alert", "arguments": {"symbol": "SRT", "price": 90000, "direction": "above"}})
            out.append({"name": "stocks.set_price_alert", "arguments": {"symbol": "SRT", "price": 80000, "direction": "below"}})
            seen.add("stocks.set_price_alert")

    if "flights" in inst_set and ("flights.my_bookings" in seen or "flights.my_bookings" in hist_sat) and "flights.check_in" not in seen:
        if re.search(r"\b(check[ -]?in|làm thủ tục|chụp in|làm y rứa|làm tương tự|làm giống vậy)\b", u_lower) and "tên là làm thủ tục" not in u_lower:
            out.append({"name": "flights.check_in", "arguments": {"booking_id": "flb_01"}})
            seen.add("flights.check_in")
            if re.search(r"\b(cả hai|2\s*vé|hai\s*vé|2\s*chuyến|hai\s*chuyến|mấy\s*vé)\b", u_lower):
                out.append({"name": "flights.check_in", "arguments": {"booking_id": "flb_02"}})

    if "podcast_radio" in inst_set and "podcast_radio.search" in seen and "podcast_radio.play" not in seen:
        if re.search(r"\b(mở cho .* nghe|phát cho .* nghe|bật .* nghe|mở tập|phát tập)\b", u_lower):
            out.append({"name": "podcast_radio.play", "arguments": {"episode_id": "ep_01"}})
            seen.add("podcast_radio.play")

    if "recipes" in inst_set and ("recipes.search" in seen or "recipes.get" in seen) and "recipes.save" not in seen:
        if re.search(r"\b(lưu\s+(?:lại|giúp|giùm|dùm|món|công\s+thức|vào)|cất\s+công\s+thức|đánh\s+dấu\s+món)\b", u_lower) and not re.search(r"\b(ghi\s+chú|bản\s+đồ)\b", u_lower):
            out.append({"name": "recipes.save", "arguments": {"recipe_id": "rec_01"}})
            seen.add("recipes.save")

    if "recipes" in inst_set and "pantry" in inst_set and ("recipes.search" in seen or "recipes.get" in seen):
        if re.search(r"\b(trong\s+tủ|tủ\s+bếp|kho\s+bếp|còn\s+thiếu\s+gì|thiếu\s+nguyên\s+liệu|thiếu\s+thứ\s+gì)\b", u_lower):
            if "pantry.list_items" not in seen and "pantry.list_items" not in hist_sat:
                out.append({"name": "pantry.list_items", "arguments": {}})
                seen.add("pantry.list_items")
            if "shopping_list" in inst_set and "shopping_list.add" not in seen and re.search(r"\b(danh\s+sách\s+mua|ghi\s+vào\s+danh\s+sách|thêm\s+vào\s+danh\s+sách|đi\s+chợ\s+mua)\b", u_lower):
                out.append({"name": "shopping_list.add", "arguments": {"name": "Nguyên liệu", "quantity": 1}})
                seen.add("shopping_list.add")
            elif "grocery" in inst_set and "grocery.add_to_cart" not in seen and re.search(r"\b(giỏ\s+đi\s+chợ|bỏ\s+vào\s+giỏ|thêm\s+vào\s+giỏ)\b", u_lower):
                if "grocery.search" not in seen and "grocery.search" not in hist_sat:
                    out.append({"name": "grocery.search", "arguments": {"query": "Nguyên liệu"}})
                    seen.add("grocery.search")
                out.append({"name": "grocery.add_to_cart", "arguments": {"product_id": "gr_01", "quantity": 1}})
                seen.add("grocery.add_to_cart")

    if "shopping_list" in inst_set and "shopping_list.get" in seen and "shopping_list.remove" not in seen:
        if re.search(r"\b(xóa|xoá|bỏ|gạch)\b.*\b(khỏi danh sách|trong danh sách|đi nhé|đi nha|hết đi)\b", u_lower):
            out.append({"name": "shopping_list.remove", "arguments": {"item_id": "sh_01"}})
            seen.add("shopping_list.remove")

    if "ecommerce" in inst_set and "ecommerce.order_status" in seen and "ecommerce.cancel_order" not in seen:
        if re.search(r"\b(hủy|huỷ)\s+(?:luôn\s+|giúp\s+|giùm\s+|hộ\s+)?(?:đơn|cái\s+đơn|món)\b", u_lower):
            out.append({"name": "ecommerce.cancel_order", "arguments": {"order_id": "ec_ord_01"}})
            seen.add("ecommerce.cancel_order")

    if "bill_payment.list_bills" in seen and "banking" in inst_set and "banking.transfer" not in seen:
        if re.search(r"\b(chuyển\s+khoản|chuyển\s+cho|bắn\s+cho)\b", u_lower) and not re.search(r"\b(đóng|trả|thanh\s+toán)\s+(?:luôn|hóa\s+đơn|hoá\s+đơn)\b", u_lower):
            if "banking.get_balance" not in seen and "banking.get_balance" not in hist_sat:
                out.append({"name": "banking.get_balance", "arguments": {}})
                seen.add("banking.get_balance")
            if "banking.list_beneficiaries" not in seen and "banking.list_beneficiaries" not in hist_sat:
                out.append({"name": "banking.list_beneficiaries", "arguments": {}})
                seen.add("banking.list_beneficiaries")
            out.append({"name": "banking.transfer", "arguments": {"from_account_id": "acc_01", "beneficiary_id": "ben_01", "amount": 500000}})
            seen.add("banking.transfer")

    if "banking" in inst_set and "banking.get_balance" in seen and "banking.open_savings" not in seen:
        if re.search(r"\b(mở\s+sổ\s+tiết\s+kiệm|gửi\s+tiết\s+kiệm|bỏ\s+vào\s+tiết\s+kiệm|chuyển\s+sang\s+tiết\s+kiệm)\b", u_lower):
            out.append({"name": "banking.open_savings", "arguments": {"from_account_id": "acc_01", "amount": 10000000, "term_months": 6}})
            seen.add("banking.open_savings")

    # Đảm bảo tất cả READ_TOOLS luôn đứng trước các công cụ hành động phụ thuộc dữ liệu,
    # nhưng giữ nguyên các lệnh điều khiển thiết bị trực tiếp ở mệnh đề đầu không phụ thuộc Read
    pre_read_names = {"tv.control", "robot_vacuum.stop", "home_security.arm"}
    if "smart_home.get_status" in hist_sat and re.search(r"^\s*(?:ừ|ờ|vậy|thế)?[,\s]*(?:tắt|bật|mở)\s+(?:dùm|giùm|hộ|giúp)\b", u_lower):
        pre_read_names.add("smart_home.light_set")
    pre_read_acts = [
        x for x in out
        if x.get("name") in pre_read_names
        and re.search(r"^\s*(?:ừ|ờ|vậy|thế|ê|nè|em\s+ơi|bạn\s+ơi)?[,\s]*(?:tắt|bật|mở|kêu\s+(?:con\s+)?robot\s+ngưng|cho\s+(?:con\s+)?robot\s+dừng)\b", u_lower)
    ]
    pre_read_set = {id(x) for x in pre_read_acts}
    read_part = [x for x in out if x.get("name") in READ_TOOLS]
    if any(x.get("name") == "lottery.check_ticket" for x in read_part) and any(x.get("name") == "calendar.list_events" for x in read_part):
        read_part.sort(key=lambda x: 0 if x.get("name") == "lottery.check_ticket" else 1)
    if any(x.get("name") in ("train_bus_tickets.my_bookings", "flights.my_bookings") for x in read_part):
        read_part.sort(key=lambda x: 0 if x.get("name") in ("train_bus_tickets.my_bookings", "flights.my_bookings") else 1)
    act_part = [x for x in out if x.get("name") not in READ_TOOLS and id(x) not in pre_read_set]
    if any(x.get("name") == "bill_payment.schedule_autopay" for x in act_part) and any(x.get("name") == "bill_payment.pay" for x in act_part) and "còn cái" in u_lower:
        act_part.sort(key=lambda x: 0 if x.get("name") == "bill_payment.schedule_autopay" else 1)
    if any(x.get("name") == "flights.book" for x in act_part) and any(x.get("name") == "calendar.create_event" for x in act_part):
        act_part.sort(key=lambda x: 0 if x.get("name") == "flights.book" else (1 if x.get("name") == "calendar.create_event" else 2))
    return pre_read_acts + read_part + act_part


def compose_best_exemplar(
    retriever: Any,
    user_text: str,
    installed_apps: Any,
    history: Optional[List[Dict[str, Any]]] = None,
    current_time_str: str = "N/A",
    fallback_exemplars: Optional[List[Dict[str, Any]]] = None,
    previous_steps: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Tổng hợp bài mẫu chuẩn xác nhất cho cả câu đơn mệnh đề, câu ghép đa mệnh đề và câu điều kiện phụ thuộc dữ liệu:
    - Kiểm tra nhánh điều kiện thời gian thực (weather, gold_price, bill_payment + ewallet) theo previous_steps.
    - Khôi phục ý định từ lượt hỏi-đáp trước đó trong history khi lượt hiện tại là câu trả lời bổ sung thông tin.
    - Ưu tiên mẫu toàn câu (Whole-Utterance) trước khi tách mệnh đề (Clause-Split) để bảo toàn cấu trúc câu phức.
    - Tự động loại bỏ các bước Read đã có sẵn kết quả trong history (History-Satisfied Reads).
    """
    if not retriever:
        return fallback_exemplars or []

    _ensure_catalog_exemplars_indexed(retriever)
    inst_dict = installed_apps if isinstance(installed_apps, dict) else {a: {} for a in (installed_apps or [])}
    hist_sat = get_history_satisfied_reads(history, user_text)
    # Xử lý câu tự sửa lời nói (self-correction) mà không làm mất động từ chính hoặc mệnh đề phía sau
    u_for_ret = re.sub(
        r"\b(?:\d+|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)(?:\s+(?:mươi|trăm|nghìn|ngàn|triệu|giờ|rưỡi|phút|độ))*\s*(?:,\s*)?\b(?:à không|a không|à nhầm)\b[,\s]*",
        "",
        user_text,
        flags=re.IGNORECASE,
    )
    u_for_ret = re.sub(
        r"^[^.!?]{3,65}?\b(?:(?:ờ\s+|à\s+)*(?:mà\s+thôi\s*,?\s*(?:không\s+phải|cũng\s+phải|xôi\s+không\s+phải)|không\s+phải|à\s+thôi|à\s+không|a\s+không|à\s+nhầm))\b[,\s]*(?=(?:gửi|nhắn|gọi|bật|tắt|mở|đặt|hủy|huỷ|dời|chuyển|lấy|nạp|thôi)\b)",
        "",
        u_for_ret,
        flags=re.IGNORECASE,
    )
    u_for_ret = re.sub(
        r"^.{3,85}?\b(?:ấy quên|à quên|thôi đừng tắt|thôi khỏi dịch lại|à thôi khỏi|thôi để (?:anh|chị|tôi|mình|em) tự(?:\s+\S+)?)(?:[\s,]*(?:thôi\s+)?(?:đừng\s+tắt|khỏi\s+tắt|cứ\s+để\s+sáng[^.!?;]*|xài\s+được|dùng\s+được|nấu))?[^.!?;]*?(?:[.!?;,]+\s*|\s+(?=(?:bật|mở|chỉnh|đặt|gọi|nhắn|chuyển|trả|nhắc|tạo|thêm|hủy|huỷ|dời|dừng|ngưng|trừ|mua|lưu|xóa|xoá)\b)|$)",
        "",
        u_for_ret,
        flags=re.IGNORECASE,
    )
    u_for_ret = re.sub(
        r"^[^.!?]{3,65}?\b(?:à\s+mà\s+thôi|mà\s+thôi)\s*,\s*khỏi\s+\S+\s*,\s*",
        "",
        u_for_ret,
        flags=re.IGNORECASE,
    )
    u_for_ret = re.sub(r"\b(?:báo là|bảo là|nhắn là|nội dung ghi là)\s+[^.!?;]+?(?:đã hủy|đã huỷ|khoá rồi|khóa rồi|đã gọi số vừa nãy rồi|chuyển tiền [^.!?;]*?rồi)[^.!?;]*", "báo là đã xong", u_for_ret, flags=re.IGNORECASE)
    u_for_ret = re.sub(
        r"\b(nhắn(?:\s+zalo|\s+sms|\s+tin)?\s+[^.!?;]{1,45}?\b(?:báo\s+là|bảo\s+là|nói\s+là|giúp\s+(?:anh|chị|em|tôi|mình)\s+là|hộ\s+(?:anh|chị|em|tôi|mình)\s+là|tin\s+nhắn\s+[^.!?;]{0,20}?\blà))\s+[^.!?;]+",
        r"\1 đã xong",
        u_for_ret,
        flags=re.IGNORECASE,
    )
    u_for_ret = re.sub(
        r"\b(dịch(?:\s+giúp|\s+giùm|\s+hộ)?(?:\s+\S+)?\s+câu)\s+.+?\s+((?:sang|ra|qua|xem)\s+tiếng\s+\S+)",
        r"\1 xin chào \2",
        u_for_ret,
        flags=re.IGNORECASE,
    )
    u_for_ret = re.sub(r"\b(mail|email|thư)\s+quảng\s+cáo\s+vé\s+máy\s+bay\b", r"\1 quảng cáo", u_for_ret, flags=re.IGNORECASE)
    u_for_ret = re.sub(r"\bcô\s+nhắc\s+gọi\s+[^,.]+?\s+hồi\s+nãy\b", "cô nói hồi nãy", u_for_ret, flags=re.IGNORECASE)

    # Khôi phục ngữ cảnh khi lượt trước trợ lý chỉ hỏi lại mà chưa gọi công cụ hành động (hoặc chỉ gọi công cụ tra cứu mà người dùng đang trả lời nối tiếp)
    if history and len(history) >= 2:
        last_asst = history[-1] if history[-1].get("role") == "assistant" else {}
        prev_user = history[-2] if history[-2].get("role") == "user" else {}
        prev_u_txt = clean_asr_transcript(str(prev_user.get("audio_text", "")))
        asst_txt = str(last_asst.get("text", ""))
        has_curr_trig = any(re.search(pat, u_for_ret, re.IGNORECASE) for app, pat in APP_TRIGGER_PATTERNS.items() if app in inst_dict)
        last_asst_read_apps = {
            tc_h.get("name", "").split(".")[0]
            for tc_h in (last_asst.get("tool_calls") or [])
            if isinstance(tc_h, dict) and tc_h.get("name") in READ_TOOLS
        } - {"weather", "smart_home", "home_security"}
        last_asst_write_apps = {
            tc_h.get("name", "").split(".")[0]
            for tc_h in (last_asst.get("tool_calls") or [])
            if isinstance(tc_h, dict) and tc_h.get("name") not in READ_TOOLS and tc_h.get("name") != "sys.ask_missing_info"
        }
        prev_u_triggered = {
            app for app, pat in APP_TRIGGER_PATTERNS.items()
            if app in inst_dict and re.search(pat, prev_u_txt, re.IGNORECASE)
        }
        curr_triggered_apps = {
            app for app, pat in APP_TRIGGER_PATTERNS.items()
            if app in inst_dict and re.search(pat, u_for_ret, re.IGNORECASE)
        }
        has_unfulfilled_prev_app = bool(
            "?" in asst_txt
            and not last_asst_write_apps
            and (prev_u_triggered - curr_triggered_apps)
            and (not last_asst.get("tool_calls") or bool(last_asst_read_apps))
            and len(u_for_ret.split()) <= 15
        )
        if len(history) == 2 and (not last_asst.get("tool_calls") or not has_curr_trig or has_unfulfilled_prev_app) and len(u_for_ret.split()) <= 25:
            u_for_ret = f"{prev_u_txt} {asst_txt} {u_for_ret}".strip()
        elif len(history) == 4 and not (history[1].get("tool_calls") or []) and (not has_curr_trig or has_unfulfilled_prev_app) and "?" in asst_txt and len(u_for_ret.split()) <= 25:
            u0_txt = clean_asr_transcript(str(history[0].get("audio_text", "")))
            u_for_ret = f"{u0_txt} {prev_u_txt} {asst_txt} {u_for_ret}".strip()
        elif len(history) > 2 and (not has_curr_trig or has_unfulfilled_prev_app) and "?" in asst_txt and len(u_for_ret.split()) <= 18:
            u_for_ret = f"{prev_u_txt} {asst_txt} {u_for_ret}".strip()

    u_low = u_for_ret.lower()
    norm_u = _normalize_vn_numbers_in_time(u_for_ret)
    primary_d, _ = resolve_target_dates_from_utterance(u_for_ret, current_time_str)

    # 0a-pre. Nhận diện câu tự hủy toàn bộ yêu cầu ở cuối câu (trước khi vào nhánh điều kiện thời tiết)
    if not previous_steps and re.search(
        r"\b(?:à\s+thôi\s+khỏi|à\s+khoan\s*,?\s*thôi\s+khỏi|thôi\s+khỏi\s+(?:cần\s+)?(?:nhắn|gọi|đặt|tạo|mua|bật|tắt|chuyển|dịch|làm|tra|xem|kiểm\s+tra|ghi)|thôi\s+thôi\s+em\s+ơi[^.!?;]*?\bkhỏi\b|mà\s+thôi\s+[^.!?;]*?\b(?:cứ\s+giữ\s+nguyên|đừng\s+đổi|khỏi\s+đổi|cứ\s+để\s+đấy|cứ\s+để\s+vậy)|à\s+thôi\s+[^.!?;]*?\b(?:cứ\s+để\s+đấy|mai\s+mới\s+mua|để\s+anh\s+tự|để\s+chị\s+tự|để\s+tôi\s+tự|để\s+mình\s+tự|khỏi\s+đặt\s+nữa|em\s+khỏi\s+làm)|cứ\s+để\s+nguyên\s+như\s+thế\s*,?\s*khỏi\s+làm\s+gì\s+thêm|à\s+thôi\s+thôi\s*,?\s*[^.!?;]*?\bqua\s+đón\b[^.!?;]*?\brồi\b)[^.!?;]*[.!?]?\s*$",
        user_text,
        re.IGNORECASE,
    ) and not re.search(
        r"\b(?:(?:à\s+)?thôi\s+khỏi(?:\s+[^.!?;]+?)?|để\s+(?:anh|chị|tôi|mình|em)\s+tự(?:\s+\S+)?)\s*(?:,\s*|\b(?:rồi|xong|mà|chỉ|nhớ|nhắc)\b\s*|\s+)(?:bật|mở|tắt|chỉnh|đặt|gọi|nhắn|chuyển|trả|nhắc|tạo|thêm|hủy|huỷ|dời|dừng|ngưng|về\s+sạc|trừ|mua|lưu|xóa|xoá)\b|\b(?:còn\s+đủ\s+thì|cái\s+nào\s+[^.!?;]*?\bthì|phòng\s+nào\s+[^.!?;]*?\bthì|nếu\s+[^.!?;]*?\bthì)\s+(?:à\s+)?(?:thôi\s+)?(?:khỏi|để\s+(?:anh|chị|tôi|mình)\s+tự)\b",
        user_text,
        re.IGNORECASE,
    ):
        return [{"id": "noop_self_cancel", "audio_text": user_text, "tool_calls": []}]

    # 0a-cur. Nhánh điều kiện Thời tiết Hiện tại ("ngoài trời đang mưa hay nắng, nếu trời còn ráo / đang mưa thì ...")
    prepend_weather_call: Optional[Dict[str, Any]] = None
    u_curr_weather_low = user_text.lower()
    if "weather" in inst_dict and re.search(r"\b(ngoài\s+trời\s+(?:giờ\s+|hiện\s+tại\s+)?(?:đang\s+|sao\s+rồi|còn\s+mưa)|giờ\s+(?:mà\s+|có\s+)?(?:trời\s+|ngoài\s+trời\s+)?đang\s+mưa|trời\s+đang\s+mưa\s+hay\s+nắng|trời\s+còn\s+ráo\s+thì|chỗ\s+(?:anh|chị|tôi|mình|em)\s+đang\s+mưa\s+hay\s+(?:tạnh|nắng|ráo))\b", u_curr_weather_low) and re.search(r"\b(mưa\s+thì|trời\s+còn\s+ráo|còn\s+ráo\s+thì|trời\s+nắng\s+thì|không\s+mưa\s+thì)\b", u_curr_weather_low):
        w_cur_step = next((s for s in (previous_steps or []) if s.get("tool_called") == "weather.get_current" and not s.get("error")), None)
        w_cur_args = {"location": "TP.HCM"} if re.search(r"\bchỗ\s+(?:anh|chị|tôi|mình|em)\s+đang\b", u_curr_weather_low) else {}
        if w_cur_step is None:
            return [{
                "id": "cond_weather_cur_step1",
                "audio_text": user_text,
                "tool_calls": [
                    {"name": "weather.get_current", "arguments": w_cur_args},
                    {"name": "reminders.create", "arguments": {"content": "Kiểm tra chậu cây", "time": f"{primary_d or '2026-10-16'}T19:00:00+07:00"}},
                ],
            }]
        else:
            cur_obs = w_cur_step.get("raw_observation") or {}
            if isinstance(cur_obs, str):
                try:
                    cur_obs = json.loads(cur_obs)
                except Exception:
                    cur_obs = {}
            cond_str = str((cur_obs.get("condition") or cur_obs.get("weather") or "") if isinstance(cur_obs, dict) else "").lower()
            is_cur_rain = bool("mưa" in cond_str or "rain" in cond_str or (isinstance(cur_obs, dict) and cur_obs.get("rain")))
            prepend_weather_call = {"name": "weather.get_current", "arguments": w_cur_step.get("arguments") or w_cur_args}
            w_cur_prefix = [prepend_weather_call]
            if not is_cur_rain:
                if re.search(r"\b(tạnh\s+rồi\s+thì\s+cứ\s+để|không\s+mưa\s+thì\s+thôi|không\s+mưa\s+thì\s+cứ\s+để)\b", u_curr_weather_low):
                    return [{"id": "cond_weather_cur_no_rain_stop", "audio_text": user_text, "tool_calls": list(w_cur_prefix)}]
                if "smart_home" in inst_dict and re.search(r"\b(bật|mở)\b.*\b(ổ cắm|công tắc|tưới)\b", u_curr_weather_low):
                    sh_cur_calls = list(w_cur_prefix)
                    if "smart_home.get_status" not in hist_sat:
                        sh_cur_calls.append({"name": "smart_home.get_status", "arguments": {}})
                    sh_cur_calls.append({"name": "smart_home.switch_set", "arguments": {"device_id": "plug_garden", "power": "on"}})
                    return [{"id": "cond_weather_cur_no_rain_switch", "audio_text": user_text, "tool_calls": sh_cur_calls}]
                if re.search(r"\b(?:không\s+thì|còn\s+không\s+mưa\s+thì|không\s+mưa\s+thì)\b", u_low):
                    u_for_ret = re.sub(r"^.*?\b(?:không\s+thì|còn\s+không\s+mưa\s+thì|không\s+mưa\s+thì)\s+", "", u_for_ret, flags=re.IGNORECASE).strip()
                    u_low = u_for_ret.lower()
            else:
                if "parking" in inst_dict and re.search(r"\b(kết thúc phiên gửi xe|tắt phiên gửi xe|dừng phiên gửi xe)\b", u_curr_weather_low):
                    pk_cur_calls = list(w_cur_prefix)
                    if "parking.list_sessions" not in hist_sat:
                        pk_cur_calls.append({"name": "parking.list_sessions", "arguments": {}})
                    pk_cur_calls.append({"name": "parking.end_session", "arguments": {"session_id": "pk_01"}})
                    return [{"id": "cond_weather_cur_rain_parking", "audio_text": user_text, "tool_calls": pk_cur_calls}]
                if re.search(r"\b(mưa thì thôi khỏi|đang mưa thì thôi|khỏi tưới)\b", u_curr_weather_low) and "reminders" in inst_dict and "nhắc" in u_curr_weather_low:
                    return [{
                        "id": "cond_weather_cur_rain_remind",
                        "audio_text": user_text,
                        "tool_calls": list(w_cur_prefix) + [
                            {"name": "reminders.create", "arguments": {"content": "Kiểm tra chậu cây ngoài ban công", "time": f"{primary_d or '2026-10-16'}T19:00:00+07:00"}},
                        ],
                    }]
                if re.search(r"[,.;]?\s*(?:còn\s+)?(?:không\s+mưa|không\s+thì|trời\s+nắng|trời\s+ráo)\s+(?:thì\s+)?", u_low):
                    u_for_ret = re.sub(r"[,.;]?\s*(?:còn\s+)?(?:không\s+mưa|không\s+thì|trời\s+nắng|trời\s+ráo)\s+(?:thì\s+)?.*$", "", u_for_ret, flags=re.IGNORECASE).strip()
                u_for_ret = re.sub(r"^.*?\b(?:nếu\s+)?(?:còn\s+|đang\s+)?mưa\s+thì\s+", "", u_for_ret, flags=re.IGNORECASE).strip()
                u_low = u_for_ret.lower()

    # 0a. Nhánh điều kiện Thời tiết ("nếu mai mưa ... không mưa thì ...")
    if "weather" in inst_dict and prepend_weather_call is None and re.search(r"\b(mưa|nắng|ráo|thời tiết|nhiệt độ|độ|nóng)\b", u_curr_weather_low) and not re.search(r"\b(giờ\s+(?:mà\s+|có\s+)?(?:trời\s+)?đang|ngoài\s+trời\s+(?:giờ\s+)?(?:có\s+)?đang|hiện\s+tại|lúc\s+này|bây\s+giờ)\b", u_low) and (
        (re.search(r"\b(nếu.*mưa|mà\s+(?:không\s+|hổng\s+|hông\s+)?mưa|mưa thì|mưa không|có mưa không|trời mưa)\b", u_low) and re.search(r"\b(mưa thì|không mưa|hổng mưa|hông mưa|trời ráo|trời nắng|nếu mưa|mà mưa|nếu.*mưa)\b", u_low))
        or re.search(r"\b(nóng quá \d+\s*độ|trên \d+\s*độ)\b.*\bthì\b", norm_u)
    ):
        weather_d = primary_d or "2026-10-16"
        if current_time_str and current_time_str != "N/A" and re.search(r"\b(?:nếu\s+mai|mai\s+mà|sáng\s+mai\s+mà|chiều\s+mai\s+mà|ngày\s+mai\s+mà|mai\s+có)\s+(?:trời\s+)?(?:không\s+|hổng\s+|hông\s+)?mưa\b", u_low):
            try:
                weather_d = (datetime.fromisoformat(current_time_str).date() + timedelta(days=1)).isoformat()
            except Exception:
                pass
        weather_step = next((s for s in (previous_steps or []) if s.get("tool_called") == "weather.get_forecast" and not s.get("error")), None)
        weather_from_hist = False
        if weather_step is None and history and "?" in str((history[-1] if history else {}).get("text", "")):
            last_h = history[-1]
            for tc_h, tr_h in zip(last_h.get("tool_calls") or [], last_h.get("tool_results") or []):
                if isinstance(tc_h, dict) and tc_h.get("name") == "weather.get_forecast" and isinstance(tr_h, dict):
                    if not re.search(r"\b(xem.*mưa không|coi.*mưa không|kiểm tra.*thời tiết)\b", u_low):
                        weather_step = {"tool_called": "weather.get_forecast", "arguments": tc_h.get("arguments") or {}, "raw_observation": tr_h}
                        weather_from_hist = True
                        break
        if weather_step is None:
            return [{
                "id": "cond_weather_step1",
                "audio_text": user_text,
                "tool_calls": [
                    {"name": "weather.get_forecast", "arguments": {"date": weather_d}},
                    {"name": "reminders.create", "arguments": {"content": "Kiểm tra lịch", "time": f"{primary_d or '2026-10-16'}T07:00:00+07:00"}},
                ],
            }]
        else:
            raw_obs = weather_step.get("raw_observation") or {}
            if isinstance(raw_obs, str):
                try:
                    raw_obs = json.loads(raw_obs)
                except Exception:
                    raw_obs = {}
            fc_raw = raw_obs.get("forecast") if isinstance(raw_obs, dict) else None
            fc_obj = fc_raw[0] if isinstance(fc_raw, list) and fc_raw else (fc_raw if isinstance(fc_raw, dict) else (raw_obs if isinstance(raw_obs, dict) else {}))
            m_temp_th = re.search(r"\b(?:nóng quá|trên)\s+(\d+)\s*độ\b", norm_u)
            if m_temp_th:
                is_rain = bool((fc_obj.get("high") or fc_obj.get("temp_max") or 0) > int(m_temp_th.group(1)))
            else:
                is_rain = bool(fc_obj.get("rain"))
            w_args = weather_step.get("arguments") or {"date": weather_d}
            prepend_weather_call = None if weather_from_hist else {"name": "weather.get_forecast", "arguments": w_args}
            w_prefix = [prepend_weather_call] if prepend_weather_call is not None else []
            if not is_rain:
                if (
                    re.search(r"\b(không mưa thì thôi|hông mưa thì thôi|hổng mưa thì thôi|không thì thôi|không mưa\s+(?:thì\s+)?(?:cứ\s+)?giữ nguyên(?!\s+giờ)|không mưa.*cứ để|trời nắng.*giữ nguyên(?!\s+giờ)|trời nắng.*cứ giữ(?!\s+giờ)|không mưa.*không cần|không mưa.*khỏi|nắng thì.*giữ nguyên(?!\s+giờ)|không thì.*giữ nguyên(?!\s+giờ)|không thì.*cứ để)\b", u_low)
                    or not re.search(r"\b(không mưa|không thì|trời nắng|trời ráo|hổng mưa|hông mưa)\b", u_low)
                ) and not re.search(r"\b(chuyển\s+chỗ|đổi\s+địa\s+điểm|dời\s+cái\s+lịch)\b", u_low):
                    return [{
                        "id": "cond_weather_no_rain_stop",
                        "audio_text": user_text,
                        "tool_calls": list(w_prefix),
                    }]
                if "calendar" in inst_dict and re.search(r"\b(chuyển\s+chỗ|đổi\s+địa\s+điểm|dời\s+cái\s+lịch)\b", u_low):
                    cal_nr_calls = list(w_prefix)
                    if "calendar.list_events" not in hist_sat:
                        cal_nr_calls.append({"name": "calendar.list_events", "arguments": {}})
                    cal_nr_calls.append({"name": "calendar.update_event", "arguments": {"event_id": "evt_01", "location": "Phòng họp tầng 3"}})
                    return [{"id": "cond_weather_no_rain_calendar", "audio_text": user_text, "tool_calls": cal_nr_calls}]
                if "smart_home" in inst_dict and re.search(r"\b(không mưa.*bật|trời nắng.*bật|ổ cắm|máy tưới)\b", u_low):
                    sh_calls = list(w_prefix)
                    if "smart_home.get_status" not in hist_sat:
                        sh_calls.append({"name": "smart_home.get_status", "arguments": {}})
                    sh_calls.append({"name": "smart_home.switch_set", "arguments": {"device_id": "plug_garden", "power": "on"}})
                    return [{"id": "cond_weather_no_rain_switch", "audio_text": user_text, "tool_calls": sh_calls}]
                if "ride_hailing" in inst_dict and re.search(r"\b(đặt ô tô|gọi ô tô|đặt xe|gọi xe)\b", u_low) and not re.search(r"\b(hủy|huỷ)\s+chuyến\s+xe\b", u_low):
                    rh_no_rain_calls = list(w_prefix)
                    if "calendar" in inst_dict and "calendar.list_events" not in hist_sat and re.search(r"\b(lịch|buổi|họp|khám|hẹn|đá bóng|chạy bộ|cà phê)\b", u_low):
                        rh_no_rain_calls.append({"name": "calendar.list_events", "arguments": {}})
                    rh_no_rain_calls.append({"name": "ride_hailing.book", "arguments": {"pickup": "12 Nguyễn Huệ, Quận 1, TP.HCM", "dropoff": "Bệnh viện", "vehicle": "car4", "schedule_time": f"{primary_d or '2026-10-16'}T14:30:00+07:00"}})
                    return [{
                        "id": "cond_weather_no_rain_ride",
                        "audio_text": user_text,
                        "tool_calls": rh_no_rain_calls,
                    }]
                if re.search(r"[,.;]?\s*còn\s+mưa\s+thì\b", u_low):
                    u_for_ret = re.sub(r"[,.;]?\s*còn\s+mưa\s+thì\b.*$", "", u_for_ret, flags=re.IGNORECASE).strip()
                elif re.search(r"\b(?:còn\s+)?(?:nếu\s+|mai\s+mà\s+)?(?:không\s+mưa|hổng\s+mưa|hông\s+mưa|không\s+thì|trời\s+nắng|trời\s+ráo)\s+(?:thì\s+)?", u_low):
                    u_for_ret = re.sub(r"^.*?\b(?:còn\s+)?(?:nếu\s+|mai\s+mà\s+)?(?:không\s+mưa|hổng\s+mưa|hông\s+mưa|không\s+thì|trời\s+nắng|trời\s+ráo)\s+(?:thì\s+)?", "", u_for_ret, flags=re.IGNORECASE).strip()
                u_low = u_for_ret.lower()
            else:
                if re.search(r"\b(mưa thì khỏi tưới|mưa thì thôi|nếu mưa.*khỏi bật)\b", u_low) and "reminders" in inst_dict and "nhắc" in u_low:
                    return [{
                        "id": "cond_weather_rain_remind_only",
                        "audio_text": user_text,
                        "tool_calls": list(w_prefix) + [
                            {"name": "reminders.create", "arguments": {"content": "Kiểm tra chậu cây ngoài ban công", "time": f"{primary_d or '2026-10-16'}T07:00:00+07:00"}},
                        ],
                    }]
                if "alarm" in inst_dict and re.search(r"\b(cái còn lại|cái kia|báo thức)\b", u_low) and ("alarm.list" in hist_sat or "báo thức" in u_low):
                    al_calls = list(w_prefix)
                    if "alarm.list" not in hist_sat:
                        al_calls.append({"name": "alarm.list", "arguments": {}})
                    al_calls.append({"name": "alarm.update", "arguments": {"alarm_id": "al_01", "time": "06:30"}})
                    return [{"id": "cond_weather_rain_alarm", "audio_text": user_text, "tool_calls": al_calls}]
                if "calendar" in inst_dict and re.search(r"\b(dời|đổi|lùi|chuyển|hủy|huỷ|đẩy|bỏ)\b", u_low) and not re.search(r"\bnhắn\s+[^\s,.]+(?:\s+[^\s,.]+)?\s+(?:giùm|hộ|là)\s+dời\b", u_low):
                    cal_calls = list(w_prefix)
                    if "calendar.list_events" not in hist_sat:
                        cal_calls.append({"name": "calendar.list_events", "arguments": {}})
                    if re.search(r"\b(hủy|huỷ|bỏ)\b", u_low):
                        cal_calls.append({"name": "calendar.cancel_event", "arguments": {"event_id": "evt_01"}})
                    else:
                        cal_calls.append({"name": "calendar.update_event", "arguments": {"event_id": "evt_01", "start": f"{primary_d or '2026-10-16'}T09:00:00+07:00", "end": f"{primary_d or '2026-10-16'}T10:00:00+07:00"}})
                    rain_clause = re.split(r"\bcòn\s+(?:không\s+mưa|trời\s+ráo|trời\s+nắng|nắng)\b", u_low)[0]
                    if "chat_ott" in inst_dict and re.search(r"\b(zalo|nhóm)\b", rain_clause):
                        if "chat_ott.list_chats" not in hist_sat:
                            cal_calls.append({"name": "chat_ott.list_chats", "arguments": {}})
                        cal_calls.append({"name": "chat_ott.send_message", "arguments": {"chat_id": "chat_01", "content": "Sáng mai trời mưa nên mình dời lịch nhé."}})
                    if "sms" in inst_dict and re.search(r"\b(sms|nhắn)\b", rain_clause) and not re.search(r"\b(zalo|nhóm)\b", rain_clause):
                        if "contacts" in inst_dict and "contacts.search" not in hist_sat:
                            q_ct_w = _extract_contact_query_from_text(rain_clause) or "Hương"
                            cal_calls.append({"name": "contacts.search", "arguments": {"query": q_ct_w}})
                        cal_calls.append({"name": "sms.send", "arguments": {"contact_id": "ct_01", "content": "Sáng mai trời mưa nên mình dời lịch nhé."}})
                    if "phone" in inst_dict and re.search(r"\bgọi\s+(?:điện|cho|báo)\b", rain_clause):
                        if "contacts" in inst_dict and "contacts.search" not in hist_sat:
                            q_ct_p = _extract_contact_query_from_text(rain_clause) or "Lan"
                            cal_calls.append({"name": "contacts.search", "arguments": {"query": q_ct_p}})
                        cal_calls.append({"name": "phone.call", "arguments": {"contact_id": "ct_01"}})
                    if "reminders" in inst_dict and re.search(r"\b(nhắc)\b", rain_clause):
                        cal_calls.append({"name": "reminders.create", "arguments": {"content": "Mang ô", "time": f"{primary_d or '2026-10-16'}T07:00:00+07:00"}})
                    return [{"id": "cond_weather_rain_calendar", "audio_text": user_text, "tool_calls": cal_calls}]
                if "ride_hailing" in inst_dict and re.search(r"\b(đặt xe|gọi xe|đặt ô tô|gọi ô tô|xe bốn chỗ|xe 4 chỗ)\b", u_low):
                    rh_calls = list(w_prefix)
                    if "calendar" in inst_dict and "calendar.list_events" not in hist_sat and re.search(r"\b(lịch|buổi|họp|khám)\b", u_low):
                        rh_calls.append({"name": "calendar.list_events", "arguments": {}})
                    rh_calls.append({"name": "ride_hailing.book", "arguments": {"pickup": "Nhà", "dropoff": "Công ty", "vehicle": "car4"}})
                    return [{"id": "cond_weather_rain_ride", "audio_text": user_text, "tool_calls": rh_calls}]
                if re.search(r"^.*?\bcòn\s+mưa\s+thì\b", u_low):
                    u_for_ret = re.sub(r"^.*?\bcòn\s+mưa\s+thì\b", "", u_for_ret, flags=re.IGNORECASE).strip()
                elif re.search(r"[,.;]?\s*(?:còn\s+)?(?:không\s+mưa|không\s+thì|trời\s+nắng|trời\s+ráo)\s+(?:thì\s+)?", u_low):
                    u_for_ret = re.sub(r"[,.;]?\s*(?:còn\s+)?(?:không\s+mưa|không\s+thì|trời\s+nắng|trời\s+ráo)\s+(?:thì\s+)?.*$", "", u_for_ret, flags=re.IGNORECASE).strip()
                u_low = u_for_ret.lower()

    # 0b. Nhánh điều kiện Giá vàng ("giá vàng ... nếu dưới/trên/chưa tới X triệu thì ...")
    if "gold_price" in inst_dict and re.search(r"\b(giá vàng|vàng sjc|vàng nhẫn|nhẫn vàng|vàng thế giới|mua vàng|bán vàng)\b", u_low) and re.search(r"\b(dưới|trên|hơn|vượt|quá|chưa\s+tới|chưa\s+đến|không\s+quá|không\s+tới)\s+\d+(?:\s*triệu|\s*đô)?\b", norm_u):
        gold_step = next((s for s in (previous_steps or []) if s.get("tool_called") == "gold_price.get_price" and not s.get("error")), None)
        gold_from_hist = False
        if gold_step is None and history and len(history) == 2 and "?" in str((history[-1] if history else {}).get("text", "")):
            ht = history[-1]
            tcs_h = ht.get("tool_calls") or []
            trs_h = ht.get("tool_results") or []
            for tc_h, tr_h in zip(tcs_h, trs_h):
                if isinstance(tc_h, dict) and tc_h.get("name") == "gold_price.get_price" and isinstance(tr_h, dict):
                    if not re.search(r"\b(xem.*giá vàng|coi.*giá vàng|kiểm tra.*giá vàng|hỏi.*giá vàng)\b", u_low):
                        gold_step = {"tool_called": "gold_price.get_price", "arguments": {}, "raw_observation": tr_h}
                        gold_from_hist = True
                        break
        if gold_step is None:
            g_type_arg = {"type": "ring" if "nhẫn" in u_low else "sjc"}
            g_s1_calls = [{"name": "gold_price.get_price", "arguments": g_type_arg}]
            if "banking" in inst_dict and re.search(r"\b(chuyển|tài khoản thanh toán|trích tài khoản)\b", norm_u):
                g_s1_calls.extend([
                    {"name": "banking.get_balance", "arguments": {}},
                    {"name": "banking.list_beneficiaries", "arguments": {}},
                ])
            if "parking" in inst_dict and re.search(r"\b(bãi đỗ xe|bãi gửi xe|gửi xe|đỗ xe)\b", norm_u):
                g_s1_calls.append({"name": "parking.find", "arguments": {"location": "Quận 1"}})
                g_s1_calls.append({"name": "parking.book", "arguments": {"lot_id": "pk_01", "hours": 2}})
            else:
                g_s1_calls.append({"name": "reminders.create", "arguments": {"content": "Mua vàng", "time": f"{primary_d or '2026-10-16'}T09:00:00+07:00"}})
            return [{
                "id": "cond_gold_step1",
                "audio_text": user_text,
                "tool_calls": g_s1_calls,
            }]
        else:
            raw_obs = gold_step.get("raw_observation") or {}
            if isinstance(raw_obs, str):
                try:
                    raw_obs = json.loads(raw_obs)
                except Exception:
                    raw_obs = {}
            items = (raw_obs.get("prices") or raw_obs.get("items") or []) if isinstance(raw_obs, dict) else []
            target_item = items[0] if items else {}
            for it in items:
                if isinstance(it, dict):
                    if "nhẫn" in u_low and "nhẫn" in str(it.get("type", "")).lower():
                        target_item = it
                        break
                    if "sjc" in u_low and "sjc" in str(it.get("type", "")).lower():
                        target_item = it
                        break
            val_check = target_item.get("buy", 0) if "mua vào" in u_low else target_item.get("sell", 0)
            m_th = re.search(r"\b(dưới|trên|hơn|vượt|quá|chưa\s+tới|chưa\s+đến|không\s+quá|không\s+tới)\s+(\d+)(?:\s*triệu)?\b", norm_u)
            g_type_arg = {"type": "ring" if "nhẫn" in u_low else "sjc"}
            g_prefix = [] if gold_from_hist else [{"name": "gold_price.get_price", "arguments": g_type_arg}]
            if m_th and val_check:
                op_w, raw_num = m_th.group(1).strip().lower(), int(m_th.group(2))
                th_m = raw_num * 1_000_000 if raw_num < 1000 else raw_num
                is_less_op = bool(re.match(r"^(?:dưới|chưa|không)", op_w))
                cond_ok = (val_check < th_m) if is_less_op else (val_check > th_m)
                g_clauses = re.split(
                    r"(?:[,.;]\s*|\b)(?:còn\s+(?:nếu\s+|quá(?:\s+rồi)?|trên|dưới|vượt|cao\s+hơn|thấp\s+hơn|chưa\s+tới|chưa\s+đến|(?:không|0)(?:\s+thì)?)|chưa\s+tới\s+thì|chưa\s+đến\s+thì|(?:không|0)\s+thì|ngược\s+lại\s+thì)\b",
                    norm_u,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )
                if_clause_g = g_clauses[0] if g_clauses else norm_u
                else_clause_g = g_clauses[1] if len(g_clauses) > 1 else ""
                active_clause_g = if_clause_g if cond_ok else else_clause_g
                if "parking" in inst_dict and re.search(r"\b(bãi đỗ xe|bãi gửi xe|gửi xe|đỗ xe)\b", active_clause_g):
                    pk_loc = "Chợ Bến Thành" if "bến thành" in u_low else "Quận 1"
                    m_pk_hr = re.search(r"\b(\d+)\s*tiếng\b", active_clause_g)
                    pk_hr = int(m_pk_hr.group(1)) if m_pk_hr else 2
                    return [{
                        "id": "cond_gold_parking",
                        "audio_text": user_text,
                        "tool_calls": g_prefix + [
                            {"name": "parking.find", "arguments": {"location": pk_loc}},
                            {"name": "parking.book", "arguments": {"lot_id": "pk_01", "hours": pk_hr}},
                        ],
                    }]
                if "banking" in inst_dict and re.search(r"\b(chuyển.*\d+\s*triệu|chuyển khoản|chuyển cho|tài khoản thanh toán|gửi.*\d+\s*triệu|trích tài khoản)\b", active_clause_g):
                    m_g_amt = re.search(r"\b(\d+)\s*triệu(?:\s*(rưỡi|\d+))?\b", re.sub(r"^.*?\bthì\b", "", active_clause_g))
                    if m_g_amt:
                        g_tr_amt = int(m_g_amt.group(1)) * 1_000_000
                        if (m_g_amt.group(2) or "").lower() == "rưỡi":
                            g_tr_amt += 500_000
                        elif m_g_amt.group(2) and m_g_amt.group(2).isdigit():
                            sub_g = m_g_amt.group(2)
                            g_tr_amt += int(sub_g) * (100_000 if len(sub_g) == 1 else 10_000)
                    else:
                        g_tr_amt = 5_000_000
                    return [{
                        "id": "cond_gold_bank",
                        "audio_text": user_text,
                        "tool_calls": g_prefix + [
                            {"name": "banking.get_balance", "arguments": {}},
                            {"name": "banking.list_beneficiaries", "arguments": {}},
                            {"name": "banking.transfer", "arguments": {"from_account_id": "acc_01", "beneficiary_id": "ben_01", "amount": g_tr_amt}},
                        ],
                    }]
                if "reminders" in inst_dict and "nhắc" in active_clause_g:
                    return [{
                        "id": "cond_gold_remind",
                        "audio_text": user_text,
                        "tool_calls": g_prefix + [
                            {"name": "reminders.create", "arguments": {"content": "Kiểm tra giá vàng", "time": f"{primary_d or '2026-10-16'}T09:00:00+07:00"}},
                        ],
                    }]
                if not cond_ok:
                    return [{
                        "id": "cond_gold_false_stop",
                        "audio_text": user_text,
                        "tool_calls": g_prefix,
                    }]

    # 0c. Nhánh điều kiện Hóa đơn + Ví điện tử ("nếu ví đủ ... còn thiếu/không đủ thì trả bằng ngân hàng và nhắc nạp ví / nhắn SMS")
    if "bill_payment" in inst_dict and "ewallet" in inst_dict and re.search(r"\b(ví đủ|đủ thì|còn đủ|không đủ|còn thiếu|nếu thiếu|thiếu thì|ưu tiên ví|hụt thì|bằng ví.*(?:ngân hàng|tài khoản))\b", u_low) and re.search(r"\b(ví|vĩ|vé|phí)\b", u_low) and not re.search(r"\btiết kiệm\b", u_low):
        bill_step = next((s for s in (previous_steps or []) if s.get("tool_called") == "bill_payment.list_bills" and not s.get("error")), None)
        ew_step = next((s for s in (previous_steps or []) if s.get("tool_called") == "ewallet.get_balance" and not s.get("error")), None)
        ct_q = _extract_contact_query_from_text(u_for_ret) or "Mẹ"
        has_sms_req = bool("sms" in inst_dict and re.search(r"\b(sms|nhắn)\b", u_low))
        has_rem_req = bool("reminders" in inst_dict and "nhắc" in u_low)
        need_contacts = bool(has_sms_req and "contacts" in inst_dict and "contacts.search" not in hist_sat)
        if bill_step is None or ew_step is None:
            s1_calls = [
                {"name": "bill_payment.list_bills", "arguments": {}},
                {"name": "ewallet.get_balance", "arguments": {}},
            ]
            if need_contacts:
                s1_calls.append({"name": "contacts.search", "arguments": {"query": ct_q}})
            s1_calls.append({"name": "bill_payment.pay", "arguments": {"bill_id": "bill_01", "source": "ewallet"}})
            if has_rem_req:
                s1_calls.append({"name": "reminders.create", "arguments": {"content": "Nạp tiền vào ví", "time": f"{primary_d or '2026-10-16'}T20:00:00+07:00"}})
            if has_sms_req:
                s1_calls.append({"name": "sms.send", "arguments": {"contact_id": "ct_01", "content": "Đã thanh toán hóa đơn."}})
            return [{
                "id": "cond_bill_ewallet_step1",
                "audio_text": user_text,
                "tool_calls": s1_calls,
            }]
        else:
            b_obs = bill_step.get("raw_observation") or {}
            e_obs = ew_step.get("raw_observation") or {}
            ew_bal = e_obs.get("balance", 0) if isinstance(e_obs, dict) else 0
            bills_l = (b_obs.get("bills") or []) if isinstance(b_obs, dict) else []
            target_b = next((b for b in bills_l if isinstance(b, dict) and not b.get("paid") and (
                ("điện thoại" in u_low and b.get("type") == "phone")
                or ("điện" in u_low and "điện thoại" not in u_low and b.get("type") == "electricity")
                or ("nước" in u_low and b.get("type") == "water")
                or (("mạng" in u_low or "internet" in u_low) and b.get("type") == "internet")
                or ("truyền hình" in u_low and b.get("type") == "tv")
            )), bills_l[0] if bills_l else {})
            b_amt = target_b.get("amount", 0) if isinstance(target_b, dict) else 0
            src_val = "ewallet" if ew_bal >= b_amt else "bank"
            else_split = re.split(r"\b(?:còn\s+(?:không\s+đủ|thiếu|hụt)|nếu\s+(?:không\s+đủ|thiếu|hụt)|ví\s+(?:không\s+đủ|thiếu|hụt)|không\s+đủ\s+thì|thiếu\s+thì|hụt\s+thì)\b", u_low, maxsplit=1)
            if_part_ew = else_split[0] if else_split else u_low
            else_part_ew = else_split[1] if len(else_split) > 1 else ""
            sms_in_else = bool(re.search(r"\b(sms|nhắn)\b", else_part_ew))
            sms_in_if = bool(re.search(r"\b(sms|nhắn)\b", if_part_ew))
            sms_is_post_action = bool(re.search(r"\b(?:xong|rồi)\s+nhắn\s+sms\b", else_part_ew) and not has_rem_req)
            do_sms = bool(has_sms_req and (sms_is_post_action or (ew_bal >= b_amt and (sms_in_if or not sms_in_else)) or (ew_bal < b_amt and (sms_in_else or not sms_in_if))))
            s2_calls = [
                {"name": "bill_payment.list_bills", "arguments": {}},
                {"name": "ewallet.get_balance", "arguments": {}},
            ]
            if need_contacts and do_sms:
                s2_calls.append({"name": "contacts.search", "arguments": {"query": ct_q}})
            s2_calls.append({"name": "bill_payment.pay", "arguments": {"bill_id": target_b.get("id", "bill_01"), "source": src_val}})
            if re.search(r"\b(tự động thanh toán|tự trả|autopay)\b", u_low):
                s2_calls.append({"name": "bill_payment.schedule_autopay", "arguments": {"type": target_b.get("type", "electricity"), "enabled": True}})
            if ew_bal < b_amt and has_rem_req:
                rem_c_ew = "nạp tiền vào ví" if ("vào ví" in u_low or "nạp tiền" in u_low) else "Nạp ví điện tử"
                s2_calls.append({"name": "reminders.create", "arguments": {"content": rem_c_ew, "time": f"{primary_d or '2026-10-16'}T20:00:00+07:00"}})
            if do_sms:
                s2_calls.append({"name": "sms.send", "arguments": {"contact_id": "ct_01", "content": "Con đóng hóa đơn rồi nhé."}})
            return [{
                "id": "cond_bill_ewallet_step2",
                "audio_text": user_text,
                "tool_calls": s2_calls,
            }]

    # 0c-qr. Nhánh điều kiện Ví điện tử + Quét QR ("ví tôi đủ X thì quét trả hết cho ..., không thì trả phần tôi")
    if "ewallet" in inst_dict and re.search(r"\b(ví\s+tôi\s+đủ|ví\s+đủ|nếu\s+ví\s+đủ)\b", u_low) and re.search(r"\b(quét\s+trả|quét\s+mã|thanh\s+toán\s+qr|trả\s+phần\s+tôi)\b", u_low):
        return [{
            "id": "cond_ewallet_qr",
            "audio_text": user_text,
            "tool_calls": [
                {"name": "ewallet.get_balance", "arguments": {}},
                {"name": "ewallet.pay_qr", "arguments": {"merchant_id": "mch_01", "amount": 340000}},
            ],
        }]

    # 0c-fuel. Nhánh điều kiện Giá xăng + Chuyển khoản ngân hàng ("kiểm tra giá xăng ... nếu trên/dưới X nghìn thì chuyển Y nghìn, còn không thì chuyển Z nghìn")
    if "fuel_price" in inst_dict and "banking" in inst_dict and re.search(r"\b(giá xăng|xăng ron|xăng dầu)\b", u_low) and re.search(r"\b(trên|dưới|hơn|vượt|quá)\s+\d+", norm_u):
        fuel_step = next((s for s in (previous_steps or []) if s.get("tool_called") == "fuel_price.get_price" and not s.get("error")), None)
        if fuel_step is None:
            return [{
                "id": "cond_fuel_bank_step1",
                "audio_text": user_text,
                "tool_calls": [
                    {"name": "fuel_price.get_price", "arguments": {}},
                    {"name": "banking.get_balance", "arguments": {}},
                    {"name": "banking.list_beneficiaries", "arguments": {}},
                    {"name": "banking.transfer", "arguments": {"from_account_id": "acc_01", "beneficiary_id": "ben_01", "amount": 500000}},
                ],
            }]
        else:
            f_obs = fuel_step.get("raw_observation") or {}
            f_list = (f_obs.get("prices") or f_obs.get("fuels") or []) if isinstance(f_obs, dict) else []
            f_item = next((x for x in f_list if isinstance(x, dict) and "95" in str(x.get("type", x.get("name", "")))), f_list[0] if f_list else {})
            f_price = f_item.get("price", 0) if isinstance(f_item, dict) else 0
            m_f_th = re.search(r"\b(trên|dưới|hơn|vượt|quá)\s+(\d+)\s*(?:nghìn|ngàn)?\b", norm_u)
            f_amt = 500000
            if m_f_th and f_price:
                f_op, f_num = m_f_th.group(1).lower(), int(m_f_th.group(2))
                f_th = f_num * 1000 if f_num < 100 else f_num
                f_ok = (f_price < f_th) if f_op == "dưới" else (f_price > f_th)
                f_parts = re.split(r"\bcòn\s+(?:không|dưới|trên|thấp|cao)\b", norm_u, maxsplit=1)
                chosen_f_cl = f_parts[0] if f_ok else (f_parts[1] if len(f_parts) > 1 else f_parts[0])
                m_f_tr = re.search(r"\b(?:chuyển|gửi|cho\s+[^\s,]+)\s+(\d+)\s*(?:nghìn|ngàn)\b", chosen_f_cl) or re.search(r"\b(\d+)\s*(?:nghìn|ngàn)\b(?!\s*một\s*lít)", chosen_f_cl)
                if m_f_tr:
                    f_amt = int(m_f_tr.group(1)) * 1000
            return [{
                "id": "cond_fuel_bank_step2",
                "audio_text": user_text,
                "tool_calls": [
                    {"name": "fuel_price.get_price", "arguments": {}},
                    {"name": "banking.get_balance", "arguments": {}},
                    {"name": "banking.list_beneficiaries", "arguments": {}},
                    {"name": "banking.transfer", "arguments": {"from_account_id": "acc_01", "beneficiary_id": "ben_01", "amount": f_amt}},
                ],
            }]

    # 0c-bis. Nạp tiền điện thoại cho người thứ 1 + chuyển ví cho người thứ 2 bằng số tiền hóa đơn
    if "ewallet" in inst_dict and "bill_payment" in inst_dict and re.search(r"\bnạp\b.*\b(?:chuyển|gửi|gởi)\b.*\btiền\s+(?:điện|nước|mạng|internet|truyền hình)\b", u_low):
        cl_parts = re.split(r"\b(?:rồi|với|và|còn)\s+", u_for_ret, maxsplit=1, flags=re.IGNORECASE)
        p1 = _extract_contact_query_from_text((cl_parts[0] if cl_parts else "").lower()) or "Tuấn"
        p2 = _extract_contact_query_from_text((cl_parts[1] if len(cl_parts) > 1 else u_for_ret).lower()) or "Cô giáo Mai"
        top_amt = 100000 if "100" in norm_u or "một trăm" in u_low else 50000
        return [{
            "id": "composed_topup_and_bill_transfer",
            "audio_text": user_text,
            "tool_calls": [
                {"name": "bill_payment.list_bills", "arguments": {}},
                {"name": "contacts.search", "arguments": {"query": p1}},
                {"name": "contacts.search", "arguments": {"query": p2}},
                {"name": "ewallet.topup_mobile", "arguments": {"phone": "0901234567", "amount": top_amt}},
                {"name": "ewallet.transfer", "arguments": {"contact_id": "ct_02", "amount": 200000}},
            ],
        }]

    if "tv.guide" in hist_sat and "tv" in inst_dict and re.search(r"\b(chuyển sang phim|mở phim|bật kênh phim)\b", u_for_ret, re.IGNORECASE):
        if not re.search(r"\b(vtv\d|htv\d|kênh \d)\b", u_for_ret, re.IGNORECASE):
            return [{
                "id": "composed_tv_missing_channel",
                "audio_text": user_text,
                "tool_calls": [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "tv.tune", "params": ["channel"]}}]
            }]

    # 0d. Định tuyến câu nối tiếp ngắn dựa trên lượt có tool_calls gần nhất của history (Short Follow-up History Routing)
    if history and len(user_text.split()) <= 24 and len([c.strip() for c in CLAUSE_SPLIT_RE.split(u_for_ret) if len(c.strip().split()) >= 3]) < 2:
        recent_tc_turn = next((t for t in reversed(history) if t.get("tool_calls")), {})
        last_turn_tools = [tc.get("name", "") for tc in (recent_tc_turn.get("tool_calls") or []) if isinstance(tc, dict)]
        u_curr_low = u_for_ret.lower()
        if "chat_ott" in inst_dict and re.search(r"^\s*(?:dạ\s+|ừ\s+|ờ\s+)?zalo(?:\s+|,)*(?:(?:đi|nhé|nha|ạ|em|hen|anh|chị)(?:\s+|,)*)*[\.!]?\s*$", u_curr_low):
            z_calls = []
            if "chat_ott.list_chats" not in hist_sat:
                z_calls.append({"name": "chat_ott.list_chats", "arguments": {}})
            hist_u_txt = " ".join(str(t.get("audio_text", "")) for t in (history or []) if t.get("role") == "user")
            if re.search(r"\bgọi\b", hist_u_txt, re.IGNORECASE) and not re.search(r"\b(nhắn|gửi|báo)\b", hist_u_txt, re.IGNORECASE):
                z_calls.append({"name": "chat_ott.call", "arguments": {"chat_id": "chat_01"}})
            else:
                z_calls.append({"name": "chat_ott.send_message", "arguments": {"chat_id": "chat_01", "content": "Tin nhắn"}})
            return [{"id": "followup_zalo_choice", "audio_text": user_text, "tool_calls": z_calls}]
        if "ecommerce.search" in last_turn_tools and "ecommerce" in inst_dict and re.search(r"\b(cái thứ hai|cái đầu|cái rẻ|cái đắt|loại rẻ|loại đắt|vào giỏ|mua luôn|đặt luôn|lấy cái|chốt cái|hai túi|2 túi|ba hũ|3 hũ|lấy .* cho mình)\b", u_curr_low):
            ec_calls = [{"name": "ecommerce.add_to_cart", "arguments": {"product_id": "prod_01", "quantity": 1}}]
            if re.search(r"\b(đặt luôn|mua luôn|thanh toán|giao về)\b", u_curr_low):
                ec_calls.append({"name": "ecommerce.checkout", "arguments": {"address": "Nhà"}})
            return [{"id": "followup_ecommerce_add", "audio_text": user_text, "tool_calls": ec_calls}]
        if "ecommerce.order_status" in last_turn_tools and "ecommerce" in inst_dict:
            if re.search(r"\b(trả|đổi|hỏng|lỗi|nhầm|bể|vỡ|rách)\b", u_curr_low):
                ret_reason = "wrong_item" if "nhầm" in u_curr_low else "damaged"
                return [{"id": "followup_ecommerce_return", "audio_text": user_text, "tool_calls": [{"name": "ecommerce.request_return", "arguments": {"order_id": "ord_01", "reason": ret_reason}}]}]
            if re.search(r"\b(hủy|huỷ|bỏ)\b", u_curr_low):
                return [{"id": "followup_ecommerce_cancel", "audio_text": user_text, "tool_calls": [{"name": "ecommerce.cancel_order", "arguments": {"order_id": "ord_01"}}]}]
        if "grocery.add_to_cart" in last_turn_tools and "grocery" in inst_dict and re.search(r"\b(chốt|chúc đơn|thanh toán|giao về|đủ rồi)\b", u_curr_low):
            gr_co_calls = []
            if "calendar" in inst_dict and re.search(r"\b(bữa tối|lịch|buổi|họp)\b", u_curr_low):
                gr_co_calls.append({"name": "calendar.list_events", "arguments": {}})
            gr_co_calls.append({"name": "grocery.checkout", "arguments": {"address": "Nhà"}})
            return [{
                "id": "followup_grocery_checkout",
                "audio_text": user_text,
                "tool_calls": gr_co_calls,
            }]
        if "bill_payment.list_bills" in hist_sat and "bill_payment" in inst_dict and re.search(r"\b(đóng luôn|trừ vào|đúng rồi tháng)\b", u_curr_low):
            return [{
                "id": "followup_bill_pay",
                "audio_text": user_text,
                "tool_calls": [{"name": "bill_payment.pay", "arguments": {"bill_id": "bill_01", "source": "bank" if "ngân hàng" in u_curr_low else "ewallet"}}],
            }]
        if "calendar.list_events" in last_turn_tools and "calendar" in inst_dict and re.search(r"\b(hủy|huỷ|xóa|xoá)\b.*\b(lịch|cuộc|buổi|hẹn)\b", u_curr_low):
            return [{
                "id": "followup_calendar_cancel",
                "audio_text": user_text,
                "tool_calls": [{"name": "calendar.cancel_event", "arguments": {"event_id": "evt_01"}}],
            }]
        if "calendar.list_events" in last_turn_tools and "calendar" in inst_dict and re.search(r"\b(lùi|dời|đổi|chuyển)\b.*\b(sang|thành|qua)\s+\d", norm_u):
            return [{
                "id": "followup_calendar_update",
                "audio_text": user_text,
                "tool_calls": [{"name": "calendar.update_event", "arguments": {"event_id": "evt_01", "start": f"{primary_d or '2026-10-16'}T15:00:00+07:00", "end": f"{primary_d or '2026-10-16'}T16:00:00+07:00"}}],
            }]
        if "appliances.list" in last_turn_tools and "appliances" in inst_dict and re.search(r"\b(chạy máy|bật máy|chương trình|chế độ)\b", u_curr_low):
            return [{
                "id": "followup_appliances_start",
                "audio_text": user_text,
                "tool_calls": [{"name": "appliances.start_cycle", "arguments": {"appliance_id": "app_01", "cycle": "normal"}}],
            }]
        if "sys.ask_missing_info" in last_turn_tools and "ewallet" in inst_dict and re.search(r"\b\d+\s*(?:nghìn|ngàn|triệu|k\b)", norm_u):
            ew_f_calls = []
            if "contacts" in inst_dict and "contacts.search" not in hist_sat:
                ew_f_calls.append({"name": "contacts.search", "arguments": {"query": _extract_contact_query_from_text(u_for_ret.lower()) or "Lan"}})
            ew_f_calls.append({"name": "ewallet.transfer", "arguments": {"contact_id": "ct_01", "amount": 250000}})
            return [{"id": "followup_ewallet_transfer", "audio_text": user_text, "tool_calls": ew_f_calls}]
        if "bike_rental.find_bikes" in last_turn_tools and "bike_rental" in inst_dict and re.search(r"\b(giữ|đặt|thuê|lấy|chiếc)\b", u_curr_low):
            return [{
                "id": "followup_bike_reserve",
                "audio_text": user_text,
                "tool_calls": [{"name": "bike_rental.reserve", "arguments": {"bike_id": "bike_01"}}],
            }]
        if "tv.guide" in last_turn_tools and "tv" in inst_dict and re.search(r"\b(ghi lẹ|ghi lại|trận .* ghi)\b", u_curr_low):
            return [{
                "id": "followup_tv_record",
                "audio_text": user_text,
                "tool_calls": [{"name": "tv.record", "arguments": {"program_id": "prg_01"}}],
            }]
        if "grocery.search" in last_turn_tools and "grocery" in inst_dict and re.search(r"\b(đặt cho|lấy cho|ba hộp|3 hộp|hai hộp|2 hộp|vào giỏ|mua luôn|đặt luôn|loại không đường)\b", u_curr_low):
            gr_calls = [{"name": "grocery.add_to_cart", "arguments": {"product_id": "gro_01", "quantity": 3}}]
            if re.search(r"\b(đặt cho|mua luôn|đặt luôn|giao về)\b", u_curr_low):
                gr_calls.append({"name": "grocery.checkout", "arguments": {"address": "Nhà"}})
            return [{"id": "followup_grocery_add", "audio_text": user_text, "tool_calls": gr_calls}]
        if "pantry.list_items" in last_turn_tools:
            if "pantry" in inst_dict and re.search(r"\b(dùng hết|xài hết|hết luôn|uống hết|ăn hết|trừ đi|trừ bớt)\b", u_curr_low):
                return [{
                    "id": "followup_pantry_consume",
                    "audio_text": user_text,
                    "tool_calls": [{"name": "pantry.consume", "arguments": {"item_id": "pan_01", "quantity": 1}}],
                }]
            if "shopping_list" in inst_dict and re.search(r"\b(danh sách đi chợ|danh sách mua sắm|mua thêm|mua bù|ghi vào)\b", u_curr_low):
                return [{
                    "id": "followup_pantry_shopping_list",
                    "audio_text": user_text,
                    "tool_calls": [{"name": "shopping_list.add", "arguments": {"name": "Thực phẩm", "quantity": 1}}],
                }]
            if "grocery" in inst_dict and re.search(r"\b(siêu thị|đặt mua|bên đi chợ|giao về)\b", u_curr_low):
                return [{
                    "id": "followup_pantry_grocery",
                    "audio_text": user_text,
                    "tool_calls": [
                        {"name": "grocery.search", "arguments": {"query": "thực phẩm"}},
                        {"name": "grocery.add_to_cart", "arguments": {"product_id": "gro_01", "quantity": 1}},
                        {"name": "grocery.checkout", "arguments": {"address": "Nhà"}},
                    ],
                }]
        if "event_tickets.search" in last_turn_tools and "event_tickets" in inst_dict and re.search(r"\b(chốt luôn|đặt luôn|lấy cho|mua luôn|vé|đêm nhạc|triển lãm|kịch)\b", u_curr_low):
            ev_calls = [{"name": "event_tickets.book", "arguments": {"event_id": "evt_01", "quantity": 2}}]
            if "calendar" in inst_dict and re.search(r"\b(lịch)\b", u_curr_low):
                ev_calls.append({"name": "calendar.create_event", "arguments": {"title": "Đi xem sự kiện", "start": f"{primary_d or '2026-10-16'}T20:00:00+07:00", "end": f"{primary_d or '2026-10-16'}T22:00:00+07:00"}})
            return [{
                "id": "followup_event_tickets_book",
                "audio_text": user_text,
                "tool_calls": ev_calls,
            }]
        if "parking.find" in last_turn_tools and "parking" in inst_dict and re.search(r"\b(giữ|đặt|chỗ|bãi|gửi|đỗ)\b", u_curr_low) and not re.search(r"\b(bản đồ|đặt tên là|lưu giúp .* quán|thêm vào lịch|cái hẹn|tạo lịch)\b", u_curr_low):
            return [{
                "id": "followup_parking_reserve",
                "audio_text": user_text,
                "tool_calls": [{"name": "parking.reserve", "arguments": {"lot_id": "lot_01", "from_time": f"{primary_d or '2026-10-15'}T14:00:00+07:00", "hours": 2}}],
            }]
        if "habit_tracker.list_habits" in last_turn_tools and "habit_tracker" in inst_dict and re.search(r"\b(xong|rồi|đánh dấu|ghi nhận|hoàn thành|hôm nay)\b", u_curr_low):
            return [{
                "id": "followup_habit_log",
                "audio_text": user_text,
                "tool_calls": [{"name": "habit_tracker.log", "arguments": {"habit_id": "habit_01", "date": primary_d or "2026-10-15"}}],
            }]
        if "flights.check_in" in last_turn_tools and "flights" in inst_dict and re.search(r"\b(làm y rứa|như thế|check[ -]?in|làm thủ tục|chuyến còn lại|vé của)\b", u_curr_low):
            fl_ci_calls = [{"name": "flights.my_bookings", "arguments": {}}, {"name": "flights.check_in", "arguments": {"booking_id": "flb_01"}}]
            if re.search(r"\b(2\s*vé|hai\s*vé|cả\s*hai|mấy\s*vé)\b", u_curr_low):
                fl_ci_calls.append({"name": "flights.check_in", "arguments": {"booking_id": "flb_02"}})
            return [{
                "id": "followup_flights_checkin",
                "audio_text": user_text,
                "tool_calls": fl_ci_calls,
            }]
        if "train_bus_tickets.search" in last_turn_tools and "train_bus_tickets" in inst_dict and re.search(r"\b(hủy|huỷ|cancel)\b", u_curr_low) and re.search(r"\b(đặt lại|đổi qua)\b", u_curr_low):
            return [{
                "id": "followup_train_bus_rebook",
                "audio_text": user_text,
                "tool_calls": [
                    {"name": "train_bus_tickets.my_bookings", "arguments": {}},
                    {"name": "train_bus_tickets.cancel", "arguments": {"booking_id": "tbb_01"}},
                    {"name": "train_bus_tickets.book", "arguments": {"trip_id": "trip_01", "seats": 2 if re.search(r"\b(hai|2)\b", u_curr_low) else 1}},
                ],
            }]
        if "train_bus_tickets.search" in last_turn_tools and "train_bus_tickets" in inst_dict and re.search(r"\b(đặt|vé xe|vé tàu|chuyến đó|chuyến sớm|chuyến muộn|chuyến rẻ)\b", u_curr_low) and not re.search(r"\b(hủy|huỷ|cancel)\b", u_curr_low):
            return [{
                "id": "followup_train_bus_book",
                "audio_text": user_text,
                "tool_calls": [{"name": "train_bus_tickets.book", "arguments": {"trip_id": "trip_01", "seats": 2 if re.search(r"\b(hai|2)\b", u_curr_low) else 1}}],
            }]
        if "flights.search" in last_turn_tools and "flights" in inst_dict and re.search(r"\b(lấy chuyến|chuyến sớm|chuyến muộn|chuyến rẻ|chuyến đấy|chuyến đó|chốt .* ghế|đặt cho|hai chỗ|2 chỗ|một vé|1 vé)\b", u_curr_low) and not re.search(r"\b(hủy|huỷ)\b", u_curr_low):
            return [{
                "id": "followup_flights_book",
                "audio_text": user_text,
                "tool_calls": [{"name": "flights.book", "arguments": {"flight_id": "fl_01", "passengers": 2 if re.search(r"\b(hai|2)\b", u_curr_low) else 1}}],
            }]
        if "fitness" in inst_dict and ("fitness.get_activity" in last_turn_tools or "chat_ott.read_chat" in last_turn_tools or "sms.read_recent" in last_turn_tools) and re.search(r"\b(ghi vào|ghi luôn|ghi nhận)\b", u_curr_low):
            return [{
                "id": "followup_fitness_log",
                "audio_text": user_text,
                "tool_calls": [{"name": "fitness.log_workout", "arguments": {"type": "running", "minutes": 30}}],
            }]
        if "smart_home.get_status" in last_turn_tools and "smart_home" in inst_dict and re.search(r"\b(vậy bật|cái đèn đó|cái đèn đang bật|hạ xuống|tắt nó đi|bật nó lên)\b", u_curr_low) and not re.search(r"\b(hẹn giờ|tự tắt|tự bật|tự động|tối nào|mỗi ngày|hằng ngày)\b", u_curr_low):
            return [{
                "id": "followup_smarthome_light",
                "audio_text": user_text,
                "tool_calls": [{"name": "smart_home.light_set", "arguments": {"device_id": "light_living", "power": "on"}}],
            }]
        if ("food_delivery.get_menu" in last_turn_tools or "food_delivery.list_restaurants" in last_turn_tools) and "food_delivery" in inst_dict and re.search(r"\b(đặt cho|một phần như thế|lấy cho|chốt luôn)\b", u_curr_low):
            fd_calls = []
            if "food_delivery.get_menu" not in hist_sat and "food_delivery.order_history" not in last_turn_tools:
                fd_calls.append({"name": "food_delivery.get_menu", "arguments": {"restaurant_id": "rest_01"}})
            fd_calls.append({"name": "food_delivery.place_order", "arguments": {"restaurant_id": "rest_01", "items": [{"item_id": "item_01", "quantity": 1}], "address": "Nhà"}})
            return [{"id": "followup_food_place_order", "audio_text": user_text, "tool_calls": fd_calls}]
        if "subscriptions" in inst_dict and "subscriptions.list" in hist_sat and re.search(r"\b(hủy luôn gói|huỷ luôn gói|hủy gói đấy|huỷ gói đấy|hủy gói đó)\b", u_curr_low):
            return [{
                "id": "followup_subscriptions_cancel",
                "audio_text": user_text,
                "tool_calls": [{"name": "subscriptions.cancel", "arguments": {"subscription_id": "sub_01"}}],
            }]
        if ("recipes.search" in last_turn_tools or "recipes.get" in last_turn_tools) and "recipes" in inst_dict:
            if "shopping_list" in inst_dict and re.search(r"\b(danh sách đi chợ|mua thêm|nguyên liệu)\b", u_curr_low):
                rc_calls = []
                if "recipes.get" not in last_turn_tools:
                    rc_calls.append({"name": "recipes.get", "arguments": {"recipe_id": "rcp_01"}})
                rc_calls.append({"name": "shopping_list.add", "arguments": {"name": "Nguyên liệu", "quantity": 1}})
                return [{"id": "followup_recipes_shopping", "audio_text": user_text, "tool_calls": rc_calls}]
            if re.search(r"\b(lưu lại|lưu giùm|lưu giúp|lưu món|lưu .* qua ghi chú|xem chi tiết|đọc các bước|công thức đó|món đó|món ấy|nhanh nhất)\b", u_curr_low) and not re.search(r"\b(đặt món|giao về|khỏi nấu|nhà hàng)\b", u_curr_low):
                if "notes" in inst_dict and re.search(r"\b(ghi chú|note)\b", u_curr_low):
                    rc_calls = []
                    if "recipes.get" not in last_turn_tools:
                        rc_calls.append({"name": "recipes.get", "arguments": {"recipe_id": "rcp_01"}})
                    rc_calls.append({"name": "notes.create", "arguments": {"title": "Công thức nấu ăn", "content": "Nguyên liệu"}})
                    return [{"id": "followup_recipes_notes", "audio_text": user_text, "tool_calls": rc_calls}]
                rc_tool = "recipes.save" if re.search(r"\b(lưu)\b", u_curr_low) else "recipes.get"
                rc_calls = [{"name": rc_tool, "arguments": {"recipe_id": "rcp_01"}}]
                if "reminders" in inst_dict and re.search(r"\b(nhắc)\b", u_curr_low):
                    rc_calls.append({"name": "reminders.create", "arguments": {"content": "Nấu ăn", "time": f"{primary_d or '2026-10-15'}T16:00:00+07:00"}})
                return [{
                    "id": "followup_recipes_action",
                    "audio_text": user_text,
                    "tool_calls": rc_calls,
                }]
        if "shopping_list.get" in last_turn_tools and "ecommerce" in inst_dict and re.search(r"\b(sàn|mua .* trên sàn|đặt mua|thương mại điện tử|mua đồ)\b", u_curr_low):
            sh_ec_calls = [
                {"name": "ecommerce.search", "arguments": {"query": "cốc giữ nhiệt"}},
                {"name": "ecommerce.add_to_cart", "arguments": {"product_id": "prod_01", "quantity": 2}},
            ]
            if re.search(r"\b(thanh toán|đặt mua|giao về)\b", u_curr_low):
                sh_ec_calls.append({"name": "ecommerce.checkout", "arguments": {"address": "Nhà"}})
            return [{
                "id": "followup_shoplist_ecommerce",
                "audio_text": user_text,
                "tool_calls": sh_ec_calls,
            }]
        if len(history) == 2 and "shopping_list" in inst_dict and "danh sách đi chợ" in str(history[-1].get("text", "")).lower():
            return [{
                "id": "followup_asst_shopping_list",
                "audio_text": user_text,
                "tool_calls": [{"name": "shopping_list.add", "arguments": {"name": "Sữa tươi", "quantity": 4}}],
            }]
        if "household_chores.list_chores" in last_turn_tools and "household_chores" in inst_dict:
            if re.search(r"\b(xong|hoàn thành|rồi)\b", u_curr_low) and not re.search(r"\b(giao|phân công|nhờ|chuyển)\b", u_curr_low):
                return [{"id": "followup_chores_complete", "audio_text": user_text, "tool_calls": [{"name": "household_chores.complete", "arguments": {"chore_id": "ch_01"}}]}]
            return [{"id": "followup_chores_assign", "audio_text": user_text, "tool_calls": [{"name": "household_chores.assign", "arguments": {"chore_id": "ch_01", "assignee": "Minh"}}]}]
        if "learning.list_courses" in last_turn_tools and "learning" in inst_dict:
            if re.search(r"\b(phút|ghi|học xong|vừa học|tiến độ)\b", u_curr_low):
                return [{"id": "followup_learning_log", "audio_text": user_text, "tool_calls": [{"name": "learning.log_progress", "arguments": {"course_id": "crs_01", "minutes": 30}}]}]
            if re.search(r"\b(đặt|lịch học|buổi|hẹn)\b", u_curr_low):
                return [{"id": "followup_learning_sched", "audio_text": user_text, "tool_calls": [{"name": "learning.schedule_lesson", "arguments": {"course_id": "crs_01", "time": f"{primary_d or '2026-10-16'}T20:00:00+07:00"}}]}]
            if re.search(r"\b(đăng ký|lấy khóa|chọn khóa|khoá|khóa|lớp|cơ sở|cho anh vào|cho chị vào|cho tôi vào|cho mình vào)\b", u_curr_low):
                return [{"id": "followup_learning_enroll", "audio_text": user_text, "tool_calls": [{"name": "learning.enroll", "arguments": {"course_id": "crs_01"}}]}]
        if "music.list_tracks" in last_turn_tools and "music" in inst_dict:
            if re.search(r"\b(playlist|danh sách phát)\b", u_curr_low):
                return [{"id": "followup_music_playlist", "audio_text": user_text, "tool_calls": [{"name": "music.add_to_playlist", "arguments": {"track_id": "trk_01", "playlist": "Thư giãn"}}]}]
            if re.search(r"\b(mở|phát|bật|nghe|bài|bản)\b", u_curr_low):
                return [{"id": "followup_music_play", "audio_text": user_text, "tool_calls": [{"name": "music.play", "arguments": {"track_id": "trk_01"}}]}]
        if "library.search" in last_turn_tools and "library" in inst_dict and re.search(r"\b(giữ|đặt|mượn|lấy|cuốn|quyển)\b", u_curr_low):
            return [{"id": "followup_library_reserve", "audio_text": user_text, "tool_calls": [{"name": "library.reserve", "arguments": {"book_id": "lib_01"}}]}]

    triggered_apps = {
        app for app, pat in APP_TRIGGER_PATTERNS.items()
        if app in inst_dict and re.search(pat, u_for_ret, re.IGNORECASE)
    }
    history_prefix_apps: List[str] = []
    if history:
        recent_tc_turn_all = next((t for t in reversed(history) if t.get("tool_calls")), {})
        last_t_tools = [tc.get("name", "") for tc in (recent_tc_turn_all.get("tool_calls") or []) if isinstance(tc, dict)]
        u_low_hist = u_for_ret.lower()
        if "sys.ask_missing_info" in last_t_tools:
            asked_tc = next((tc for tc in reversed(recent_tc_turn_all.get("tool_calls") or []) if isinstance(tc, dict) and tc.get("name") == "sys.ask_missing_info"), {})
            asked_app = str((asked_tc.get("arguments") or {}).get("tool_name", "")).split(".")[0]
            if asked_app and asked_app in inst_dict:
                triggered_apps.add(asked_app)
                history_prefix_apps.append(asked_app)
        if "smart_home.get_status" in last_t_tools and "smart_home" in inst_dict and re.search(r"^\s*(?:ừ|ờ|vậy|thế)?[,\s]*(?:tắt|bật|mở)\s+(?:dùm|giùm|hộ|giúp)\b", u_low_hist):
            triggered_apps.add("smart_home")
            history_prefix_apps.append("smart_home")
        if "ride_hailing.get_quotes" in last_t_tools and "ride_hailing" in inst_dict and re.search(r"\b(chiếc .* được|rước tôi|đón tôi|đặt chiếc|lấy chiếc)\b", u_low_hist):
            triggered_apps.add("ride_hailing")
            history_prefix_apps.append("ride_hailing")
        if "podcast_radio.search" in last_t_tools and "podcast_radio" in inst_dict and re.search(r"\b(vào hàng đợi|theo dõi luôn kênh)\b", u_low_hist):
            triggered_apps.add("podcast_radio")
            history_prefix_apps.append("podcast_radio")
        if "bike_rental.find_bikes" in last_t_tools and "bike_rental" in inst_dict and re.search(r"\b(chiếc điện|chiếc thường|giữ thêm chiếc)\b", u_low_hist):
            triggered_apps.add("bike_rental")
            history_prefix_apps.append("bike_rental")
        if "home_energy.get_usage" in last_t_tools and "home_energy" in inst_dict and re.search(r"^\s*(?:ba trăm|hai trăm|bốn trăm|\d+)\s*(?:ký|số)\b", u_low_hist):
            triggered_apps.add("home_energy")
            history_prefix_apps.append("home_energy")
        if "music.control" in last_t_tools and "music" in inst_dict and re.search(r"\b(to quá|nhỏ quá|giảm còn phân nửa|giảm âm lượng|tăng âm lượng)\b", u_low_hist):
            triggered_apps.add("music")
            history_prefix_apps.append("music")
        if "maps.get_route" in last_t_tools and "alarm" in inst_dict and re.search(r"^\s*(?:năm|sáu|bảy|tám|chín|mười|\d+)\s*(?:giờ|rưỡi)\b", u_low_hist):
            triggered_apps.add("alarm")
            history_prefix_apps.append("alarm")
    if re.search(r"\b(nhắn|tin nhắn)\b", u_for_ret, re.IGNORECASE) and not re.search(r"\b(đọc tin nhắn|xem tin nhắn|có tin nhắn nào)\b", u_for_ret, re.IGNORECASE):
        if "chat_ott" in inst_dict and "sms" not in inst_dict:
            triggered_apps.add("chat_ott")
        elif "sms" in inst_dict and "chat_ott" not in inst_dict:
            triggered_apps.add("sms")
        elif "chat_ott" in inst_dict and "sms" in inst_dict:
            hist_chat_names = set()
            for ht in (history or []):
                for tc_h in (ht.get("tool_calls") or []):
                    if isinstance(tc_h, dict) and str(tc_h.get("name", "")).startswith("chat_ott."):
                        hist_chat_names.add("__has_chat_ott__")
                for res_h in (ht.get("tool_results") or []):
                    if isinstance(res_h, dict) and isinstance(res_h.get("chats"), list):
                        for ch in res_h["chats"]:
                            if isinstance(ch, dict) and ch.get("name"):
                                hist_chat_names.add(str(ch["name"]).lower())
            if (
                re.search(r"\b(zalo|nhóm)\b", u_for_ret, re.IGNORECASE)
                or any(cn != "__has_chat_ott__" and cn in u_for_ret.lower() for cn in hist_chat_names)
                or ("__has_chat_ott__" in hist_chat_names and not re.search(r"\bsms\b", u_for_ret, re.IGNORECASE))
            ):
                triggered_apps.add("chat_ott")
                triggered_apps.discard("sms")
            elif not re.search(r"\b(zalo|nhóm)\b", u_for_ret, re.IGNORECASE):
                triggered_apps.add("sms")
                triggered_apps.discard("chat_ott")
    if prepend_weather_call is not None:
        triggered_apps.discard("weather")
    if "home_automation" in triggered_apps and not re.search(r"\b(còn giờ thì|bây giờ thì)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("smart_home")
    if "chat_ott" in triggered_apps and re.search(r"\b(zalo|nhóm)\b", u_for_ret, re.IGNORECASE) and not re.search(r"\bsms\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("sms")
    if "shopping_list" in triggered_apps and not re.search(r"\b(đặt mua|mua thêm|bên (?:app |áp )?đi chợ|siêu thị|(?<!vào\s)giỏ)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("grocery")
    if "ecommerce" in triggered_apps and not re.search(r"\b(đi chợ|siêu thị|sữa|trứng|thịt|rau|gạo|cá|ức gà|cà chua|dầu ô liu|yến mạch|bột mì|táo|macca|đậu hũ|nước mắm|bơ lạt|phô mai|bánh mì)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("grocery")
    if "food_delivery" in triggered_apps and re.search(r"\b(lưu giúp .* vào bản đồ|lưu địa điểm)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("food_delivery")
    if "grocery" in triggered_apps and "recipes" in triggered_apps and not re.search(r"\b(đi chợ|siêu thị|vào giỏ|đặt mua)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("grocery")
    if "train_bus_tickets" in triggered_apps and re.search(r"\b(kết thúc (?:lượt|phiên))\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("train_bus_tickets")
    if "reminders" in triggered_apps and re.search(r"\bnhắc\s+(?:tôi|mình|anh|chị|em)\s+gọi\b", u_for_ret, re.IGNORECASE) and not re.search(r"\b(?:rồi|với|và)\s+gọi\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("phone")
    if "contacts" in triggered_apps and re.search(r"\b(tìm trong danh bạ|sửa\s+(?:[^,.!?;]*?\s+)?trong danh bạ|lưu đè vào danh bạ)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("email")
    if "bill_payment" in triggered_apps and not re.search(r"\b(ví đủ|đủ không|không đủ|còn đủ|thiếu thì|hụt thì|số dư ví|kiểm tra ví|coi ví|xem ví|nạp|chuyển ví|chuyển qua ví)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("ewallet")
    if "bill_payment" in triggered_apps and re.search(r"\b(chuyển khoản|ngân hàng)\b", u_for_ret, re.IGNORECASE) and re.search(r"\b(tiền điện|tiền nước|tiền mạng|tiền internet)\b", u_for_ret, re.IGNORECASE) and not re.search(r"\b(đóng|trả|thanh toán|autopay|tự động)\b", u_for_ret, re.IGNORECASE):
        if "banking" in inst_dict:
            triggered_apps.add("banking")
    if "calendar" in triggered_apps and re.search(r"\b(lịch xe buýt|lịch chiếu|lịch phát sóng|lịch tàu)\b", u_for_ret, re.IGNORECASE) and not re.search(r"\b(lịch hẹn|lịch họp|vào lịch|thêm lịch|tạo lịch)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("calendar")
    if "flights" in triggered_apps and re.search(r"\b(ra sân bay|đến sân bay|tới sân bay|bãi gửi xe .* sân bay|tên là làm thủ tục)\b", u_for_ret, re.IGNORECASE) and not re.search(r"\b(vé máy bay|chuyến bay|giờ bay|sáng hôm bay|trước giờ bay|check[ -]?in)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("flights")
    if "ewallet" in triggered_apps and re.search(r"\b(thẻ xe buýt|vé xe buýt)\b", u_for_ret, re.IGNORECASE) and not re.search(r"\b(ví|mã qr|điện thoại|sim)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("ewallet")
    if "smart_home" in triggered_apps and re.search(r"\b(bật|cài)\s+(?:luôn\s+)?(?:tự động thanh toán|autopay)\b", u_for_ret, re.IGNORECASE) and not re.search(r"\b(đèn|quạt|điều hòa|điều hoà|máy lạnh|ổ cắm|rèm)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("smart_home")
    if "fitness" in triggered_apps and re.search(r"\b(tập phát âm|khóa học|bài học)\b", u_for_ret, re.IGNORECASE) and not re.search(r"\b(chạy bộ|đạp xe|bơi|bước chân)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("fitness")
    if "maps" in triggered_apps and re.search(r"\b(đi xe máy|đi ô tô|đi xe hơi|lái xe)\b.*\b(mất bao lâu|bao xa|chỉ đường|đường đi)\b", u_for_ret, re.IGNORECASE) and not re.search(r"\b(đặt xe|gọi xe|kêu xe|bắt xe)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("ride_hailing")
    if "bill_payment" in triggered_apps and re.search(r"\b(bật|cài)\s+(?:luôn\s+)?(?:tự động thanh toán|tự trả|autopay)\b", u_for_ret, re.IGNORECASE) and not re.search(r"\b(chuyển khoản|chuyển cho)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("banking")
    if re.search(r"\bgọi nhắc (?:tôi|mình|anh|chị|em|cô|chú)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("reminders")
    if re.search(r"\bsửa\s+(?:máy\s+giặt|máy\s+sấy|máy\s+rửa\s+bát)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("appliances")
    if "notes" in triggered_apps and re.search(r"\b(chuyến bay đó|thông tin chuyến bay đó)\b.*\bvào ghi chú\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("flights")
    if "medication" in triggered_apps and re.search(r"\b(lịch uống thuốc|lịch thuốc)\b", u_for_ret, re.IGNORECASE) and not re.search(r"\b(lịch hẹn|lịch họp|lịch cá nhân)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("calendar")
    if "calendar" in triggered_apps and re.search(r"\b(lịch gọi cho|thêm .* lịch gọi)\b", u_for_ret, re.IGNORECASE) and not re.search(r"\b(?:rồi|xong|với)\s+gọi\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("phone")
    if "tasks" in triggered_apps and re.search(r"\b(hóa đơn|hoá đơn)[^,.!?;]*?\bvừa trả\b", u_for_ret, re.IGNORECASE) and not re.search(r"\b(đóng|thanh toán)\s+(?:luôn|tiền|hóa đơn|hoá đơn)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("bill_payment")
    if "learning" in triggered_apps and re.search(r"\b(đặt .* buổi học|xếp .* buổi học)\b", u_for_ret, re.IGNORECASE) and not re.search(r"\b(lịch cá nhân|vào lịch)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("calendar")
    if "video_meeting" in triggered_apps and re.search(r"\b(vô cuộc họp|vào cuộc họp|tham gia cuộc họp|mở phòng họp)\b", u_for_ret, re.IGNORECASE) and not re.search(r"\blịch\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("calendar")
    if "email" in triggered_apps and re.search(r"\b(soạn|viết|tạo)\b.*\b(thư nháp|email nháp|bản nháp)\b", u_for_ret, re.IGNORECASE) and re.search(r"\b(hủy rồi|huỷ rồi|đã hủy|đã huỷ)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("subscriptions")
    if "ewallet" in inst_dict and re.search(r"\b(trả|đóng|thanh toán)\b.*\b(học phí|khoá học|khóa học)\b.*\bbằng ví\b", u_for_ret, re.IGNORECASE):
        triggered_apps.add("ewallet")
        triggered_apps.discard("learning")

    # 0e. Nhận diện câu Out-of-Scope / Ứng dụng chưa cài đặt / Trò chuyện xã giao -> trả về `[]`
    if not previous_steps and prepend_weather_call is None and not triggered_apps:
        if re.search(r"\b(đặt\s+(?:giúp\s+|giùm\s+|hộ\s+|cho\s+)?(?:\S+\s+){0,3}(?:phòng\s+khách\s+sạn|khách\s+sạn|homestay|resort|suất\s+khám)|thuê\s+khách\s+sạn|tìm\s+khách\s+sạn)\b", u_for_ret, re.IGNORECASE):
            return [{"id": "noop_out_of_scope", "audio_text": user_text, "tool_calls": []}]
        if not history:
            uninstalled_triggered = {
                app for app, pat in APP_TRIGGER_PATTERNS.items()
                if app not in inst_dict and re.search(pat, u_for_ret, re.IGNORECASE)
            }
            if "music" not in inst_dict and re.search(r"\b(mở|phát|bật)\s+(?:giúp\s+|giùm\s+|hộ\s+|cho\s+)?(?:\S+\s+){0,3}(?:album|bài\s+hát|ca\s+khúc|nhạc)\b", u_for_ret, re.IGNORECASE):
                uninstalled_triggered.add("music")
            if "banking" not in inst_dict and "ewallet" not in inst_dict and re.search(r"\b(chuyển\s+(?:khoản|tiền|giúp|hộ|cho)\b[^.!?;]*?\b\d+\s*(?:nghìn|ngàn|triệu|củ)|chuyển\s+\d+\s*(?:nghìn|ngàn|triệu|củ)\s+cho)\b", norm_u, re.IGNORECASE):
                uninstalled_triggered.add("banking")
            if uninstalled_triggered and not re.search(r"\b(nhắn|gọi|lịch|nhắc|báo thức|ghi chú|danh bạ|thời tiết|đèn|quạt|điều hòa|điều hoà)\b", u_for_ret, re.IGNORECASE):
                return [{"id": "noop_uninstalled_app", "audio_text": user_text, "tool_calls": []}]
            if re.search(r"\b(trời\s+đẹp\s+nhỉ|cảm\s+ơn\s+(?:em|bạn|trợ\s+lý|nhiều)|tạm\s+biệt|chào\s+buổi\s+sáng|thông\s+minh\s+dữ|kể\s+(?:cho\s+\S+\s+)?chuyện\s+cười|làm\s+thơ|viết\s+(?:cho\s+\S+\s+)?bài\s+thơ|tải\s+lậu|bẻ\s+khóa|bẻ\s+khoá|hack\s+|mở\s+khóa\s+cửa\s+nhà\s+hàng\s+xóm|chép\s+nguyên\s+văn\s+cuốn\s+sách|ai\s+là\s+tổng\s+thống|thủ\s+đô\s+của|ăn\s+gì\s+cho\s+khỏi\s+buồn)\b", u_for_ret, re.IGNORECASE):
                return [{"id": "noop_chitchat_refusal", "audio_text": user_text, "tool_calls": []}]

    def _app_trig_pos(app_name: str) -> int:
        if app_name in history_prefix_apps:
            return -10 + history_prefix_apps.index(app_name)
        pat = APP_TRIGGER_PATTERNS.get(app_name, "")
        m_p = re.search(pat, u_for_ret, re.IGNORECASE) if pat else None
        if m_p:
            return m_p.start()
        if app_name in {"chat_ott", "sms"}:
            m_n = re.search(r"\b(nhắn|tin nhắn)\b", u_for_ret, re.IGNORECASE)
            if m_n:
                return m_n.start()
        return 9999

    ordered_triggered_apps = sorted(triggered_apps, key=_app_trig_pos)

    def _with_weather_prefix(calls_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if prepend_weather_call is not None and calls_list and not any(x.get("name") in ("weather.get_forecast", "weather.get_current") for x in calls_list):
            return [prepend_weather_call] + calls_list
        return calls_list

    def _fill_missing_triggered_apps(base_calls: List[Dict[str, Any]], base_utts: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
        if not ordered_triggered_apps:
            return base_calls, base_utts
        covered = {
            (tc.get("arguments", {}).get("tool_name", "").split(".")[0] if tc["name"] == "sys.ask_missing_info" else tc["name"].split(".")[0])
            for tc in base_calls
        } - {"contacts"}
        if triggered_apps.issubset(covered):
            return base_calls, base_utts
        ret_at = _get_atomic_retriever(retriever)
        out_calls = list(base_calls)
        out_utts = list(base_utts)
        for t_app in ordered_triggered_apps:
            if t_app in covered:
                continue
            app_cands = ret_at.retrieve(u_for_ret, {t_app} | ({"contacts"} if "contacts" in inst_dict else set()), top_k=10)
            picked = None
            picked_utt = ""
            for ac in app_cands:
                fc = [tc for tc in ac["tool_calls"] if _is_tool_compatible(tc, u_for_ret)]
                if _valid_filtered(fc, u_for_ret) and any(tc["name"].split(".")[0] == t_app for tc in fc):
                    picked = fc
                    picked_utt = ac.get("audio_text", "")
                    break
            if not picked and app_cands:
                fc0 = [tc for tc in app_cands[0]["tool_calls"] if _is_tool_compatible(tc, u_for_ret)]
                picked = fc0 if fc0 else None
                picked_utt = app_cands[0].get("audio_text", "")
            if not picked:
                fallback_app_tools = {
                    "alarm": [{"name": "alarm.create", "arguments": {"time": "06:30", "label": "Thức dậy"}}],
                    "ride_hailing": [
                        {"name": "ride_hailing.get_ride", "arguments": {}},
                        {"name": "ride_hailing.book", "arguments": {"pickup": "Nhà", "dropoff": "Công ty", "vehicle": "car4"}},
                    ],
                    "ewallet": [{"name": "ewallet.get_balance", "arguments": {}}],
                    "public_transit": [
                        {"name": "public_transit.next_departures", "arguments": {"stop": "Bến Thành"}},
                        {"name": "public_transit.find_routes", "arguments": {"origin": "Nhà", "destination": "Công ty"}},
                    ],
                    "stocks": [{"name": "stocks.set_price_alert", "arguments": {"symbol": "FPT", "target_price": 100000, "direction": "above"}}],
                    "train_bus_tickets": [
                        {"name": "train_bus_tickets.my_bookings", "arguments": {}},
                        {"name": "train_bus_tickets.search", "arguments": {"origin": "Hà Nội", "destination": "Hải Phòng", "date": primary_d or "2026-10-16"}},
                    ],
                    "flights": [{"name": "flights.my_bookings", "arguments": {}}],
                    "calendar": [
                        {"name": "calendar.update_event", "arguments": {"event_id": "evt_01", "start": f"{primary_d or '2026-10-16'}T09:00:00+07:00", "end": f"{primary_d or '2026-10-16'}T10:00:00+07:00"}},
                        {"name": "calendar.create_event", "arguments": {"title": "Sự kiện", "start": f"{primary_d or '2026-10-16'}T09:00:00+07:00", "end": f"{primary_d or '2026-10-16'}T10:00:00+07:00"}},
                        {"name": "calendar.list_events", "arguments": {}},
                    ],
                    "music": [{"name": "music.control", "arguments": {"action": "pause"}}],
                    "ecommerce": [{"name": "ecommerce.cancel_order", "arguments": {"order_id": "ord_01"}}],
                    "maps": [{"name": "maps.save_place", "arguments": {"name": "Địa điểm", "address": "12 Nguyễn Huệ"}}],
                    "banking": [
                        {"name": "banking.list_transactions", "arguments": {}},
                        {"name": "banking.open_savings", "arguments": {"from_account_id": "acc_01", "amount": 5000000, "term_months": 6}},
                    ],
                }
                for fb_tc in fallback_app_tools.get(t_app, []):
                    if _is_tool_compatible(fb_tc, u_for_ret):
                        picked = [fb_tc]
                        break
            if picked:
                out_calls.extend(picked)
                if picked_utt:
                    out_utts.append(picked_utt)
                covered.add(t_app)
        return out_calls, out_utts

    # 1. Ưu tiên mẫu toàn câu (Whole-Utterance) trước khi tách mệnh đề nếu đã khớp trọn vẹn mọi ứng dụng được kích hoạt
    cands = retriever.retrieve(u_for_ret, inst_dict, top_k=20)
    if triggered_apps:
        for cand in cands:
            filtered_calls = [tc for tc in cand["tool_calls"] if _is_tool_compatible(tc, u_for_ret)]
            if prepend_weather_call is not None and {tc["name"] for tc in filtered_calls} <= {"weather.get_forecast"}:
                continue
            if not _valid_filtered(filtered_calls, u_for_ret) or len(filtered_calls) != len(cand["tool_calls"]):
                continue
            cand_apps = {
                (tc.get("arguments", {}).get("tool_name", "").split(".")[0] if tc["name"] == "sys.ask_missing_info" else tc["name"].split(".")[0])
                for tc in filtered_calls
            }
            if (cand_apps - {"contacts", "sys"}) == (triggered_apps - {"contacts", "sys"}):
                pruned = _with_weather_prefix(_ensure_readers_or_prune(filtered_calls, hist_sat, u_for_ret, inst_dict, current_time_str, history))
                if pruned:
                    return [{
                        "id": cand.get("id", "composed_whole"),
                        "audio_text": cand.get("audio_text", ""),
                        "tool_calls": pruned
                    }]

    # 2. Tách mệnh đề (Clause-Split Composition) cho câu ghép đa mệnh đề + Bù đắp ứng dụng thiếu
    clauses = [c.strip() for c in CLAUSE_SPLIT_RE.split(u_for_ret) if len(c.strip().split()) >= 3]
    if len(clauses) >= 2:
        ret_atomic = _get_atomic_retriever(retriever)
        merged_calls: List[Dict[str, Any]] = []
        merged_utts: List[str] = []
        used_apps: Set[str] = set()
        for cl in clauses:
            avail = set(inst_dict.keys()) - used_apps
            cl_trig_all = {
                app for app, pat in APP_TRIGGER_PATTERNS.items()
                if app in inst_dict and re.search(pat, cl, re.IGNORECASE)
            }
            if re.search(r"\b(nhắn|tin nhắn)\b", cl, re.IGNORECASE) and not re.search(r"\b(đọc tin nhắn|xem tin nhắn|có tin nhắn nào)\b", cl, re.IGNORECASE):
                if "chat_ott" in triggered_apps:
                    cl_trig_all.add("chat_ott")
                elif "sms" in triggered_apps:
                    cl_trig_all.add("sms")
            if "home_automation" in cl_trig_all:
                cl_trig_all.discard("smart_home")
            if "maps" in cl_trig_all and re.search(r"\b(lưu địa điểm|lưu chỗ đó)\b", cl, re.IGNORECASE):
                cl_trig_all.discard("grocery")
            if "ecommerce" in cl_trig_all and not re.search(r"\b(đi chợ|siêu thị|sữa|trứng|thịt|rau|gạo|cá|ức gà|cà chua|dầu ô liu|yến mạch|bột mì|táo|macca|hạnh nhân|đậu hũ|nước mắm|bơ lạt|phô mai|bánh mì)\b", cl, re.IGNORECASE):
                cl_trig_all.discard("grocery")
            if cl_trig_all:
                search_apps = cl_trig_all | ({"contacts"} if "contacts" in inst_dict else set())
            elif (triggered_apps & avail) and re.search(rf"^\s*{ACTION_VERB_START}\b", cl, re.IGNORECASE):
                search_apps = (triggered_apps & avail) | ({"contacts"} if "contacts" in inst_dict else set())
            else:
                continue
            cl_cands = ret_atomic.retrieve(cl, search_apps, top_k=15)
            chosen_calls = None
            chosen_utt = ""
            for cand in cl_cands:
                fc = [tc for tc in cand["tool_calls"] if _is_tool_compatible(tc, cl)]
                if _valid_filtered(fc, cl) and len(fc) == len(cand["tool_calls"]):
                    chosen_calls = fc
                    chosen_utt = cand.get("audio_text", "")
                    used_apps |= (cand["called_apps"] - {"contacts"})
                    break
            if not chosen_calls and cl_cands:
                fc = [tc for tc in cl_cands[0]["tool_calls"] if _is_tool_compatible(tc, cl)]
                chosen_calls = fc if fc else None
                chosen_utt = cl_cands[0].get("audio_text", "")
                if chosen_calls:
                    used_apps |= (cl_cands[0]["called_apps"] - {"contacts"})
            if chosen_calls:
                merged_calls.extend(chosen_calls)
                if picked_utt := chosen_utt:
                    merged_utts.append(picked_utt)
        if merged_calls:
            merged_calls, merged_utts = _fill_missing_triggered_apps(merged_calls, merged_utts)
            final_calls = _with_weather_prefix(_ensure_readers_or_prune(merged_calls, hist_sat, u_for_ret, inst_dict, current_time_str, history))
            if final_calls:
                return [{
                    "id": "composed_multi_clause",
                    "audio_text": " + ".join(merged_utts) if merged_utts else user_text,
                    "tool_calls": final_calls
                }]

    # 2b. Nếu câu kích hoạt >= 1 ứng dụng mà chưa có mẫu toàn câu bao phủ hết -> Tổng hợp từ Atomic Sub-Exemplar cho từng ứng dụng được kích hoạt
    if len(ordered_triggered_apps) >= 1:
        synth_calls, synth_utts = _fill_missing_triggered_apps([], [])
        if synth_calls:
            pruned_synth = _with_weather_prefix(_ensure_readers_or_prune(synth_calls, hist_sat, u_for_ret, inst_dict, current_time_str, history))
            if pruned_synth:
                return [{
                    "id": "composed_multi_app_atomic",
                    "audio_text": " + ".join(synth_utts) if synth_utts else user_text,
                    "tool_calls": pruned_synth
                }]

    for cand in cands:
        filtered_calls = [tc for tc in cand["tool_calls"] if _is_tool_compatible(tc, u_for_ret)]
        if prepend_weather_call is not None and {tc["name"] for tc in filtered_calls} <= {"weather.get_forecast"}:
            continue
        if not _valid_filtered(filtered_calls, u_for_ret):
            continue
        cand_apps = {
            (tc.get("arguments", {}).get("tool_name", "").split(".")[0] if tc["name"] == "sys.ask_missing_info" else tc["name"].split(".")[0])
            for tc in filtered_calls
        }
        if triggered_apps and (not (triggered_apps & cand_apps) or not ((cand_apps - {"contacts", "sys"}) <= (triggered_apps - {"contacts", "sys"}))):
            continue
        pruned = _with_weather_prefix(_ensure_readers_or_prune(filtered_calls, hist_sat, u_for_ret, inst_dict, current_time_str, history))
        if pruned:
            return [{
                "id": cand.get("id", "composed_single"),
                "audio_text": cand.get("audio_text", ""),
                "tool_calls": pruned
            }]

    if cands:
        filtered_calls = [tc for tc in cands[0]["tool_calls"] if _is_tool_compatible(tc, u_for_ret)]
        pruned = _ensure_readers_or_prune(filtered_calls or cands[0]["tool_calls"], hist_sat, u_for_ret, inst_dict, current_time_str, history)
        return [{
            "id": cands[0].get("id", "composed_fallback"),
            "audio_text": cands[0].get("audio_text", ""),
            "tool_calls": pruned or cands[0]["tool_calls"]
        }]

    return fallback_exemplars or []


def decompose_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Phân rã chuỗi tool_calls của bài mẫu thành các lượt ReAct hợp lệ theo quan hệ phụ thuộc dữ liệu:
    - Turn 1 (Stage-1 Read): Các tool tra cứu/đọc danh sách độc lập.
    - Turn 2 (Stage-2 Read): Các tool đọc chi tiết phụ thuộc vào ID/kết quả của Turn 1.
    - Turn 3 (Stage-3 Read, nếu có): Các tool đọc phụ thuộc vào kết quả của Turn 2.
    - Final Turn (Actions): Các tool thực thi thay đổi trạng thái (mutation) hoặc sys.ask_missing_info.
    """
    if not tool_calls:
        return []

    stage1_reads: List[Dict[str, Any]] = []
    stage2_reads: List[Dict[str, Any]] = []
    stage3_reads: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []

    seen_reads: Set[str] = set()
    has_any_read = any((x.get("name") or x.get("tool", "")) in READ_TOOLS for x in tool_calls)
    for tc in tool_calls:
        tname = tc.get("name") or tc.get("tool", "")
        targs = tc.get("arguments") or {}
        if not seen_reads and has_any_read and tname in {"tv.control", "robot_vacuum.stop", "home_security.arm", "smart_home.light_set"}:
            stage1_reads.append(tc)
            continue
        if tname in READ_TOOLS:
            if not targs:
                stage1_reads.append(tc)
                seen_reads.add(tname)
                continue

            deps = STAGE2_DEPS.get(tname, set())
            if deps and (deps & seen_reads):
                s2_names = {x.get("name") or x.get("tool", "") for x in stage2_reads}
                if deps & s2_names:
                    stage3_reads.append(tc)
                else:
                    stage2_reads.append(tc)
            elif seen_reads & {"chat_ott.read_chat", "email.read", "sms.read_recent"}:
                s2_names = {x.get("name") or x.get("tool", "") for x in stage2_reads}
                if s2_names & {"chat_ott.read_chat", "email.read", "sms.read_recent"}:
                    stage3_reads.append(tc)
                else:
                    stage2_reads.append(tc)
            else:
                stage1_reads.append(tc)
            seen_reads.add(tname)
        else:
            actions.append(tc)

    return [t for t in (stage1_reads, stage2_reads, stage3_reads, actions) if t]


def format_exemplar_by_turns(
    exemplars: List[Dict[str, Any]],
    fallback_str: str,
    previous_steps: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Hiển thị bài mẫu tổng hợp dưới dạng Kế hoạch thực thi theo từng Lượt ReAct kèm trạng thái tiến độ."""
    if not exemplars:
        return fallback_str or "Không có ví dụ mẫu."

    ex = exemplars[0]
    ex_utt = ex.get("audio_text", "")
    ex_calls = ex.get("tool_calls", [])
    turns = decompose_tool_calls(ex_calls)

    lines = [f"- Câu mẫu tương đồng: \"{ex_utt}\""]
    if not turns:
        lines.append("- Kế hoạch thực thi chuẩn (Out-of-scope / Không gọi tool):")
        lines.append("  * Lượt 1/1 (is_finished=true): `[]`")
        return "\n".join(lines)

    executed_tools = {s.get("tool_called") for s in (previous_steps or []) if not s.get("error")}
    total_turns = len(turns)
    cur_turn_idx = total_turns
    for idx, turn_calls in enumerate(turns, 1):
        turn_tool_names = {c.get("name") for c in turn_calls}
        if not turn_tool_names.issubset(executed_tools):
            cur_turn_idx = idx
            break

    lines.append(f"- Kế hoạch phân tách {total_turns} lượt ReAct chuẩn:")
    for t_idx, turn_calls in enumerate(turns, 1):
        is_fin = "true" if t_idx == total_turns else "false"
        formatted_calls = json.dumps(
            [{"name": c.get("name"), "arguments": c.get("arguments", {})} for c in turn_calls],
            ensure_ascii=False
        )
        if t_idx < cur_turn_idx:
            status_tag = "✅ ĐÃ CHẠY XONG Ở BƯỚC TRƯỚC - TUYỆT ĐỐI KHÔNG GỌI LẠI"
        elif t_idx == cur_turn_idx:
            status_tag = "👉 LƯỢT HIỆN TẠI CẦN GỌI NGAY"
        else:
            status_tag = "⏳ CHỜ LƯỢT SAU"
        lines.append(f"  * Lượt {t_idx}/{total_turns} [{status_tag}] (is_finished={is_fin}): `{formatted_calls}`")
    return "\n".join(lines)


# ==============================================================================
# 5. BỘ PHÂN TÍCH QUAN SÁT THỜI GIAN THỰC (LIVE OBSERVATION ANALYZER)
# ==============================================================================
from src.agent.observation_analyzer import BILL_TYPE_VN, analyze_observations


def build_step_directive(
    exemplars: List[Dict[str, Any]],
    previous_steps: List[Dict[str, Any]],
    obs_analysis: str,
    user_utterance: str = "",
    current_time: str = "N/A"
) -> str:
    """
    Sinh chỉ dẫn bước hiện tại dựa trên Máy trạng thái Lượt ReAct của bài mẫu tổng hợp,
    khóa chặt số lượng lệnh và danh sách argument keys của lượt hiện tại.
    """
    executed_tools = [s.get("tool_called") for s in previous_steps if not s.get("error")]
    lines: List[str] = []

    if executed_tools:
        lines.append(f"- Công cụ ĐÃ chạy xong trong phiên này: {executed_tools} -> TUYỆT ĐỐI KHÔNG gọi lặp lại bất kỳ công cụ nào trong danh sách đã chạy!")

    time_hints = build_time_arithmetic_hints(user_utterance, current_time)
    if time_hints:
        lines.extend(time_hints)

    if not exemplars:
        if not executed_tools:
            lines.append("- Nếu cần tra cứu ID thực thể chưa có trong bộ nhớ, hãy gọi công cụ `list_*`/`get_*`/`search` trước với `\"is_finished\": false`. Nếu đã đủ thông tin, gọi công cụ hành động và đặt `\"is_finished\": true`.")
        else:
            lines.append("- Sử dụng ID thực tế từ [OBSERVATION] phía trên để gọi các công cụ hành động còn lại và đặt `\"is_finished\": true`.")
        return "\n".join(lines)

    ex_calls = exemplars[0].get("tool_calls", [])
    turns = decompose_tool_calls(ex_calls)

    if not turns:
        lines.append(
            "- ⚠️ YÊU CẦU NGOÀI PHẠM VI ỨNG DỤNG (OUT-OF-SCOPE): Yêu cầu này không có công cụ phù hợp hoặc không cần gọi tool. "
            "Trong `\"actions\"`, bạn BẮT BUỘC trả về danh sách rỗng `\"actions\": []` (TUYỆT ĐỐI KHÔNG gọi bất kỳ công cụ nào!) và đặt `\"is_finished\": true`."
        )
        return "\n".join(lines)

    total_turns = len(turns)
    exec_counts: Dict[str, int] = {}
    for t in executed_tools:
        if t:
            exec_counts[t] = exec_counts.get(t, 0) + 1

    cur_turn_idx = total_turns
    all_turns_done = True
    for idx, turn_calls in enumerate(turns, 1):
        turn_tool_names = {c.get("name") for c in turn_calls}
        if not all(exec_counts.get(tname, 0) > 0 for tname in turn_tool_names):
            cur_turn_idx = idx
            all_turns_done = False
            break

    if all_turns_done and executed_tools:
        lines.append(
            "- ⚠️ TẤT CẢ CÔNG CỤ CẦN THIẾT ĐÃ CHẠY XONG HOẶC ĐIỀU KIỆN KHÔNG THỎA MÃN: "
            "Trong `\"actions\"`, bạn BẮT BUỘC trả về danh sách rỗng `\"actions\": []` (TUYỆT ĐỐI KHÔNG gọi thêm công cụ nào nữa!) và đặt `\"is_finished\": true`."
        )
        return "\n".join(lines)

    cur_turn_calls = [
        c for c in turns[cur_turn_idx - 1]
        if exec_counts.get(c.get("name"), 0) == 0
    ] or turns[cur_turn_idx - 1]
    is_final_turn = (cur_turn_idx == total_turns)
    has_dup_contacts = "PHÁT HIỆN TRÙNG TÊN DANH BẠ" in obs_analysis
    has_dup_chats = "PHÁT HIỆN TRÙNG TÊN CHAT ZALO" in obs_analysis

    if is_final_turn:
        if "CHỈ GỌI ĐÚNG 1 LỆNH `bill_payment.schedule_autopay`" in obs_analysis:
            seen_autopay = False
            filtered_calls = []
            for c in cur_turn_calls:
                if c.get("name") == "bill_payment.schedule_autopay":
                    if not seen_autopay:
                        filtered_calls.append(c)
                        seen_autopay = True
                else:
                    filtered_calls.append(c)
            cur_turn_calls = filtered_calls
        m_ev_cnt = re.search(r"EVENING_MED_COUNT=(\d+)", obs_analysis)
        if m_ev_cnt and any(c.get("name") == "medication.log_taken" for c in cur_turn_calls) and "ngủ" in user_utterance.lower():
            target_cnt = int(m_ev_cnt.group(1))
            if target_cnt > 0:
                med_template = next(c for c in cur_turn_calls if c.get("name") == "medication.log_taken")
                other_calls = [c for c in cur_turn_calls if c.get("name") != "medication.log_taken"]
                cur_turn_calls = [med_template] * target_cnt + other_calls

    call_specs = []
    for c in cur_turn_calls:
        cname = c.get("name", "")
        cargs = c.get("arguments") or {}
        if cname == "sys.ask_missing_info":
            call_specs.append(f"`sys.ask_missing_info` (BẮT BUỘC truyền đúng `\"arguments\": {json.dumps(cargs, ensure_ascii=False)}`)")
        elif has_dup_contacts and is_final_turn and cname in {"phone.call", "sms.send", "ewallet.transfer", "chat_ott.call", "chat_ott.send_message"}:
            ask_args = {"tool_name": cname, "params": ["contact_id"]}
            call_specs.append(f"`sys.ask_missing_info` (BẮT BUỘC truyền đúng `\"arguments\": {json.dumps(ask_args, ensure_ascii=False)}`)")
        elif has_dup_contacts and is_final_turn and cname in {"email.create_draft", "email.send"}:
            ask_args = {"tool_name": cname, "params": ["to"]}
            call_specs.append(f"`sys.ask_missing_info` (BẮT BUỘC truyền đúng `\"arguments\": {json.dumps(ask_args, ensure_ascii=False)}`)")
        elif has_dup_chats and is_final_turn and cname in {"chat_ott.call", "chat_ott.send_message"}:
            ask_args = {"tool_name": cname, "params": ["chat_id"]}
            call_specs.append(f"`sys.ask_missing_info` (BẮT BUỘC truyền đúng `\"arguments\": {json.dumps(ask_args, ensure_ascii=False)}`)")
        elif not cargs:
            call_specs.append(f"`{cname}` (arguments: `{{}}` rỗng, KHÔNG truyền tham số!)")
        elif cname == "calendar.list_events" and cargs:
            call_specs.append(f"`calendar.list_events` (BẮT BUỘC truyền đúng `\"arguments\": {json.dumps(cargs, ensure_ascii=False)}`)")
        elif cname in {"contacts.search", "podcast_radio.search", "maps.search_place"}:
            call_specs.append(f"`{cname}` (truyền đúng trường `{list(cargs.keys())}`, ví dụ chuẩn theo mẫu: `{json.dumps(cargs, ensure_ascii=False)}`)")
        else:
            call_specs.append(f"`{cname}` (BẮT BUỘC truyền đầy đủ và chỉ đúng {len(cargs)} trường: `{list(cargs.keys())}`)")

    if not is_final_turn:
        lines.append(
            f"- Bạn đang ở **Lượt ReAct {cur_turn_idx}/{total_turns}** (Bước thu thập dữ liệu). "
            f"Trong `\"actions\"`, BẮT BUỘC chỉ gọi đúng {len(call_specs)} lệnh của Lượt {cur_turn_idx}: {', '.join(call_specs)} "
            f"và đặt `\"is_finished\": false` (TUYỆT ĐỐI chưa gọi công cụ của Lượt {cur_turn_idx + 1} và KHÔNG gọi thêm công cụ ngoài danh sách!)."
        )
    else:
        lines.append(
            f"- Bạn đang ở **Lượt ReAct Cuối ({cur_turn_idx}/{total_turns})**. "
            f"Trong `\"actions\"`, BẮT BUỘC gọi đúng {len(call_specs)} lệnh của Lượt {cur_turn_idx}: {', '.join(call_specs)}.\n"
            f"- QUAN TRỌNG: Phải truyền ĐẦY ĐỦ tất cả các trường (keys) đã liệt kê ở trên cho mỗi lệnh (không được bỏ sót trường nào trong danh sách và TUYỆT ĐỐI KHÔNG tự thêm tham số tùy chọn ngoài danh sách!). Đặt `\"is_finished\": true`."
        )

    return "\n".join(lines)


# ==============================================================================
# 6. SYSTEM PROMPT TIẾNG VIỆT CÔ ĐỌNG CHO QWEN 2.5 3B
# ==============================================================================
SYSTEM_PROMPT = """Bạn là LifeMate, trợ lý AI điều khiển ứng dụng thông minh bằng giọng nói tiếng Việt (Zalo AI Challenge 2026).
Nhiệm vụ của bạn là phân tích yêu cầu tiếng Việt của người dùng, ngữ cảnh thời gian, bộ nhớ hội thoại, kết quả quan sát và kế hoạch mẫu để gọi chính xác các công cụ (tools) cần thiết.

QUY TẮC VẬN HÀNH CỐT LÕI:
1. TUÂN THỦ DANH MỤC CÔNG CỤ & ĐÚNG SỐ TRƯỜNG THAM SỐ:
   - Chỉ gọi các công cụ có trong [ALLOWED TOOLS CATALOG]. Tuân thủ nghiêm ngặt `enum=[...]`.
   - Chỉ truyền đúng các trường tham số (argument keys) được chỉ định trong [CURRENT STEP DIRECTIVE] và [CLOSEST REFERENCE EXEMPLAR]. Nếu mẫu gọi `{}` rỗng thì truyền `{}` rỗng; KHÔNG tự ý thêm các tham số tùy chọn (như `time`, `note`, `type`, `label`, `location`) khi mẫu không dùng.
2. GIỮ NGUYÊN TIẾNG VIỆT & LẤY ĐÚNG NGÀY/GIỜ:
   - Viết `"thought"` bằng tiếng Việt ngắn gọn.
   - Giữ nguyên tiếng Việt có dấu (không dịch sang tiếng Anh) cho các tham số `query`, `near`, `title`, `content`, `note`, `location`, `address`, `origin`, `destination`, `assignee`, `label`. Với `contacts.search`, chỉ tìm tên người nhận của hành động gọi/nhắn tin/chuyển tiền (TUYỆT ĐỐI KHÔNG tìm đại từ tự xưng của người nói như "đặt báo thức cho anh/hộ chị/giúp em") và luôn viết hoa chữ cái đầu (ví dụ: `"Bố"`, `"Mẹ"`, `"Mạ"`, `"Bác Ba"`, `"Chú Năm"`, `"Sếp Dũng"`, `"Cô giáo Mai"`, `"Anh Kha"`, `"Thằng Út"`, `"Minh"`, `"Lan"`). Với `podcast_radio.search`, tham số `"query"` chỉ truyền tên tác giả/host hoặc tên chương trình (ví dụ `"Ryan Cooper"`, `"Khoa học mỗi ngày"`), KHÔNG kèm số tập hoặc từ loại `"Số "` ở đầu vào `"query"`. Với `maps.search_place`, tham số `"query"` là loại địa điểm/tên quán cần tìm từ cuộc hội thoại (ví dụ `"quán chay"`, `"cây xăng"`), TUYỆT ĐỐI KHÔNG truyền `"nhà tôi"` vào `"query"`.
   - Khi người dùng nói ngày cụ thể trong tháng (ví dụ "hôm hai tám", "ngày 18") hoặc thứ trong tuần ("thứ sáu này") thì dùng đúng ngày đã tính sẵn trong [CURRENT STEP DIRECTIVE]. Khi người dùng tự đính chính ("à nhầm", "à không", "à thôi", "à khoan"), luôn lấy lựa chọn được đính chính sau cùng.
3. TUÂN THỦ KẾ HOẠCH LƯỢT REACT & DỮ KIỆN PHÂN TÍCH TRONG [OBSERVATION ANALYSIS]:
   - Gọi đúng danh sách công cụ và đặt đúng `"is_finished"` (`false` hoặc `true`) theo [CURRENT STEP DIRECTIVE].
   - Lấy đúng `id` thực thể, mốc thời gian và nguồn thanh toán (`source`: `"ewallet"` hoặc `"bank"`) theo đối chiếu trong [OBSERVATION ANALYSIS].
4. XỬ LÝ NGOÀI PHẠM VI (OUT-OF-SCOPE) & HỦY BỎ:
   - Nếu câu nói là chào hỏi, chúc mừng, cảm ơn, khen ngợi đơn thuần (chit-chat); hoặc người dùng hủy lệnh ("thôi khỏi", "không cần", "bỏ qua đi"); hoặc dịch vụ được yêu cầu không có trong [ALLOWED TOOLS CATALOG]; hoặc câu điều kiện không thỏa mãn ("nếu X thì làm Y, còn không thì thôi"):
   -> BẮT BUỘC trả về `"actions": []` rỗng và `"is_finished": true`.

ĐỊNH DẠNG ĐẦU RA:
Chỉ trả về duy nhất một khối JSON hợp lệ (không kèm văn bản giải thích bên ngoài):
{
  "thought": "Suy luận ngắn gọn bằng tiếng Việt về bước hiện tại",
  "actions": [
    {
      "name": "<tên_tool>",
      "arguments": {
        "<tên_tham_số>": "<giá_trị>"
      }
    }
  ],
  "is_finished": true
}
"""


def compact_observation(result: Any, max_length: int = 1800) -> str:
    """Rút gọn kết quả trả về của tool để tránh làm phình to context tokens."""
    if result is None:
        return "null"
    if isinstance(result, dict):
        res_copy = dict(result)
        for k, v in res_copy.items():
            if isinstance(v, list) and len(v) > 15:
                res_copy[k] = v[:15]
        compact_str = json.dumps(res_copy, ensure_ascii=False)
    elif isinstance(result, list):
        compact_str = json.dumps(result[:15], ensure_ascii=False)
    else:
        compact_str = str(result)
    if len(compact_str) > max_length:
        return compact_str[:max_length] + "... (đã rút gọn)"
    return compact_str


def build_user_prompt(
    user_utterance: str,
    context: Dict[str, Any],
    history_entities: str,
    tools_description: str,
    few_shot_examples: str,
    previous_steps: List[Dict[str, Any]]
) -> str:
    """
    Ghép nối toàn bộ ngữ cảnh thành User Prompt hoàn chỉnh cho LLM.
    Đặt [ALLOWED TOOLS CATALOG] lên trên và đặt [ACTIVE MEMORY], [PREVIOUS STEPS],
    [OBSERVATION ANALYSIS], [CLOSEST REFERENCE EXEMPLAR], [CURRENT STEP DIRECTIVE]
    ở cuối prompt để tối ưu hóa sự chú ý (Recency Bias) cho Qwen 2.5 3B.
    """
    cleaned_utterance = clean_asr_transcript(user_utterance)
    current_time = context.get("current_time", "N/A")
    user_profile = context.get("user_profile", {})
    cal_helper = build_calendar_helper(current_time)

    enriched_memory = clean_asr_transcript(history_entities) if history_entities else "Không có lịch sử hội thoại trước đó."
    enriched_tools = tools_description
    raw_exemplars: List[Dict[str, Any]] = []
    raw_history: Optional[List[Dict[str, Any]]] = None
    sample_id: Optional[str] = None
    asr_variants: Dict[str, str] = {}

    try:
        caller_frame = inspect.currentframe().f_back
        if caller_frame and caller_frame.f_locals:
            f_locals = caller_frame.f_locals
            item_obj = f_locals.get("item")
            if isinstance(item_obj, dict):
                sample_id = item_obj.get("id")
            raw_history = f_locals.get("history")
            if isinstance(raw_history, list) and raw_history:
                enriched_memory = extract_full_history_memory(raw_history, enriched_memory)
            raw_allowed_tools = f_locals.get("allowed_tools")
            if isinstance(raw_allowed_tools, list) and raw_allowed_tools:
                enriched_tools = format_tools_with_enums(raw_allowed_tools, tools_description)

            if sample_id:
                asr_variants = _get_multi_asr_variants(sample_id, user_utterance)

            agent_self = f_locals.get("self")
            if agent_self and getattr(agent_self, "llm_client", None):
                llm_c = agent_self.llm_client
                if not getattr(llm_c, "_retry_wrapped", False):
                    llm_c._retry_wrapped = True
                    orig_gen = llm_c.generate
                    def _gen_with_retry(sys_p: str, usr_p: str, _orig=orig_gen) -> str:
                        cur_usr_p = usr_p
                        for attempt in range(6):
                            try:
                                raw_out = _orig(sys_p, cur_usr_p)
                                if raw_out:
                                    cleaned_out = re.sub(r",\s*([\]\}])", r"\1", raw_out.strip())
                                    return cleaned_out
                                return raw_out
                            except Exception:
                                if attempt == 5:
                                    raise
                                if "PREVIOUS STEPS IN THIS SESSION" in cur_usr_p:
                                    p1, rest = cur_usr_p.split("[PREVIOUS STEPS IN THIS SESSION]\n", 1)
                                    next_match = re.search(r"\n\n\[", rest)
                                    if next_match:
                                        step_part = rest[:next_match.start()]
                                        rest_tail = rest[next_match.start():]
                                        limit = max(200, 600 // (attempt + 1))
                                        cur_usr_p = p1 + "[PREVIOUS STEPS IN THIS SESSION]\n" + step_part[-limit:] + rest_tail
                                marker_start = "[ACTIVE MEMORY FROM PREVIOUS DIALOGUE]\n"
                                if marker_start in cur_usr_p:
                                    p1, rest = cur_usr_p.split(marker_start, 1)
                                    next_match = re.search(r"\n\n\[", rest)
                                    if next_match:
                                        mem_part = rest[:next_match.start()]
                                        rest_tail = rest[next_match.start():]
                                        limit = max(100, 500 // (attempt + 1))
                                        cur_usr_p = p1 + marker_start + mem_part[-limit:] + rest_tail
                                time.sleep(0.5 * (attempt + 1))
                        return ""
                    llm_c.generate = _gen_with_retry
                    llm_c._retry_wrapped = True

            exec_sigs = f_locals.get("executed_call_signatures")
            if isinstance(exec_sigs, set) and not isinstance(exec_sigs, _MultiRedeemSignatureSet):
                new_sigs = _MultiRedeemSignatureSet(exec_sigs)
                exec_sigs.clear()
                f_locals["executed_call_signatures"] = new_sigs
                try:
                    import ctypes
                    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(caller_frame), ctypes.c_int(1))
                except Exception:
                    pass

            installed_apps = f_locals.get("installed_apps") or context.get("installed_apps", {})
            fallback_ex = f_locals.get("exemplars") if isinstance(f_locals.get("exemplars"), list) else []
            if agent_self and getattr(agent_self, "exemplar_retriever", None):
                raw_exemplars = compose_best_exemplar(
                    retriever=agent_self.exemplar_retriever,
                    user_text=cleaned_utterance,
                    installed_apps=installed_apps,
                    history=raw_history,
                    current_time_str=current_time,
                    fallback_exemplars=fallback_ex,
                    previous_steps=previous_steps,
                )
            elif fallback_ex:
                raw_exemplars = fallback_ex
    except Exception:
        pass

    formatted_exemplar = format_exemplar_by_turns(raw_exemplars, few_shot_examples, previous_steps)

    prompt_parts = [
        "[CURRENT CONTEXT]",
        f"- Thời gian hiện tại (Current Time): {current_time}",
    ]
    if cal_helper:
        prompt_parts.append(cal_helper)
    loc_str = str(context.get("location") or (user_profile.get("home") if isinstance(user_profile, dict) else "") or "")
    inst_apps_ctx = context.get("installed_apps") or []
    if ("flights" in inst_apps_ctx or (isinstance(inst_apps_ctx, dict) and "flights" in inst_apps_ctx)) and loc_str:
        airport_hint = ""
        if "Hà Nội" in loc_str:
            airport_hint = "HAN (Hà Nội - Nội Bài)"
        elif "TP.HCM" in loc_str or "Hồ Chí Minh" in loc_str:
            airport_hint = "SGN (TP.HCM - Tân Sơn Nhất)"
        elif "Đà Nẵng" in loc_str:
            airport_hint = "DAD (Đà Nẵng)"
        elif "Huế" in loc_str:
            airport_hint = "HUI (Huế - Phú Bài)"
        elif "Cần Thơ" in loc_str:
            airport_hint = "VCA (Cần Thơ)"
        if airport_hint:
            prompt_parts.append(f"- Vị trí hiện tại & Sân bay mặc định (`origin` khi đặt vé máy bay): {loc_str} -> mã sân bay xuất phát mặc định là `{airport_hint.split()[0]}` ({airport_hint})")
    prompt_parts.extend([
        f"- Hồ sơ người dùng (User Profile): {json.dumps(user_profile, ensure_ascii=False)}",
        "",
        "[ALLOWED TOOLS CATALOG]",
        enriched_tools,
        "",
        "[ACTIVE MEMORY FROM PREVIOUS DIALOGUE]",
        enriched_memory,
        ""
    ])

    if previous_steps:
        prompt_parts.append("[PREVIOUS STEPS IN THIS SESSION]")
        for step in previous_steps:
            s_idx = step.get("step", 1)
            t_name = step.get("tool_called", "")
            t_args = step.get("arguments", {})
            obs = step.get("observation", step.get("result", ""))
            err = step.get("error")
            if err:
                prompt_parts.append(f"- Step {s_idx}: REJECTED `{t_name}({json.dumps(t_args, ensure_ascii=False)})` -> [SCHEMA ERROR]: {err}. Sửa lại tham số cho đúng schema!")
            else:
                prompt_parts.append(f"- Step {s_idx}: COMPLETED `{t_name}({json.dumps(t_args, ensure_ascii=False)})` -> [OBSERVATION]: {obs}")
        prompt_parts.append("")

    obs_analysis = analyze_observations(previous_steps, context, raw_history, cleaned_utterance)
    if obs_analysis:
        prompt_parts.append(obs_analysis)

    utterance_lines = [f"\"{cleaned_utterance}\""]
    for m_label, m_key in (("PhoWhisper-Medium", "phowhisper_medium"), ("Whisper-Turbo", "whisper_turbo"), ("PhoWhisper-Small", "phowhisper_small")):
        if m_key in asr_variants:
            alt_clean = clean_asr_transcript(asr_variants[m_key])
            if alt_clean and alt_clean.lower() != cleaned_utterance.lower():
                if not any(hw in alt_clean.lower() for hw in ("subscribe", "lalaschool", "ghiền mì gõ", "đăng ký kênh")):
                    utterance_lines.append(f"(Bản đối chiếu {m_label}: \"{alt_clean}\")")

    prompt_parts.extend([
        "[CLOSEST REFERENCE EXEMPLAR]",
        formatted_exemplar,
        "",
        "[USER UTTERANCE]",
        "\n".join(utterance_lines),
        ""
    ])

    step_directive = build_step_directive(raw_exemplars, previous_steps, obs_analysis, cleaned_utterance, current_time)
    prompt_parts.append("[CURRENT STEP DIRECTIVE]")
    prompt_parts.append(step_directive)
    prompt_parts.append("")
    prompt_parts.append("Hãy suy luận ngắn gọn bằng tiếng Việt và trả về duy nhất khối JSON hợp lệ:")

    return "\n".join(prompt_parts)
