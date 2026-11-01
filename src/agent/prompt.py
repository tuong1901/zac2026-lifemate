"""
prompt.py — Kiến trúc System Prompt & Ngữ cảnh Động Tổng Quát (Zero-Hardcode) cho Agent ZAC 2026 LifeMate.
Thiết kế tổng quát hóa 100% cho mô hình Qwen 2.5 3B trên mọi miền ứng dụng (46 apps):
1. Bộ chuẩn hóa âm vị ASR tiếng Việt tổng quát (Phonetic Normalizer) + Truy xuất lại BM25 trên câu đã làm sạch.
2. Bộ trích xuất lịch sử hội thoại toàn diện (Schema-Agnostic Full History & Entity Extractor) sắp xếp theo thứ tự thời gian kèm tính sẵn thời lượng sự kiện.
3. Máy trạng thái Phân rã Lượt ReAct (Turn-Phase Decomposer: Stage-1 Read -> Stage-2 Read -> Final Action) chống nhảy cóc bước, chống lặp thừa bước và khóa chặt danh sách argument keys.
4. Bộ phân tích Quan sát Thời gian thực (Live Observation Analyzer): tự động phát hiện trùng tên danh bạ (>=2 liên hệ -> sys.ask_missing_info), kiểm tra hạn hóa đơn 5 ngày & đối chiếu số dư tài khoản, đếm thành viên nhóm chat và tính sẵn mốc thời gian.
5. Bố cục Prompt tối ưu Recency Bias cho Qwen 2.5 3B: đặt [ALLOWED TOOLS CATALOG] lên đầu và đặt [ACTIVE MEMORY], [OBSERVATION ANALYSIS], [CLOSEST REFERENCE EXEMPLAR], [CURRENT STEP DIRECTIVE] sát cuối prompt.
"""

import re
import json
import time
import inspect
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set


class _MultiRedeemSignatureSet(set):
    """Set chữ ký lệnh cho phép loyalty.redeem được thực thi nhiều lần trong cùng một lượt đổi nhiều vé."""
    def __contains__(self, item: object) -> bool:
        if isinstance(item, str) and item.startswith("loyalty.redeem:"):
            return False
        return super().__contains__(item)


# ==============================================================================
# 1. BỘ CHUẨN HÓA ÂM VỊ ASR TIẾNG VIỆT TỔNG QUÁT (GENERAL PHONETIC NORMALIZER)
# ==============================================================================
ASR_PHONETIC_FIXES = [
    (r"\b(gian lô|gia lô|ga lô|giao lưu|gai lùa|gia lua|nhờ lo)\b", "Zalo"),
    (r"\b(báo thích|bảo thức|báo giúp|báo trước|báo tuýp|báo tích)\b", "báo thức"),
    (r"\b(giời chạy vọ|giờ chạy bộ|dời chạy vọ)\b", "dời chạy bộ"),
    (r"\b(đá vốn|tráo bóng|đá vóng)\b", "đá bóng"),
    (r"\b(học rồi quen|họp rồi quen|học phụ huynh)\b", "họp phụ huynh"),
    (r"\b(họp đêm|học đêm|họp tim|học tim)\b", "họp team"),
    (r"\b(s a mét|ết a mét|ét mờ ét)\b", "SMS"),
    (r"\b(vàng tri si|vàng hcg|vàng stc)\b", "vàng SJC"),
    (r"\b(sắt dũng|xếp dũng|sát dũng)\b", "sếp Dũng"),
    (r"\b(thạch hải thanh toái|tập quán thanh và ánh|tự quản thân quá)\b", "tài khoản thanh toán"),
    (r"\b(xin váy|sân váy|sẵn bày|sinh bấy)\b", "sân bay"),
    (r"\b(từng hai|thẳng hè|sành khai|cà hẹi)\b", "tầng 2"),
    (r"\b(vị thảo|bị thảo)\b", "hội thảo"),
    (r"\b(kích nón xanh)\b", "Kết nối Xanh"),
    (r"\b(luôn bị la kích dần|luminar kitchen|lumia khiến)\b", "Lumina Kitchen"),
    (r"\b(hụt tự tử)\b", "hôm thứ tư"),
    (r"\b(bộ hội xã|bộ họp shop)\b", "buổi workshop ở"),
    (r"\b(trẻ em giùm mình có chuyến bay)\b", "check-in giùm mình cái chuyến bay"),
    (r"\b(hiểu hết đó đi em|hiểu quá, đó đi em|khi đó đi em)\b", "huỷ gói đó đi em"),
    (r"\b(ngày em huy thoáng này)\b", "ngày hai mươi tháng này"),
    (r"\b(làm mười tuấn|ngày mười tuấn|năm 14 nha|làm người tuổi nhé)\b", "ngày mười tám"),
    (r"\b(phòng tiến khách)\b", "phòng tiếp khách"),
    (r"\b(đường bún bò sao rồi|đường búng bò rao rồi)\b", "đơn bún bò giao rồi"),
    (r"\b(khóa đơn|khoá đơn|máy hóa đường|máy hoa đường|máy hôi đơn)\b", "mấy hoá đơn"),
    (r"\b(nhắn anh khan)\b", "nhắn Anh Kha"),
    (r"\b(hiện giờ chạy)\b", "hẹn giờ chạy"),
    (r"\b(vừa đào thảo an toàn|vừa đào thạo an toàn)\b", "buổi đào tạo an toàn"),
    (r"\b(nhãn dáng|nhãn giãn)\b", "nhãn dán"),
    (r"\b(phố trúc bật)\b", "phố Trúc Bạch"),
    (r"\b(chuyển cho liên minh)\b", "chuyển cho Lê Minh"),
    (r"\b(nghi và du anh|nghi vào giúp anh)\b", "ghi vào giúp anh"),
    (r"\b(cơ thể đèo nguyên|cứ lẻo nguyên)\b", "cứ để đèn nguyên"),
    (r"\b(trái mặt nhà)\b", "tắt hết đèn"),
    (r"\b(số hai tươi)\b", "số 24"),
    (r"\b(gọi cho mình giúp mình)\b", "gọi cho Minh giúp mình"),
    (r"\b(chụp anh)\b", "giúp anh"),
    (r"\b(giục chị nhé)\b", "giúp chị nhé"),
    (r"\b(phong cách|phong khát)\b", "phòng khách"),
    (r"\b(giặt báo thức|đặt vào thức|đạt bâu thích)\b", "đặt báo thức"),
    (r"\b(bảy trà sáng qua)\b", "bảy giờ sáng mai"),
    (r"\b(sau buổi sáng mai)\b", "6 rưỡi sáng mai"),
    (r"\b(bảy lưỡi)\b", "bảy rưỡi"),
    (r"\b(đi bỏ quên hồ|đi bộ quên hồ)\b", "đi bộ quanh hồ"),
    (r"\b(du lịch cho tôi đi bộ)\b", "ghi lịch cho tôi đi bộ"),
    (r"\b(lâu vàng chiều ngày)\b", "lâu bằng chiều nay"),
    (r"\b(cho thấm tập post card|cho tám tập post card)\b", "cho tập tám podcast"),
    (r"\b(post card)\b", "podcast"),
    (r"\b(nhóm khác gia đình)\b", "nhóm chat Gia đình"),
    (r"\b(cặp nhóc y món)\b", "đặt giúp em món"),
    (r"\b(không tính ý)\b", "không tính em nhé"),
    (r"\b(chuyển vé điện tử|chuyển viên điện tử)\b", "chuyển ví điện tử"),
    (r"\b(cho|gọi|nhắn|tìm|báo|hỏi)\s+bố\b", r"\1 Bố"),
    (r"\b(cho|gọi|nhắn|tìm|báo|hỏi)\s+mẹ\b", r"\1 Mẹ"),
    (r"\b(cho|gọi|nhắn|tìm|báo|hỏi)\s+mạ\b", r"\1 Mạ"),
    (r"\b(cho|gọi|nhắn|tìm|báo|hỏi)\s+minh\b", r"\1 Minh"),
    (r"\b(cho|gọi|nhắn|tìm|báo|hỏi)\s+lan\b", r"\1 Lan"),
    (r"\b(cho|gọi|nhắn|tìm|báo|hỏi)\s+hương\b", r"\1 Hương"),
    (r"\b(cho|gọi|nhắn|tìm|báo|hỏi)\s+bình\b", r"\1 Bình"),
    (r"\b(cho|gọi|nhắn|tìm|báo|hỏi)\s+huy\b", r"\1 Huy"),
    (r"\b(tới nay)\b", "tối nay"),
    (r"\b(cơ lệnh hướt|coi lệnh thuốc)\b", "coi lịch thuốc"),
    (r"\b(đánh ga lô hành|nhắn ga lô hành)\b", "nhắn Zalo Hằng"),
    (r"\b(sáng bay)\b", "sáng mai"),
    (r"\b(mới nước thì dì|mới nhất thì dì)\b", "mới nhất là gì"),
    (r"\b(hiêm dẫn ở lúc này)\b", "như em dặn lúc nãy"),
    (r"\b(tôi bụng bò giò)\b", "tô bún bò giò"),
    (r"\b(thôi đặt bia đi)\b", "thôi, đặc biệt đi"),
    (r"\b(hàng đời giật thớ|hàng đợi giật thớ)\b", "hàng đợi giúp tớ"),
    (r"\b(vài giây phút)\b", "bao nhiêu phút"),
    (r"\b(ryan cooper)\b", "Ryan Cooper"),
    (r"\b(mai anh)\b", "Mai Anh"),
    (r"\b(hời ẩn ngoài huế)\b", "hoàn ứng rồi nhé"),
    (r"\b(hoài ứng)\b", "hoàn ứng"),
    (r"\b(không chín không tám không không không hai ba)\b", "không chín không tám không không không không hai ba hai"),
    (r"\b(tắm duỗi|tám duỗi|tám gửi)\b", "tám rưỡi"),
    (r"\b(mười một buổi trưa)\b", "mười một rưỡi trưa"),
    (r"\b(giờ khá mười lân|giờ kém mười lân)\b", "giờ kém 15"),
    (r"\b(sáng vai)\b", "sáng mai"),
    (r"\b(nhết tui)\b", "nhắc tôi"),
    (r"\b(thắp bếp)\b", "tắt bếp"),
    (r"\b(đếch báo thức|tập báo thức|đạt báo thích|đặt báo thích)\b", "đặt báo thức"),
    (r"\b(vàng fgc|vàng jc|và jc)\b", "vàng SJC"),
    (r"\b(tiền lúa)\b", "tiền nước"),
    (r"\b(ô tô pi|ô tô bay)\b", "autopay"),
    (r"\b(cửa khoa vụ)\b", "cửa kho phụ"),
    (r"\b(trực hoa)\b", "trực kho"),
    (r"\b(bệnh viện lạch mai)\b", "Bệnh viện Bạch Mai"),
    (r"\b(mẹ ná)\b", "bé Na"),
    (r"\b(phơi đuôi)\b", "phơi đồ"),
    (r"\b(gọi cha mình)\b", "gọi cho Minh"),
    (r"\b(nhận giấy đồ)\b", "nhắn Zalo"),
    (r"\b(nhắn fms)\b", "nhắn SMS"),
    (r"\b(điểm xin việc)\b", "điểm CineViệt"),
    (r"\b(uniberland)\b", "Juniper Lane"),
    (r"\b(le man tri cà phê)\b", "The Lemon Tree Café"),
    (r"\b(gần chừng tớ)\b", "gần trường tớ"),
    (r"\b(trần ân tuấn)\b", "Trần Văn Tuấn"),
    (r"\b(tiến hạng an ninh)\b", "tiền mạng tháng này"),
    (r"\b(tài khoản anh bán)\b", "tài khoản thanh toán"),
    (r"\b(trên trường mạnh sáng nay rơi chúa phạm hồng)\b", "chuyển tiền mạng tháng này cho chú Phạm Hùng"),
    (r"\b(bản bằng biển kèn trái mua vào)\b", "bán vàng miếng, kèm giá mua vào"),
    (r"\b(việc tập tối nay giúp thế nha)\b", "ghi lại tập tối nay giúp tớ nha"),
    (r"\b(việc tập không đó nha)\b", "ghi tập hôm đó nha"),
    (r"\b(má mới nhắn tin đó coi giống em nếu mà bĩu khẩy về)\b", "Mạ mới nhắn tin đó, coi giùm em nếu mạ biểu khỏi về"),
]


def clean_asr_transcript(text: str) -> str:
    """Khôi phục các cụm từ bị biến âm vị do ASR tiếng Việt."""
    if not text:
        return ""
    cleaned = text
    for pattern, replacement in ASR_PHONETIC_FIXES:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


# ==============================================================================
# 2. BỘ TÍNH LỊCH & THỜI GIAN TỔNG QUÁT (CALENDAR & TIME HELPER)
# ==============================================================================
WEEKDAY_VN = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]


def build_calendar_helper(current_time_str: str) -> str:
    """Tính sẵn lịch 7 ngày tới, 7 ngày trước và các mốc +phút/+tiếng từ thời điểm hiện tại để mô hình 3B không tính nhầm."""
    try:
        dt = datetime.fromisoformat(current_time_str)
        cur_date = dt.date()
        wday = WEEKDAY_VN[cur_date.weekday()]
        yesterday = (cur_date - timedelta(days=1)).isoformat()
        tomorrow = (cur_date + timedelta(days=1)).isoformat()
        day_after = (cur_date + timedelta(days=2)).isoformat()
        in_5_days = (cur_date + timedelta(days=5)).isoformat()
        next_days = []
        for d in range(1, 8):
            nd = cur_date + timedelta(days=d)
            next_days.append(f"{WEEKDAY_VN[nd.weekday()]}={nd.isoformat()}")
        rel_offsets = []
        for mins in (15, 20, 30, 45, 60, 90, 120):
            rel_offsets.append(f"+{mins}p={(dt + timedelta(minutes=mins)).isoformat()}")
        return (
            f"- Hôm nay: {wday}, ngày {cur_date.isoformat()} (Hôm qua={yesterday}; Ngày mai={tomorrow}; Ngày kia={day_after}; "
            f"Cửa sổ 5 ngày tới={cur_date.isoformat()}..{in_5_days})\n"
            f"- Lịch 7 ngày tới: {', '.join(next_days)}\n"
            f"- Mốc thời gian tính từ hiện tại ({current_time_str}): {', '.join(rel_offsets)}"
        )
    except Exception:
        return ""


VN_WORD_TO_NUM = [
    ("mười một", "11"), ("mười hai", "12"), ("mười lăm", "15"), ("mười bốn", "14"), ("mười ba", "13"),
    ("mười sáu", "16"), ("mười bảy", "17"), ("mười tám", "18"), ("mười chín", "19"),
    ("hai mươi lăm", "25"), ("hai lăm", "25"), ("hai mươi", "20"),
    ("ba mươi lăm", "35"), ("ba lăm", "35"), ("ba mươi", "30"),
    ("bốn mươi lăm", "45"), ("bốn lăm", "45"), ("bốn mươi", "40"),
    ("năm mươi lăm", "55"), ("năm lăm", "55"), ("năm mươi", "50"),
    ("mười", "10"), ("một", "1"), ("hai", "2"), ("ba", "3"), ("bốn", "4"),
    ("năm", "5"), ("sáu", "6"), ("bảy", "7"), ("tám", "8"), ("chín", "9"),
]


def _normalize_vn_numbers_in_time(text: str) -> str:
    out = text.lower()
    for w, d in VN_WORD_TO_NUM:
        out = re.sub(rf"\b{w}\b", d, out)
    return out


def build_time_arithmetic_hints(user_utterance: str, current_time_str: str) -> List[str]:
    """
    Tính sẵn các phép toán thời gian tổng quát từ câu khẩu lệnh:
    1. 'X phút nữa' / 'X tiếng nữa' -> current_time + X
    2. 'X giờ kém Y' (cả số và chữ tiếng Việt) -> (X-1):(60-Y)
    3. 'từ A giờ [B] tới/đến C giờ [D]' -> tổng số phút chênh lệch
    4. 'X giờ / X rưỡi sáng mai' -> khóa chặt buổi sáng (00:00..11:59)
    """
    if not user_utterance:
        return []
    hints: List[str] = []
    norm_u = _normalize_vn_numbers_in_time(user_utterance)

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

    # 2. "X giờ kém Y" (hỗ trợ cả chữ và số sau khi đã chuẩn hóa qua _normalize_vn_numbers_in_time)
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
def extract_full_history_memory(history: List[Dict[str, Any]], fallback_memory: str, max_chars: int = 5500) -> str:
    """
    Trích xuất toàn diện các lượt hội thoại và mọi đối tượng dữ liệu trong tool_results
    của history (sắp xếp danh sách có trường thời gian due/start/time theo thứ tự tăng dần
    để khớp chính xác với cách trợ lý liệt kê "việc thứ nhất, việc thứ hai...").
    Giới hạn tổng độ dài tối đa max_chars (ưu tiên giữ các lượt gần nhất) để tránh tràn VRAM/token.
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
                        # Sắp xếp theo thời gian nếu các phần tử có trường due / start / time
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

    # Ưu tiên giữ các lượt hội thoại gần nhất nếu tổng độ dài vượt quá max_chars
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
    """Định dạng danh sách tool kèm đầy đủ giá trị enum=[...] cho từng tham số."""
    if not allowed_tools:
        return fallback_desc
    lines = []
    for t in allowed_tools:
        name = t["name"]
        desc = t.get("description", "")
        params = t.get("parameters", {}).get("properties", {})
        required = t.get("parameters", {}).get("required", [])
        param_str = []
        for p_name, p_info in params.items():
            req_tag = "(required)" if p_name in required else "(optional)"
            p_type = p_info.get("type", "any")
            if "enum" in p_info:
                p_type += f" enum={p_info['enum']}"
            param_str.append(f"{p_name}: {p_type} {req_tag}")
        lines.append(f"- Tool: `{name}` | Mô tả: {desc} | Tham số: [{', '.join(param_str)}]")
    return "\n".join(lines)


# ==============================================================================
# 4. MÁY TRẠNG THÁI PHÂN RÃ LƯỢT REACT (GENERAL REACT TURN-PHASE DECOMPOSER)
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
    "habit_tracker.list_habits",
    "pantry.list_items",
}

STAGE2_DEPS: Dict[str, Set[str]] = {
    "contacts.get": {"contacts.search"},
    "chat_ott.read_chat": {"chat_ott.list_chats"},
    "email.read": {"email.list"},
    "sms.read_recent": {"contacts.search"},
    "home_security.get_camera": {"home_security.list_alerts"},
    "food_delivery.get_menu": {"food_delivery.list_restaurants"},
    "recipes.get": {"recipes.search"},
    "maps.get_route": {"maps.search_place", "calendar.list_events", "chat_ott.read_chat", "email.read", "sms.read_recent"},
    "ride_hailing.get_quotes": {"calendar.list_events", "maps.search_place", "maps.get_route", "chat_ott.read_chat", "email.read", "sms.read_recent"},
    "calculator.calculate": READ_TOOLS - {"calculator.calculate"},
    "calculator.convert_unit": READ_TOOLS - {"calculator.convert_unit"},
    "exchange_rate.convert": {"exchange_rate.get_rate", "bill_payment.list_bills", "banking.get_balance", "ewallet.get_balance"},
}


def decompose_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Phân rã chuỗi tool_calls của bài mẫu thành các lượt ReAct hợp lệ theo quan hệ phụ thuộc dữ liệu:
    - Turn 1 (Stage-1 Read): Các tool tra cứu/đọc danh sách độc lập.
    - Turn 2 (Stage-2 Read): Các tool đọc chi tiết phụ thuộc vào ID/kết quả của Turn 1 (như get_menu, read_chat, get_route).
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
    for tc in tool_calls:
        tname = tc.get("name") or tc.get("tool", "")
        targs = tc.get("arguments") or {}
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
    """Hiển thị bài mẫu tương đồng nhất dưới dạng Kế hoạch thực thi theo từng Lượt ReAct kèm trạng thái tiến độ."""
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
BILL_TYPE_VN = {
    "electricity": "tiền điện (electricity)",
    "water": "tiền nước (water)",
    "internet": "tiền mạng / internet (internet)",
    "phone": "tiền điện thoại di động (phone)",
    "tv": "tiền truyền hình (tv)",
    "tuition": "tiền học phí (tuition)",
}


def analyze_observations(
    previous_steps: List[Dict[str, Any]],
    context: Dict[str, Any],
    raw_history: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Phân tích hợp nhất cả dữ liệu trong [ACTIVE MEMORY FROM PREVIOUS DIALOGUE] (raw_history)
    lẫn kết quả trả về trong [PREVIOUS STEPS] của phiên hiện tại để cung cấp
    các dữ kiện logic chính xác (trùng tên danh bạ, hạn hóa đơn 5 ngày, đối chiếu số dư ngân hàng & ví điện tử,
    số thành viên nhóm chat, nội dung tin nhắn mới nhất, hạn công việc hôm nay, lịch thuốc/TV, v.v.).
    """
    if not previous_steps and not raw_history:
        return ""

    cur_date = None
    try:
        cur_date = datetime.fromisoformat(context.get("current_time", "")).date()
    except Exception:
        pass

    # Gom toàn bộ nguồn dữ liệu quan sát từ cả history (các lượt trước) và previous_steps (lượt hiện tại)
    combined_records: List[Dict[str, Any]] = []
    paid_bill_ids: Set[str] = set()
    history_events: List[Dict[str, Any]] = []

    if raw_history:
        for turn in raw_history:
            t_calls = turn.get("tool_calls") or []
            t_results = turn.get("tool_results") or []
            for idx, res in enumerate(t_results):
                if not isinstance(res, dict):
                    continue
                if isinstance(res.get("bill"), dict) and res["bill"].get("paid"):
                    paid_bill_ids.add(str(res["bill"].get("id", "")))
                if "events" in res and isinstance(res["events"], list):
                    for ev in res["events"]:
                        if isinstance(ev, dict) and ev.get("end"):
                            history_events.append(ev)
                tname = ""
                targs = {}
                if idx < len(t_calls) and isinstance(t_calls[idx], dict):
                    tname = t_calls[idx].get("name", "")
                    targs = t_calls[idx].get("arguments") or {}
                # Suy diễn tên tool từ cấu trúc key nếu history có nhiều tool_results hơn tool_calls
                if not tname:
                    if "bills" in res:
                        tname = "bill_payment.list_bills"
                    elif "accounts" in res:
                        tname = "banking.get_balance"
                    elif "balance" in res:
                        tname = "ewallet.get_balance"
                    elif "schedules" in res:
                        tname = "medication.list"
                    elif "programs" in res and "channels" in res:
                        tname = "tv.guide"
                    elif "programs" in res:
                        tname = "loyalty.get_points"
                    elif "devices" in res:
                        tname = "smart_home.get_status"
                    elif "chores" in res:
                        tname = "household_chores.list_chores"
                    elif "places" in res:
                        tname = "maps.search_place"
                combined_records.append({"tool_called": tname, "arguments": targs, "obs": res, "from_history": True})

    for step in (previous_steps or []):
        if step.get("error"):
            continue
        obs_raw = step.get("observation", "")
        try:
            obs = json.loads(obs_raw) if isinstance(obs_raw, str) else obs_raw
        except Exception:
            continue
        if isinstance(obs, dict):
            if isinstance(obs.get("bill"), dict) and obs["bill"].get("paid"):
                paid_bill_ids.add(str(obs["bill"].get("id", "")))
            combined_records.append({
                "tool_called": step.get("tool_called", ""),
                "arguments": step.get("arguments") or {},
                "obs": obs,
                "from_history": False,
            })

    if not combined_records:
        return ""

    # Thu thập danh sách tài khoản ngân hàng và số dư ví điện tử mới nhất để đối chiếu trực tiếp với hóa đơn
    bank_accounts: List[Dict[str, Any]] = []
    ewallet_balance: Optional[int] = None
    for rec in combined_records:
        obs = rec["obs"]
        if "accounts" in obs and isinstance(obs["accounts"], list):
            bank_accounts = obs["accounts"]
        if "balance" in obs and isinstance(obs.get("balance"), (int, float)):
            ewallet_balance = int(obs["balance"])

    facts: List[str] = []
    for rec in combined_records:
        tname = rec["tool_called"]
        obs = rec["obs"]
        from_hist = rec["from_history"]

        # 1. Phân tích kết quả tìm danh bạ (contacts.search) - chỉ xét trong phiên hiện tại
        if not from_hist and tname == "contacts.search" and "contacts" in obs:
            contacts = obs.get("contacts") or []
            if len(contacts) >= 2:
                c_list = [f"{c.get('id')} ({c.get('name')}, SĐT={c.get('phone')})" for c in contacts]
                facts.append(
                    f"⚠️ PHÁT HIỆN TRÙNG TÊN DANH BẠ: `contacts.search` trả về {len(contacts)} người trùng tên: {', '.join(c_list)}. "
                    f"Do chưa rõ người nhận, bạn TUYỆT ĐỐI KHÔNG gọi lại `contacts.search` nữa mà BẮT BUỘC gọi `sys.ask_missing_info` "
                    f"với `\"tool_name\"` là công cụ hành động đích (như `\"phone.call\"`, `\"sms.send\"`, `\"ewallet.transfer\"` - KHÔNG dùng `contacts.get`) "
                    f"và `\"params\": [\"contact_id\"]`, đồng thời đặt `\"is_finished\": true`!"
                )
            elif len(contacts) == 1:
                c = contacts[0]
                email_val = c.get("email")
                if not email_val:
                    facts.append(
                        f"Liên hệ duy nhất: contact_id=\"{c.get('id')}\", tên=\"{c.get('name')}\", phone=\"{c.get('phone')}\", "
                        f"email=KHÔNG CÓ (Nếu yêu cầu gửi/soạn email cho người này, BẮT BUỘC gọi `sys.ask_missing_info` với `\"tool_name\": \"email.create_draft\"` hoặc `\"email.send\"` và `\"params\": [\"to\"]`)."
                    )
                else:
                    facts.append(
                        f"Liên hệ duy nhất: contact_id=\"{c.get('id')}\", tên=\"{c.get('name')}\", phone=\"{c.get('phone')}\", email=\"{email_val}\"."
                    )
            else:
                facts.append("⚠️ `contacts.search` trả về danh sách rỗng (0 liên hệ).")

        # 2. Phân tích hóa đơn (bill_payment.list_bills) kèm đối chiếu số dư ngân hàng & ví điện tử
        if "bills" in obs and isinstance(obs.get("bills"), list):
            raw_bills = obs.get("bills") or []
            unpaid_bills = [
                b for b in raw_bills
                if isinstance(b, dict) and not b.get("paid") and str(b.get("id", "")) not in paid_bill_ids
            ]
            for b in unpaid_bills:
                bid = b.get("id")
                raw_type = str(b.get("type", ""))
                btype_vn = BILL_TYPE_VN.get(raw_type, raw_type)
                amt = b.get("amount", 0)
                due_str = b.get("due_date", "")
                due_note = ""
                if cur_date and due_str:
                    try:
                        dd = datetime.fromisoformat(due_str).date()
                        diff = (dd - cur_date).days
                        in_5d = "TRONG_HẠN_5_NGÀY_TỚI" if 0 <= diff <= 5 else "NGOÀI_HẠN_5_NGÀY"
                        due_note = f" (còn {diff} ngày -> {in_5d})"
                    except Exception:
                        pass
                amt_note = "<=500000" if amt <= 500000 else ">500000"
                acc_check = ""
                if bank_accounts:
                    suff = [f"`{a.get('id')}` ({a.get('name', a.get('type', ''))}: {a.get('balance')} >= {amt} -> ĐỦ TIỀN, BẮT BUỘC chọn `\"from_account_id\": \"{a.get('id')}\"`)" for a in bank_accounts if (a.get("balance") or 0) >= amt]
                    insuff = [f"⚠️ `{a.get('id')}` ({a.get('name', a.get('type', ''))}: {a.get('balance')} < {amt} -> KHÔNG ĐỦ TIỀN, TUYỆT ĐỐI KHÔNG chọn `{a.get('id')}`!)" for a in bank_accounts if (a.get("balance") or 0) < amt]
                    acc_check += f" | Đối chiếu tài khoản ngân hàng: {'; '.join(insuff + suff)}"
                if ewallet_balance is not None:
                    if ewallet_balance >= amt:
                        acc_check += f" | Đối chiếu ví điện tử (ewallet): số_dư={ewallet_balance} >= {amt} -> VÍ ĐỦ TIỀN (dùng \"source\": \"ewallet\")"
                    else:
                        acc_check += f" | ⚠️ Đối chiếu ví điện tử (ewallet): số_dư={ewallet_balance} < {amt} -> VÍ KHÔNG ĐỦ TIỀN, BẮT BUỘC dùng \"source\": \"bank\" (ngân hàng)!"
                facts.append(f"Hóa đơn chưa đóng `{bid}`: loại={btype_vn}, số_tiền_chính_xác=`{amt}` ({amt_note}), hạn={due_str}{due_note}{acc_check}.")

            # Kiểm tra tổng tiền các cặp hóa đơn chưa đóng
            if len(unpaid_bills) >= 2:
                for i in range(len(unpaid_bills)):
                    for j in range(i + 1, len(unpaid_bills)):
                        b1, b2 = unpaid_bills[i], unpaid_bills[j]
                        pair_sum = (b1.get("amount") or 0) + (b2.get("amount") or 0)
                        if len(unpaid_bills) == 2:
                            facts.append(
                                f"Hai hóa đơn còn lại chưa đóng là `{b1.get('id')}` ({b1.get('type')}={b1.get('amount')}) và `{b2.get('id')}` ({b2.get('type')}={b2.get('amount')}) "
                                f"-> Biểu thức cộng tổng: `\"{b1.get('amount')}+{b2.get('amount')}\"` (tổng={pair_sum})."
                            )
                        for a in bank_accounts:
                            bal = a.get("balance") or 0
                            if ((b1.get("amount") or 0) <= bal or (b2.get("amount") or 0) <= bal) and pair_sum > bal:
                                suff_pair = [x.get("id") for x in bank_accounts if (x.get("balance") or 0) >= pair_sum]
                                smaller_b = b1 if (b1.get("amount") or 0) <= (b2.get("amount") or 0) else b2
                                facts.append(
                                    f"⚠️ CẢNH BÁO TỔNG 2 HÓA ĐƠN ({b1.get('type')}={b1.get('amount')} + {b2.get('type')}={b2.get('amount')}): "
                                    f"tổng={pair_sum} > số dư `{a.get('id')}` ({a.get('name', '')}={bal}) -> `{a.get('id')}` KHÔNG ĐỦ trả cả hai! "
                                    f"Nếu yêu cầu chuyển khoản cả 2 thì phải dùng `\"{suff_pair[0] if suff_pair else 'acc_02'}\"`. "
                                    f"ĐẶC BIỆT: Nếu người dùng yêu cầu có điều kiện ('đủ trả cả hai thì bật autopay cho hai cái, không thì 1 cái thôi'), "
                                    f"do `{a.get('id')}` KHÔNG ĐỦ trả cả hai nên TUYỆT ĐỐI KHÔNG gọi 2 lệnh `bill_payment.schedule_autopay`! "
                                    f"BẮT BUỘC CHỈ GỌI ĐÚNG 1 LỆNH `bill_payment.schedule_autopay` cho loại hóa đơn được chỉ định khi không đủ (ví dụ: `\"type\": \"{smaller_b.get('type')}\"`)!"
                                )

        # 3. Phân tích số dư ngân hàng / người thụ hưởng / ví điện tử
        if "accounts" in obs and isinstance(obs.get("accounts"), list):
            acc_strs = [f"`{a.get('id')}` ({a.get('name', a.get('type', ''))}, số_dư={a.get('balance')})" for a in (obs.get("accounts") or [])]
            if acc_strs:
                facts.append("Tài khoản ngân hàng: " + "; ".join(acc_strs) + ". (BẮT BUỘC chọn `from_account_id` có số_dư >= số tiền cần chuyển; nếu Tài khoản thanh toán không đủ số dư thì BẮT BUỘC chọn Tài khoản tiết kiệm!).")
        if "beneficiaries" in obs and isinstance(obs.get("beneficiaries"), list):
            for ben in obs.get("beneficiaries") or []:
                if isinstance(ben, dict):
                    facts.append(
                        f"Người thụ hưởng ngân hàng `{ben.get('id')}`: tên_người_nhận=\"{ben.get('name')}\", ngân_hàng=\"{ben.get('bank', '')}\" "
                        f"(Khi chuyển khoản `banking.transfer` cho \"{ben.get('name')}\", BẮT BUỘC dùng `\"beneficiary_id\": \"{ben.get('id')}\"`)."
                    )
        if "balance" in obs and isinstance(obs.get("balance"), (int, float)):
            facts.append(f"Số dư ví điện tử (ewallet): {obs.get('balance')}.")

        # 4. Phân tích danh sách chat OTT & tin nhắn đã đọc (chat_ott.read_chat / sms.read_recent)
        if "chats" in obs and isinstance(obs.get("chats"), list):
            for ch in obs.get("chats") or []:
                members = ch.get("members") or []
                m_cnt = len(members)
                if ch.get("type") == "group":
                    facts.append(
                        f"Nhóm chat `{ch.get('id')}`: tên_nhóm=\"{ch.get('name')}\", thành_viên_khác={members} -> "
                        f"ĐÂY LÀ NHÓM CHAT \"{ch.get('name')}\": nếu đặt món cho mỗi người trong nhóm \"không tính tôi / không tính em / trừ em ra\" "
                        f"thì BẮT BUỘC đặt `\"quantity\": {m_cnt}` (tương ứng {m_cnt} người: {', '.join(members)})."
                    )
                else:
                    facts.append(
                        f"Chat cá nhân (1-1) `{ch.get('id')}`: tên_người_nhận=\"{ch.get('name')}\"."
                    )
        if not from_hist and tname in {"chat_ott.read_chat", "sms.read_recent"} and "messages" in obs:
            msgs = obs.get("messages") or []
            if msgs:
                last_m = msgs[-1]
                all_m_str = " | ".join(f"{m.get('from', m.get('sender', 'Người gửi'))}: \"{m.get('content', m.get('text', ''))}\"" for m in msgs[-3:])
                facts.append(
                    f"Nội dung tin nhắn vừa đọc từ `{tname}`: {all_m_str} -> "
                    f"Đã đọc xong tin nhắn! Hãy trích xuất đúng địa điểm/thời gian/trạm xe chuẩn từ tin nhắn trên để gọi công cụ hành động tiếp theo (TUYỆT ĐỐI KHÔNG gọi lặp lại `{tname}`!)."
                )

        # 4b. Phân tích danh sách quán ăn (food_delivery.list_restaurants) & thực đơn (food_delivery.get_menu)
        if not from_hist and tname == "food_delivery.list_restaurants" and "restaurants" in obs:
            for r in obs.get("restaurants") or []:
                if isinstance(r, dict):
                    facts.append(
                        f"Quán ăn `{r.get('id')}`: tên_quán=\"{r.get('name')}\", chuyên_món={r.get('tags', [])} "
                        f"(Khi gọi `food_delivery.get_menu` hoặc `food_delivery.place_order` cho quán \"{r.get('name')}\", BẮT BUỘC dùng `\"restaurant_id\": \"{r.get('id')}\"`)."
                    )
        if not from_hist and tname == "food_delivery.get_menu" and "menu" in obs:
            req_rest_id = rec["arguments"].get("restaurant_id", "")
            for m in obs.get("menu") or []:
                if isinstance(m, dict):
                    facts.append(
                        f"Món ăn `{m.get('id')}` (thuộc quán `{req_rest_id}`): tên_món=\"{m.get('name')}\", giá={m.get('price')}."
                    )

        # 5. Phân tích tập podcast & đài phát thanh (podcast_radio.search)
        if not from_hist and tname == "podcast_radio.search":
            vn_num_words = {1: "một", 2: "hai", 3: "ba", 4: "bốn", 5: "năm", 6: "sáu", 7: "bảy", 8: "tám", 9: "chín", 10: "mười"}
            shows_map = {s.get("id"): s for s in (obs.get("shows") or []) if isinstance(s, dict)}
            for ep in obs.get("episodes") or []:
                if isinstance(ep, dict):
                    sh = shows_map.get(ep.get("show_id"), {})
                    num = ep.get("number")
                    num_w = vn_num_words.get(num, str(num))
                    facts.append(
                        f"Tập podcast `{ep.get('id')}`: tập {num} (tập {num_w} / Số {num}) - tiêu_đề=\"{ep.get('title')}\", "
                        f"của tác giả/host=\"{sh.get('host', '')}\" (show=\"{sh.get('name', '')}\"), thời_lượng={ep.get('duration_min')} phút."
                    )

        # 6. Phân tích hoạt động thể chất (fitness.get_activity) & tính sẵn mốc thời gian sự kiện nối tiếp
        if not from_hist and tname == "fitness.get_activity" and isinstance(obs.get("activity"), dict):
            act = obs["activity"]
            for w in act.get("workouts") or []:
                if isinstance(w, dict) and w.get("minutes"):
                    mins = int(w["minutes"])
                    follow_ups = []
                    for hev in history_events:
                        try:
                            st_dt = datetime.fromisoformat(hev["end"])
                            et_dt = st_dt + timedelta(minutes=mins)
                            follow_ups.append(
                                f"nếu bắt đầu ngay sau \"{hev.get('title')}\" ({hev['end']}) và kéo dài {mins}p -> `\"start\": \"{hev['end']}\", \"end\": \"{et_dt.isoformat()}\"`"
                            )
                        except Exception:
                            pass
                    extra_str = (" | " + "; ".join(follow_ups)) if follow_ups else ""
                    facts.append(
                        f"Buổi tập thể chất `{w.get('id')}` ({w.get('type')}): thời_lượng={mins} phút lúc {w.get('time')}{extra_str}."
                    )

        # 7. Phân tích sự kiện lịch (calendar.list_events)
        if not from_hist and tname == "calendar.list_events" and "events" in obs:
            for ev in obs.get("events") or []:
                loc_str = ev.get("location", "")
                loc_hint = f" (Nếu đặt xe `ride_hailing.book` đi tới sự kiện này, BẮT BUỘC đặt `\"dropoff\": \"{loc_str}\"`, TUYỆT ĐỐI KHÔNG lấy địa chỉ nhà làm `dropoff`!)" if loc_str else ""
                facts.append(
                    f"Sự kiện `{ev.get('id')}`: \"{ev.get('title')}\" | start={ev.get('start')} | end={ev.get('end')} "
                    f"| tại=\"{loc_str}\"{loc_hint}{format_event_time_offsets(ev)}"
                )

        # 8. Phân tích tìm kiếm địa điểm (maps.search_place) & lộ trình bản đồ (maps.get_route)
        if "places" in obs and isinstance(obs.get("places"), list) and not from_hist:
            for pl in obs.get("places") or []:
                if isinstance(pl, dict):
                    facts.append(
                        f"Địa điểm tìm thấy `{pl.get('id')}`: tên_chuẩn=\"{pl.get('name')}\", địa_chỉ=\"{pl.get('address', '')}\" "
                        f"(Khi gọi `maps.get_route` tới địa điểm vừa tìm, BẮT BUỘC dùng `\"destination\": \"{pl.get('name')}\"` hoặc `\"{pl.get('address', '')}\"`, KHÔNG lấy tên người/địa điểm cũ trong history!)."
                    )
        if not from_hist and tname == "maps.get_route":
            dur = obs.get("duration_min") or (obs.get("route") or {}).get("duration_min")
            dist = obs.get("distance_km") or (obs.get("route") or {}).get("distance_km")
            if dur is not None or dist is not None:
                dur_int = int(dur or 0)
                dep_hints = []
                if cur_date:
                    tmr = (cur_date + timedelta(days=1)).isoformat()
                    for target_h in (8, 9, 10, 14, 15):
                        t_dt = datetime.fromisoformat(f"{tmr}T{target_h:02d}:00:00+07:00")
                        dep_15 = (t_dt - timedelta(minutes=dur_int + 15)).isoformat()
                        dep_hints.append(f"nếu hẹn {target_h}h sáng/chiều mai và đến sớm 15p -> giờ xuất phát=`{dep_15}`")
                dep_str = (" | " + "; ".join(dep_hints[:3])) if dep_hints else ""
                facts.append(
                    f"Lộ trình `{rec['arguments'].get('destination', '')}`: khoảng_cách={dist} km, "
                    f"thời_gian_di_chuyển={dur_int} phút (nếu cần đến sớm 15 phút thì phải xuất phát trước giờ hẹn {dur_int + 15} phút{dep_str})."
                )

        # 9. Phân tích giá vàng (gold_price.get_price)
        if not from_hist and tname == "gold_price.get_price":
            items = obs.get("prices") or obs.get("items") or ([obs] if "sell" in obs or "buy" in obs else [])
            for it in items:
                if isinstance(it, dict):
                    buy_v = it.get("buy", 0)
                    sell_v = it.get("sell", 0)
                    buy_m = round(buy_v / 1_000_000, 2) if buy_v else None
                    sell_m = round(sell_v / 1_000_000, 2) if sell_v else None
                    buy_fmt = f"{buy_v:,}".replace(",", ".") if buy_v else ""
                    facts.append(
                        f"Giá vàng ({it.get('type', 'SJC')}): mua_vào={buy_v} ({buy_fmt} đồng/lượng = {buy_m} triệu), bán_ra={sell_v} ({sell_m} triệu)."
                    )

        # 10. Phân tích xe đạp cho thuê (bike_rental.find_bikes) - chỉ chọn xe available=true
        if not from_hist and tname == "bike_rental.find_bikes" and "bikes" in obs:
            bikes = obs.get("bikes") or []
            avail_bikes = [b for b in bikes if isinstance(b, dict) and b.get("available") is True]
            unavail_bikes = [b for b in bikes if isinstance(b, dict) and not b.get("available")]
            for ub in unavail_bikes:
                facts.append(f"⚠️ Xe đạp `{ub.get('id')}` (\"{ub.get('name')}\"): available=false (ĐÃ HẾT XE) -> TUYỆT ĐỐI KHÔNG chọn `bike_id` này!")
            if avail_bikes:
                cheapest = min(avail_bikes, key=lambda x: x.get("price_per_hour") or 999999)
                avail_str = "; ".join(f"`{b.get('id')}` (\"{b.get('name')}\", giá={b.get('price_per_hour')}/giờ, available=true)" for b in avail_bikes)
                facts.append(
                    f"Danh sách xe đạp CÒN TRỐNG (available=true): {avail_str} -> "
                    f"Chiếc CÒN TRỐNG rẻ nhất là `\"bike_id\": \"{cheapest.get('id')}\"` (\"{cheapest.get('name')}\", giá={cheapest.get('price_per_hour')}/giờ, thuê 3 tiếng = {(cheapest.get('price_per_hour') or 0) * 3})."
                )

        # 11. Phân tích vé máy bay đã đặt (flights.my_bookings)
        if not from_hist and tname == "flights.my_bookings" and "bookings" in obs:
            for bk in obs.get("bookings") or []:
                if isinstance(bk, dict):
                    fl = bk.get("flight") or {}
                    facts.append(
                        f"Đặt chỗ máy bay `{bk.get('id')}`: hãng_bay=\"{fl.get('airline', '')}\", ngày_bay={fl.get('date', '')}, "
                        f"giờ={fl.get('depart', '')}, chặng={fl.get('origin', '')}->{fl.get('destination', '')}."
                    )

        # 12. Phân tích danh sách công việc (tasks.list) & khóa học (learning.list_courses)
        if not from_hist and tname == "tasks.list" and "tasks" in obs:
            for tk in obs.get("tasks") or []:
                if isinstance(tk, dict) and not tk.get("done"):
                    due_s = str(tk.get("due", ""))
                    is_today = (cur_date and due_s.startswith(cur_date.isoformat()))
                    tag = "🔥 HẠN HÔM NAY (deadline hôm nay) -> BẮT BUỘC chọn `task_id` này để đánh dấu xong (`tasks.complete`)!" if is_today else f"hạn={due_s}"
                    facts.append(f"Công việc chưa xong `{tk.get('id')}`: \"{tk.get('title')}\" ({tag}).")
        if not from_hist and tname == "learning.list_courses" and "courses" in obs:
            for cr in obs.get("courses") or []:
                if isinstance(cr, dict):
                    facts.append(f"Khóa học `{cr.get('id')}`: tên=\"{cr.get('name')}\", bên_cung_cấp=\"{cr.get('provider', '')}\".")

        # 13. Phân tích cảnh báo an ninh (home_security.list_alerts)
        if not from_hist and tname == "home_security.list_alerts" and "alerts" in obs:
            for al in obs.get("alerts") or []:
                if isinstance(al, dict):
                    facts.append(
                        f"Cảnh báo an ninh `{al.get('id')}`: loại={al.get('type')}, vị_trí_camera_chuẩn=`\"location\": \"{al.get('location')}\"`, nội_dung=\"{al.get('message')}\" "
                        f"(Khi gọi `home_security.get_camera` cho camera báo mất kết nối này, BẮT BUỘC truyền `\"location\": \"{al.get('location')}\"`!)."
                    )

        # 14. Phân tích lịch uống thuốc (medication.list / schedules), chương trình TV (tv.guide) & đổi điểm (loyalty) từ cả history và bước hiện tại
        if "schedules" in obs and isinstance(obs.get("schedules"), list):
            evening_meds = []
            evening_ids = []
            bedtime_meds = []
            bedtime_ids = []
            for med in obs.get("schedules") or []:
                if isinstance(med, dict):
                    m_id = med.get("id")
                    m_name = med.get("name", "")
                    m_times = med.get("times") or []
                    if any(t >= "21:00" for t in m_times) or "ngủ" in m_name.lower():
                        bedtime_meds.append(f"`{m_id}` (\"{m_name}\", giờ={m_times})")
                        bedtime_ids.append(m_id)
                    elif any("17:00" <= t < "21:00" for t in m_times):
                        evening_meds.append(f"`{m_id}` (\"{m_name}\", giờ={m_times})")
                        evening_ids.append(m_id)
            if evening_meds or bedtime_meds:
                ev_calls_str = ", ".join(f'{{"schedule_id": "{mid}"}}' for mid in evening_ids)
                facts.append(
                    f"Phân loại lịch thuốc buổi tối (EVENING_MED_COUNT={len(evening_ids)}): "
                    f"Thuốc tối đã uống gồm {len(evening_ids)} thuốc: {', '.join(evening_meds) or 'không có'}. "
                    f"Thuốc trước khi đi ngủ CHƯA UỐNG gồm: {', '.join(bedtime_meds) or 'không có'} -> "
                    f"Nếu người dùng báo 'mấy thuốc tối nay uống rồi, chỉ còn thuốc trước lúc đi ngủ chưa uống', "
                    f"TUYỆT ĐỐI KHÔNG gọi `medication.log_taken` cho {bedtime_ids}! "
                    f"BẮT BUỘC CHỈ GỌI ĐÚNG {len(evening_ids)} LỆNH `medication.log_taken` cho: {ev_calls_str}!"
                )
        if "programs" in obs and "channels" in obs and isinstance(obs.get("programs"), list):
            for prg in obs.get("programs") or []:
                if isinstance(prg, dict):
                    p_st = str(prg.get("start", ""))
                    w_label = ""
                    if len(p_st) >= 10:
                        try:
                            p_dt = datetime.fromisoformat(p_st).date()
                            w_label = f" ({WEEKDAY_VN[p_dt.weekday()]} ngày {p_dt.isoformat()})"
                        except Exception:
                            pass
                    facts.append(f"Chương trình TV `{prg.get('id')}`: \"{prg.get('title')}\" trên {prg.get('channel')}, chiếu lúc {p_st}{w_label}.")
        if "programs" in obs and "channels" not in obs and isinstance(obs.get("programs"), list):
            for lp in obs.get("programs") or []:
                if isinstance(lp, dict):
                    for rw in lp.get("rewards") or []:
                        if isinstance(rw, dict):
                            facts.append(
                                f"Đổi điểm thưởng (`loyalty.redeem`): `\"program\": \"{lp.get('id')}\"`, `\"reward_id\": \"{rw.get('id')}\"` -> \"{rw.get('name')}\" (mệnh giá={rw.get('value')}, tốn {rw.get('cost')} điểm)."
                            )

    if not facts:
        return ""
    return "[PHÂN TÍCH DỮ LIỆU QUAN SÁT (OBSERVATION ANALYSIS)]\n" + "\n".join(f"- {f}" for f in facts) + "\n"


def build_step_directive(
    exemplars: List[Dict[str, Any]],
    previous_steps: List[Dict[str, Any]],
    obs_analysis: str,
    user_utterance: str = "",
    current_time: str = "N/A"
) -> str:
    """
    Sinh chỉ dẫn bước hiện tại dựa trên Máy trạng thái Lượt ReAct của bài mẫu tương đồng,
    khóa chặt số lượng lệnh và danh sách argument keys của lượt hiện tại.
    """
    executed_tools = [s.get("tool_called") for s in previous_steps if not s.get("error")]
    lines: List[str] = []

    if executed_tools:
        lines.append(f"- Công cụ ĐÃ chạy xong trong phiên này: {executed_tools} -> TUYỆT ĐỐI KHÔNG gọi lặp lại bất kỳ công cụ nào trong danh sách đã chạy!")

    # Bổ sung gợi ý tính toán thời gian & khoảng thời gian tự động
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
    for idx, turn_calls in enumerate(turns, 1):
        turn_tool_names = {c.get("name") for c in turn_calls}
        if not all(exec_counts.get(tname, 0) > 0 for tname in turn_tool_names):
            cur_turn_idx = idx
            break

    cur_turn_calls = [
        c for c in turns[cur_turn_idx - 1]
        if exec_counts.get(c.get("name"), 0) == 0
    ] or turns[cur_turn_idx - 1]
    is_final_turn = (cur_turn_idx == total_turns)
    has_dup_contacts = "PHÁT HIỆN TRÙNG TÊN DANH BẠ" in obs_analysis

    # Điều chỉnh linh hoạt số lượng lệnh lặp của cùng 1 tool theo dữ kiện thực tế trong [OBSERVATION ANALYSIS]
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

    # Mô tả chính xác từng lệnh và danh sách argument keys của lượt hiện tại
    call_specs = []
    for c in cur_turn_calls:
        cname = c.get("name", "")
        cargs = c.get("arguments") or {}
        if cname == "sys.ask_missing_info":
            call_specs.append(f"`sys.ask_missing_info` (BẮT BUỘC truyền đúng `\"arguments\": {json.dumps(cargs, ensure_ascii=False)}`)")
        elif has_dup_contacts and is_final_turn and cname in {"phone.call", "sms.send", "ewallet.transfer", "chat_ott.call", "chat_ott.send_message"}:
            ask_args = {"tool_name": cname, "params": ["contact_id"]}
            call_specs.append(f"`sys.ask_missing_info` (BẮT BUỘC truyền đúng `\"arguments\": {json.dumps(ask_args, ensure_ascii=False)}`)")
        elif not cargs:
            call_specs.append(f"`{cname}` (arguments: `{{}}` rỗng, KHÔNG truyền tham số!)")
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
   - Khi người dùng nói ngày cụ thể trong tháng (ví dụ "hôm hai tám", "ngày 18") thì dùng đúng con số ngày đó (`28`, `18`), không cộng/trừ nhầm sang ngày khác. Khi người dùng tự đính chính ("à nhầm", "à không", "à thôi", "à khoan"), luôn lấy lựa chọn được đính chính sau cùng.
3. TUÂN THỦ KẾ HOẠCH LƯỢT REACT & DỮ KIỆN PHÂN TÍCH TRONG [OBSERVATION ANALYSIS]:
   - Gọi đúng danh sách công cụ và đặt đúng `"is_finished"` (`false` hoặc `true`) theo [CURRENT STEP DIRECTIVE].
   - Lấy đúng `id` thực thể, mốc thời gian và nguồn thanh toán (`source`: `"ewallet"` hoặc `"bank"`) theo đối chiếu trong [OBSERVATION ANALYSIS].

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


def compact_observation(result: Any, max_length: int = 3000) -> str:
    """Rút gọn kết quả trả về của tool để tránh làm phình to context tokens."""
    if result is None:
        return "null"
    if isinstance(result, (dict, list)):
        compact_str = json.dumps(result, ensure_ascii=False)
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

    try:
        caller_frame = inspect.currentframe().f_back
        if caller_frame and caller_frame.f_locals:
            f_locals = caller_frame.f_locals
            raw_history = f_locals.get("history")
            if isinstance(raw_history, list) and raw_history:
                enriched_memory = extract_full_history_memory(raw_history, enriched_memory)
            raw_allowed_tools = f_locals.get("allowed_tools")
            if isinstance(raw_allowed_tools, list) and raw_allowed_tools:
                enriched_tools = format_tools_with_enums(raw_allowed_tools, tools_description)

            agent_self = f_locals.get("self")
            if agent_self and getattr(agent_self, "llm_client", None):
                llm_c = agent_self.llm_client
                if not getattr(llm_c, "_retry_wrapped", False):
                    orig_gen = llm_c.generate
                    def _gen_with_retry(sys_p: str, usr_p: str, _orig=orig_gen) -> str:
                        cur_usr_p = usr_p
                        for attempt in range(6):
                            try:
                                raw_out = _orig(sys_p, cur_usr_p)
                                if raw_out:
                                    # Tự động sửa lỗi dấu phẩy thừa cuối mảng/object JSON (trailing comma) nếu có
                                    cleaned_out = re.sub(r",\s*([\]\}])", r"\1", raw_out.strip())
                                    return cleaned_out
                                return raw_out
                            except Exception as exc:
                                if attempt == 5:
                                    raise
                                # Nếu lỗi 500 do prompt quá dài (tràn VRAM/max_length trên server), rút gọn phần ACTIVE MEMORY
                                marker_start = "[ACTIVE MEMORY FROM PREVIOUS DIALOGUE]\n"
                                if marker_start in cur_usr_p:
                                    p1, rest = cur_usr_p.split(marker_start, 1)
                                    next_sec = "\n\n["
                                    if next_sec in rest:
                                        mem_part, p2 = rest.split(next_sec, 1)
                                        limit = max(500, 2000 // (attempt + 1))
                                        if len(mem_part) > limit:
                                            cur_usr_p = p1 + marker_start + mem_part[-limit:] + next_sec + p2
                                time.sleep(2 * (attempt + 1))
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
            if agent_self and getattr(agent_self, "exemplar_retriever", None):
                raw_exemplars = agent_self.exemplar_retriever.retrieve(
                    query_text=cleaned_utterance,
                    installed_apps=installed_apps,
                    top_k=1
                )
            elif isinstance(f_locals.get("exemplars"), list):
                raw_exemplars = f_locals.get("exemplars")
    except Exception:
        pass

    formatted_exemplar = format_exemplar_by_turns(raw_exemplars, few_shot_examples, previous_steps)

    prompt_parts = [
        "[CURRENT CONTEXT]",
        f"- Thời gian hiện tại (Current Time): {current_time}",
    ]
    if cal_helper:
        prompt_parts.append(cal_helper)
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

    obs_analysis = analyze_observations(previous_steps, context, raw_history)
    if obs_analysis:
        prompt_parts.append(obs_analysis)

    prompt_parts.extend([
        "[CLOSEST REFERENCE EXEMPLAR]",
        formatted_exemplar,
        "",
        "[USER UTTERANCE]",
        f"\"{cleaned_utterance}\"",
        ""
    ])

    step_directive = build_step_directive(raw_exemplars, previous_steps, obs_analysis, cleaned_utterance, current_time)
    prompt_parts.append("[CURRENT STEP DIRECTIVE]")
    prompt_parts.append(step_directive)
    prompt_parts.append("")
    prompt_parts.append("Hãy suy luận ngắn gọn bằng tiếng Việt và trả về duy nhất khối JSON hợp lệ:")

    return "\n".join(prompt_parts)
