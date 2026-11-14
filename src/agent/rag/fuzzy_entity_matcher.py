"""
fuzzy_entity_matcher.py — Bộ đối sánh thực thể động từ ngữ cảnh và lịch sử (Zero-Hardcode).
Tự động bóc tách các thực thể (tên người, địa điểm, sự kiện, bài hát, nhà hàng, kênh TV)
từ context.user_profile và history.tool_results để sửa sai âm vị ASR cho các tên riêng mới lạ (unseen entities).
Sử dụng difflib (chuẩn Python, 0 dependency, 0 VRAM, thực thi < 2ms).
"""

import re
import difflib
from typing import Dict, List, Any, Set, Tuple, Optional


def extract_candidate_entities(item: Dict[str, Any]) -> Set[str]:
    """
    Bóc tách tất cả các thực thể tiềm năng từ hồ sơ người dùng và lịch sử cuộc gọi tool.
    """
    entities: Set[str] = set()
    context = item.get("context", {})
    user_prof = context.get("user_profile", {})

    # 1. Hồ sơ người dùng
    for k, v in user_prof.items():
        if isinstance(v, str) and len(v.strip()) > 1:
            entities.add(v.strip())

    # 2. Lịch sử hội thoại & tool_results
    for turn in item.get("history", []):
        for res in turn.get("tool_results", []):
            if not isinstance(res, dict):
                continue
            for k, val in res.items():
                if k == "ok":
                    continue
                if isinstance(val, list):
                    for obj in val:
                        if isinstance(obj, dict):
                            for name_field in ("name", "title", "label", "channel", "host", "artist", "dish_name", "location", "address"):
                                if name_field in obj and isinstance(obj[name_field], str):
                                    entities.add(obj[name_field].strip())
                        elif isinstance(obj, str):
                            entities.add(obj.strip())
                elif isinstance(val, dict):
                    for name_field in ("name", "title", "label", "channel", "host", "artist", "dish_name", "location", "address"):
                        if name_field in val and isinstance(val[name_field], str):
                            entities.add(val[name_field].strip())

    # Lọc các chuỗi quá ngắn, toàn số, hoặc câu văn thông thường chỉ viết hoa chữ đầu câu (chỉ giữ Tên riêng thực sự)
    filtered = {
        e for e in entities
        if len(e) >= 3 and not e.isdigit()
        and (
            (len(e.split()) == 1 and any(ch.isupper() for ch in e))
            or any(w[0].isupper() for w in e.split()[1:] if w)
        )
        and not e.startswith(("evt_", "alm_", "not_", "lock_", "rule_", "sub_", "rem_", "task_", "chore_"))
    }
    return filtered


def repair_utterance_with_context_entities(
    text: str,
    candidate_entities: Set[str],
    threshold: float = 0.84
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Tìm kiếm các cụm từ trong text có độ tương đồng cao với thực thể trong ngữ cảnh
    và tự động thay thế bằng tên thực thể chuẩn hóa.

    Returns:
        (repaired_text, list_of_repairs)
    """
    if not text or not candidate_entities:
        return text, []

    words = text.split()
    n_words = len(words)
    if n_words == 0:
        return text, []

    repairs = []
    repaired_text = text

    # Các từ hạt (particles), đại từ, liên từ, danh từ chung phổ biến trong tiếng Việt
    # tuyệt đối không được biến thành một phần của tên riêng khi sửa lỗi mờ.
    FORBIDDEN_BOUNDARY_WORDS = {
        "nhé", "nha", "nhỉ", "nghen", "nghe", "nhen", "hả", "hỉ", "ạ", "đi", "luôn",
        "thôi", "rồi", "đâu", "đó", "đấy", "vậy", "chưa", "không", "hông", "nữa", "với",
        "nhưng", "mà", "thì", "là", "để", "khi", "nếu", "vì", "do", "bởi", "hoặc", "hay",
        "ra", "vào", "vô", "lên", "xuống", "qua", "lại", "cho", "của", "từ", "tới", "đến",
        "bản", "cái", "chiếc", "cuốn", "quyển", "bộ", "tập", "món", "quán", "tiệm", "nhà",
        "phòng", "lịch", "việc", "thuốc", "đèn", "quạt", "xe", "vé", "tiền", "khoản",
        "thằng", "dương", "hướng", "trường", "thường", "tháng", "chứng", "phương", "chương",
    }

    # Preserve entities already transcribed exactly. A fuzzy neighbour such as
    # "Kevin Welsh" must not overwrite an exact "Kevin Walsh" occurrence.
    exact_entity_texts = {
        re.sub(r"[^\w\s]", "", entity).strip().lower()
        for entity in candidate_entities
    }

    # Deterministic tie-breaker avoids process-dependent results from set order.
    sorted_entities = sorted(candidate_entities, key=lambda x: (-len(x), x.casefold()))

    for ent in sorted_entities:
        ent_lower = ent.lower()
        ent_clean = re.sub(r"[^\w\s]", "", ent_lower).strip()
        ent_words = ent_clean.split()
        k = len(ent_words)
        # Chỉ sửa mờ các thực thể từ 2 từ trở lên để tránh biến từ đơn tiếng Việt ("nhưng", "dương", "thằng") thành tên người
        if k < 2 or k > n_words:
            continue
        if re.search(rf"(?<!\w){re.escape(ent)}(?!\w)", repaired_text, re.IGNORECASE):
            continue

        best_ratio = 0.0
        best_ngram = ""

        # Chỉ đối sánh cửa sổ có cùng số từ k == len(ent_words)
        for i in range(n_words - k + 1):
            raw_slice = words[i : i + k]
            # Không vượt qua dấu câu ngăn cách mệnh đề (, . ! ? ; :)
            if any(re.search(r"[,.;:!?]", w) for w in raw_slice[:-1]):
                continue
            ngram = " ".join(raw_slice)
            # Bỏ dấu câu ở cuối từ cuối cùng nếu có để chỉ thay thế phần chữ
            ngram_stripped = re.sub(r"[,.!?;:]+$", "", ngram)
            ngram_clean = re.sub(r"[^\w\s]", "", ngram_stripped).strip().lower()
            ngram_words = ngram_clean.split()
            if len(ngram_words) != k:
                continue

            if ngram_clean in exact_entity_texts:
                continue

            if ngram_clean == ent_clean or set(ngram_words).issubset(set(ent_words)):
                continue

            # Không cho phép từ đầu hoặc từ cuối của ngram là hư từ/đại từ/từ phổ biến nếu từ đó không khớp chính xác với thực thể
            if ngram_words[0] != ent_words[0] and ngram_words[0] in FORBIDDEN_BOUNDARY_WORDS:
                continue
            if ngram_words[-1] != ent_words[-1] and ngram_words[-1] in FORBIDDEN_BOUNDARY_WORDS:
                continue

            # Không đổi danh xưng tiếng Anh Mr. <-> Mrs. / Ms.
            if {ngram_words[0], ent_words[0]} <= {"mr", "mrs", "ms", "miss"} and ngram_words[0] != ent_words[0]:
                continue

            # Nếu cụm từ gốc trong câu đã viết hoa tất cả các từ (ASR đã nhận diện một tên riêng hoàn chỉnh như "Sophie Martin", "Mekong Riverside Grill"),
            # không ghi đè sang thực thể khác nếu có bất kỳ từ nào độ tương đồng < 0.85
            orig_tokens_alpha = [re.sub(r"[^\w]", "", w) for w in raw_slice]
            all_capitalized = all(t and t[0].isupper() for t in orig_tokens_alpha)

            # Kiểm tra độ tương đồng từng từ tương ứng
            word_ratios = [difflib.SequenceMatcher(None, nw, ew).ratio() for nw, ew in zip(ngram_words, ent_words)]
            min_word_r = min(word_ratios)
            if min_word_r < 0.66:
                continue
            if all_capitalized and min_word_r < 0.88:
                continue
            # Ít nhất (k - 1) từ phải khớp rất cao (>= 0.85) để đảm bảo chỉ sửa 1 âm tiết bị lệch âm vị ASR
            high_match_count = sum(1 for wr in word_ratios if wr >= 0.85)
            if high_match_count < k - 1:
                continue

            r = difflib.SequenceMatcher(None, ngram_clean, ent_clean).ratio()
            if r > best_ratio:
                best_ratio = r
                best_ngram = ngram_stripped

        if best_ratio >= threshold and best_ngram:
            repairs.append({
                "original": best_ngram,
                "repaired": ent,
                "score": round(best_ratio, 3)
            })
            pattern = re.compile(re.escape(best_ngram), re.IGNORECASE)
            repaired_text = pattern.sub(ent, repaired_text, count=1)
            words = repaired_text.split()
            n_words = len(words)

    return repaired_text, repairs
