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
    threshold: float = 0.78
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
        ent_words = ent_lower.split()
        k = len(ent_words)
        if k == 0:
            continue
        if re.search(rf"(?<!\w){re.escape(ent)}(?!\w)", repaired_text, re.IGNORECASE):
            continue

        best_ratio = 0.0
        best_ngram = ""

        # Thử các cửa sổ có độ dài k, k-1, k+1
        for w_len in (k, k - 1, k + 1):
            if w_len <= 0 or w_len > n_words:
                continue
            for i in range(n_words - w_len + 1):
                ngram = " ".join(words[i : i + w_len])
                ngram_clean = re.sub(r"[^\w\s]", "", ngram).strip().lower()
                ent_clean = re.sub(r"[^\w\s]", "", ent_lower).strip()

                if ngram_clean in exact_entity_texts:
                    continue

                # Bỏ qua nếu đã khớp chính xác hoặc toàn bộ từ trong cụm đều đã có sẵn trong thực thể
                if ngram_clean == ent_clean or set(ngram_clean.split()).issubset(set(ent_clean.split())):
                    continue

                r = difflib.SequenceMatcher(None, ngram_clean, ent_clean).ratio()
                if r > best_ratio:
                    best_ratio = r
                    best_ngram = ngram

        if best_ratio >= threshold:
            repairs.append({
                "original": best_ngram,
                "repaired": ent,
                "score": round(best_ratio, 3)
            })
            # Thay thế cụm từ trong văn bản
            pattern = re.compile(re.escape(best_ngram), re.IGNORECASE)
            repaired_text = pattern.sub(ent, repaired_text, count=1)
            # Cập nhật lại words cho các vòng lặp tiếp theo
            words = repaired_text.split()
            n_words = len(words)

    return repaired_text, repairs
