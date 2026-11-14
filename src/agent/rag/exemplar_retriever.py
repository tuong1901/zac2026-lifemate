"""
exemplar_retriever.py — BM25 Sparse Retrieval trên tập dữ liệu train (ZAC 2026 LifeMate).
Phụ trách: Thành viên 1 (Few-Shot Exemplar RAG).

Đặc điểm kỹ thuật:
- Thuần túy giải tích toán học (BM25Okapi với Inverted Index), KHÔNG dùng mô hình AI (0 tham số, 0 VRAM).
- Tokenize tiếng Việt chuyên sâu: Chuẩn hóa Unicode NFC + Lowercase + Unigram + Bigram để bắt trọn
  các từ ghép ("máy giặt", "máy sấy", "chuyển tiền", "báo thức", "cửa ban công").
- Bộ lọc Hard App Filter: Chỉ lấy các bài mẫu mà tất cả tools gọi (bao gồm cả tool_name trong sys.ask_missing_info)
  thuộc danh sách installed_apps của mẫu hiện tại, tránh tuyệt đối việc gợi ý tool của app chưa cài.
- Tốc độ suy luận: < 1ms / mẫu.
"""

import os
import re
import math
import json
import pickle
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple


_VN_TONE_NORM_RES = [
    (re.compile(rf"(?<![qQ]){old_s}\b"), new_s)
    for old_s, new_s in (
        ("oà", "òa"), ("oá", "óa"), ("oả", "ỏa"), ("oã", "õa"), ("oạ", "ọa"),
        ("uỳ", "ùy"), ("uý", "úy"), ("uỷ", "ủy"), ("uỹ", "ũy"), ("uỵ", "ụy"),
    )
]


def normalize_vietnamese_text(text: str) -> str:
    """
    Chuẩn hóa chuỗi tiếng Việt theo chuẩn NFC, đồng nhất vị trí dấu thanh và lowercase.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text.strip().lower())
    for comp_re, new_s in _VN_TONE_NORM_RES:
        text = comp_re.sub(new_s, text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_vietnamese(text: str, use_bigrams: bool = True) -> List[str]:
    """
    Tách từ tiếng Việt dạng Unigram + Bigram để BM25 bắt chính xác từ ghép.
    """
    cleaned = normalize_vietnamese_text(text)
    if not cleaned:
        return []
    words = cleaned.split()
    tokens = list(words)
    if use_bigrams and len(words) >= 2:
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")
    return tokens


class BM25ExemplarRetriever:
    """
    Hệ thống truy xuất ví dụ mẫu tương đồng bằng thuật toán BM25Okapi kết hợp Inverted Index.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freq: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}
        self.is_indexed: bool = False

    @staticmethod
    def _compute_called_apps(tool_calls: List[Dict[str, Any]]) -> Set[str]:
        called_apps: Set[str] = set()
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name", "")
            if name == "sys.ask_missing_info":
                called_apps.add("sys")
                target_tn = tc.get("arguments", {}).get("tool_name", "")
                if "." in target_tn:
                    called_apps.add(target_tn.split(".")[0])
            elif "." in name:
                app_name = name.split(".")[0]
                called_apps.add(app_name)
            elif name:
                called_apps.add(name)
        return called_apps

    @classmethod
    def _extract_history_exemplars(
        cls,
        item: Dict[str, Any],
        installed_apps: List[str],
    ) -> List[Dict[str, Any]]:
        """
        History Mining (muahahaha55): Khai thác các lượt (user -> assistant tool_calls)
        hợp lệ trong history của mẫu train để mở rộng kho ví dụ BM25.
        """
        history = item.get("history") or []
        if not isinstance(history, list) or not history:
            return []
        sample_id = item.get("id", "")
        mined: List[Dict[str, Any]] = []
        pending_user_text = ""
        turn_idx = 0
        for msg in history:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            if role == "user":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    pending_user_text = content.strip()
            elif role == "assistant" and pending_user_text:
                raw_tcs = msg.get("tool_calls") or []
                clean_tcs: List[Dict[str, Any]] = []
                for tc in raw_tcs:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                    t_name = fn.get("name", "")
                    t_args = fn.get("arguments", {})
                    if isinstance(t_args, str):
                        try:
                            t_args = json.loads(t_args)
                        except Exception:
                            t_args = {}
                    if t_name and isinstance(t_args, dict):
                        clean_tcs.append({"name": t_name, "arguments": t_args})
                if clean_tcs:
                    called_apps = cls._compute_called_apps(clean_tcs)
                    if called_apps:
                        turn_idx += 1
                        mined.append({
                            "id": f"{sample_id}#hist_{turn_idx}",
                            "audio_text": pending_user_text,
                            "installed_apps": installed_apps,
                            "called_apps": called_apps,
                            "tool_calls": clean_tcs,
                        })
                pending_user_text = ""
        return mined

    def build_from_train_file(self, train_path: str, exclude_ids: Optional[Set[str]] = None):
        """
        Nạp toàn bộ các mẫu từ train.jsonl (kể cả History Mining) và khởi tạo chỉ mục BM25.
        """
        path = Path(train_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file train tại: {train_path}")

        records = []
        seen_pairs: Set[Tuple[str, str]] = set()
        exclude_ids = exclude_ids or set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item.get("id") in exclude_ids:
                    continue

                sample_id = item.get("id", "")
                audio_text = item.get("audio_text", "")
                context = item.get("context", {})
                installed_apps = list(context.get("installed_apps", {}).keys())

                label = item.get("label", {})
                tool_calls = label.get("tool_calls", [])
                called_apps = self._compute_called_apps(tool_calls)

                sig_names = ",".join(tc.get("name", "") for tc in tool_calls if isinstance(tc, dict))
                pair_key = (normalize_vietnamese_text(audio_text), sig_names)
                seen_pairs.add(pair_key)

                records.append({
                    "id": sample_id,
                    "audio_text": audio_text,
                    "installed_apps": installed_apps,
                    "called_apps": called_apps,
                    "tool_calls": tool_calls
                })

                for h_rec in self._extract_history_exemplars(item, installed_apps):
                    h_sig = ",".join(tc.get("name", "") for tc in h_rec["tool_calls"])
                    h_key = (normalize_vietnamese_text(h_rec["audio_text"]), h_sig)
                    if h_key not in seen_pairs:
                        seen_pairs.add(h_key)
                        records.append(h_rec)

        self.index_records(records)

    def index_records(self, records: List[Dict[str, Any]]):
        """
        Đánh chỉ mục Inverted Index BM25 cho danh sách bản ghi.
        """
        self.corpus = records
        num_docs = len(records)
        if num_docs == 0:
            return

        self.doc_len = []
        self.doc_freq = {}
        self.inverted_index = {}

        total_len = 0
        for doc_idx, rec in enumerate(records):
            tokens = tokenize_vietnamese(rec["audio_text"])
            d_len = len(tokens)
            self.doc_len.append(d_len)
            total_len += d_len

            tf_map: Dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for t, count in tf_map.items():
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
                if t not in self.inverted_index:
                    self.inverted_index[t] = []
                self.inverted_index[t].append((doc_idx, count))

        self.avgdl = total_len / num_docs if num_docs > 0 else 1.0

        self.idf = {}
        for t, df in self.doc_freq.items():
            self.idf[t] = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))

        self.is_indexed = True

    def retrieve(
        self,
        query_text: str,
        installed_apps: Optional[Dict[str, Any]] = None,
        top_k: int = 2,
        hard_app_filter: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm Top-K ví dụ mẫu tương đồng nhất bằng BM25.
        """
        if not self.is_indexed or not self.corpus:
            return []

        query_tokens = tokenize_vietnamese(query_text)
        if not query_tokens:
            return []

        allowed_apps: Set[str] = set()
        if installed_apps:
            if isinstance(installed_apps, dict):
                allowed_apps = set(installed_apps.keys())
            elif isinstance(installed_apps, (list, set)):
                allowed_apps = set(installed_apps)
        allowed_apps.add("sys")

        scores: Dict[int, float] = {}
        for t in query_tokens:
            if t not in self.inverted_index:
                continue
            t_idf = self.idf[t]
            for doc_idx, tf in self.inverted_index[t]:
                if hard_app_filter and allowed_apps:
                    rec_called = self.corpus[doc_idx]["called_apps"]
                    if not rec_called or not rec_called.issubset(allowed_apps):
                        continue

                d_len = self.doc_len[doc_idx]
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (d_len / self.avgdl))
                numerator = tf * (self.k1 + 1.0)
                term_score = t_idf * (numerator / denominator)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + term_score

        if not scores:
            return []

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_idx, score in sorted_docs:
            rec = self.corpus[doc_idx].copy()
            rec["bm25_score"] = round(score, 4)
            results.append(rec)

        return results

    def format_exemplars_for_prompt(self, exemplars: List[Dict[str, Any]]) -> str:
        """
        Định dạng danh sách ví dụ thành chuỗi Markdown ngắn gọn cho LLM Prompt.
        """
        if not exemplars:
            return "No reference example available."

        lines = ["[CLOSEST REFERENCE EXEMPLAR]"]
        for idx, ex in enumerate(exemplars, 1):
            lines.append(f"- Example Utterance: \"{ex['audio_text']}\"")
            tool_calls = ex.get("tool_calls", [])
            if not tool_calls:
                lines.append("- Expected Tools: [] (Out of scope)")
            else:
                lines.append("- Expected Tool Sequence:")
                for s_i, tc in enumerate(tool_calls, 1):
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})
                    args_compact = json.dumps(args, ensure_ascii=False)
                    lines.append(f"  + Step {s_i}: `{name}({args_compact})`")

        return "\n".join(lines)

    def save_cache(self, cache_path: str):
        data = {
            "version": "v2_history_mined",
            "k1": self.k1,
            "b": self.b,
            "corpus": self.corpus,
            "doc_len": self.doc_len,
            "avgdl": self.avgdl,
            "doc_freq": self.doc_freq,
            "idf": self.idf,
            "inverted_index": self.inverted_index,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_cache(self, cache_path: str):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if data.get("version") != "v2_history_mined":
            raise ValueError("Outdated BM25 cache version")
        self.k1 = data["k1"]
        self.b = data["b"]
        self.corpus = data["corpus"]
        self.doc_len = data["doc_len"]
        self.avgdl = data["avgdl"]
        self.doc_freq = data["doc_freq"]
        self.idf = data["idf"]
        self.inverted_index = data["inverted_index"]
        self.is_indexed = True
