"""
evaluate.py — Hệ thống đánh giá chi tiết (Detailed Evaluation System) cho ASR trong ZAC 2026 LifeMate.

Đo lường đầy đủ và chi tiết:
1. WER (Word Error Rate) kèm chi tiết: Substitutions (S), Deletions (D), Insertions (I).
2. CER (Character Error Rate).
3. Token-F1 theo chuẩn ZAC 2026 (NFC normalization, lowercased, multiset token match).
4. Tỷ lệ Exact Match (EM).
5. Tốc độ suy luận: RTF (Real-Time Factor), Độ trễ trung bình (ms/mẫu), Throughput (mẫu/giây).
6. Phân tích Ablation: So sánh chi tiết trước và sau từng tầng hậu xử lý (Raw vs Tầng 1 vs Tầng 2 vs Tầng 3).
7. Xuất báo cáo đa định dạng: Console bảng biểu, Markdown Report (.md), JSON metrics (.json).
"""

import json
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


def normalize_text_for_eval(text: str) -> str:
    """Chuẩn hóa văn bản trước khi tính điểm: Unicode NFC, viết thường, chuẩn hóa dấu cách."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).strip().lower()
    return " ".join(text.split())


def compute_levenshtein_distance(seq1: List[str], seq2: List[str]) -> Tuple[int, int, int, int]:
    """
    Tính khoảng cách Levenshtein giữa 2 chuỗi token (từ hoặc ký tự) bằng quy hoạch động thuần Python.
    Trả về: (total_dist, substitutions, deletions, insertions)
    """
    n, m = len(seq1), len(seq2)
    # dp[i][j] = (cost, s, d, i)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)  # deletion from seq1
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)  # insertion to seq1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # substitution
                cost_sub, s, d, ins = dp[i - 1][j - 1]
                op_sub = (cost_sub + 1, s + 1, d, ins)

                # deletion
                cost_del, s, d, ins = dp[i - 1][j]
                op_del = (cost_del + 1, s, d + 1, ins)

                # insertion
                cost_ins, s, d, ins = dp[i][j - 1]
                op_ins = (cost_ins + 1, s, d, ins + 1)

                dp[i][j] = min(op_sub, op_del, op_ins, key=lambda x: x[0])

    return dp[n][m]


def compute_wer(ref: str, hyp: str) -> Dict[str, Any]:
    """Tính Word Error Rate (WER) giữa chuỗi gốc (ref) và chuỗi dự đoán (hyp)."""
    ref_words = normalize_text_for_eval(ref).split()
    hyp_words = normalize_text_for_eval(hyp).split()

    if not ref_words and not hyp_words:
        return {"wer": 0.0, "s": 0, "d": 0, "i": 0, "n": 0}
    if not ref_words:
        return {"wer": 1.0, "s": 0, "d": 0, "i": len(hyp_words), "n": len(hyp_words)}

    dist, s, d, ins = compute_levenshtein_distance(ref_words, hyp_words)
    n = len(ref_words)
    wer = dist / n
    return {"wer": wer, "s": s, "d": d, "i": ins, "n": n}


def compute_cer(ref: str, hyp: str) -> float:
    """Tính Character Error Rate (CER)."""
    ref_chars = list(normalize_text_for_eval(ref).replace(" ", ""))
    hyp_chars = list(normalize_text_for_eval(hyp).replace(" ", ""))

    if not ref_chars and not hyp_chars:
        return 0.0
    if not ref_chars:
        return 1.0

    dist, _, _, _ = compute_levenshtein_distance(ref_chars, hyp_chars)
    return dist / len(ref_chars)


def compute_token_f1(ref: str, hyp: str) -> float:
    """Tính Token-F1 theo chuẩn Zalo AI Challenge 2026."""
    ref_tokens = normalize_text_for_eval(ref).split()
    hyp_tokens = normalize_text_for_eval(hyp).split()

    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0

    c_ref = Counter(ref_tokens)
    c_hyp = Counter(hyp_tokens)
    common = sum((c_ref & c_hyp).values())

    p = common / len(hyp_tokens) if hyp_tokens else 0.0
    r = common / len(ref_tokens) if ref_tokens else 0.0

    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


@dataclass
class ASREvalMetrics:
    total_samples: int
    wer: float
    cer: float
    token_f1: float
    exact_match: float
    substitutions: int
    deletions: int
    insertions: int
    total_words: int
    avg_latency_ms: float
    throughput_samples_per_sec: float
    rtf: Optional[float] = None


class ASREvaluator:
    """Bộ đánh giá toàn diện ASR và so sánh Ablation các tầng Post-Processing."""

    def __init__(self, name: str = "ASR Evaluation"):
        self.name = name

    def evaluate_predictions(
        self,
        references: List[str],
        hypotheses: List[str],
        sample_ids: Optional[List[str]] = None,
        elapsed_time_sec: float = 0.0,
        total_audio_duration_sec: float = 0.0
    ) -> Tuple[ASREvalMetrics, List[Dict[str, Any]]]:
        assert len(references) == len(hypotheses), "Độ dài references và hypotheses phải bằng nhau"

        total_n = len(references)
        if total_n == 0:
            raise ValueError("Danh sách mẫu rỗng!")

        total_s = 0
        total_d = 0
        total_i = 0
        total_words = 0
        total_cer = 0.0
        total_f1 = 0.0
        exact_matches = 0

        sample_details = []

        for idx, (ref, hyp) in enumerate(zip(references, hypotheses)):
            sample_id = sample_ids[idx] if sample_ids else f"sample_{idx}"
            w_res = compute_wer(ref, hyp)
            c_val = compute_cer(ref, hyp)
            f1_val = compute_token_f1(ref, hyp)

            total_s += w_res["s"]
            total_d += w_res["d"]
            total_i += w_res["i"]
            total_words += w_res["n"]
            total_cer += c_val
            total_f1 += f1_val

            is_em = (normalize_text_for_eval(ref) == normalize_text_for_eval(hyp))
            if is_em:
                exact_matches += 1

            sample_details.append({
                "id": sample_id,
                "gold": ref,
                "pred": hyp,
                "wer": round(w_res["wer"] * 100, 2),
                "cer": round(c_val * 100, 2),
                "f1": round(f1_val * 100, 2),
                "exact_match": is_em,
                "s": w_res["s"],
                "d": w_res["d"],
                "i": w_res["i"]
            })

        overall_wer = (total_s + total_d + total_i) / max(total_words, 1) * 100
        overall_cer = (total_cer / total_n) * 100
        overall_f1 = (total_f1 / total_n) * 100
        overall_em = (exact_matches / total_n) * 100

        avg_latency = (elapsed_time_sec / total_n) * 1000 if elapsed_time_sec > 0 else 0.0
        throughput = total_n / elapsed_time_sec if elapsed_time_sec > 0 else 0.0
        rtf = (elapsed_time_sec / total_audio_duration_sec) if total_audio_duration_sec > 0 else None

        metrics = ASREvalMetrics(
            total_samples=total_n,
            wer=round(overall_wer, 2),
            cer=round(overall_cer, 2),
            token_f1=round(overall_f1, 2),
            exact_match=round(overall_em, 2),
            substitutions=total_s,
            deletions=total_d,
            insertions=total_i,
            total_words=total_words,
            avg_latency_ms=round(avg_latency, 2),
            throughput_samples_per_sec=round(throughput, 2),
            rtf=round(rtf, 4) if rtf else None
        )

        return metrics, sample_details

    def evaluate_ablation(
        self,
        references: List[str],
        processed_dicts: List[Dict[str, str]],
        sample_ids: Optional[List[str]] = None
    ) -> Dict[str, ASREvalMetrics]:
        """
        Đánh giá so sánh chi tiết từng bước:
        1. Raw (ASR nguyên bản)
        2. + Tầng 1 (Lexicon/Rule)
        3. + Tầng 2 (Contextual ViT5)
        4. + Tầng 3 (Punctuation Restoration)
        """
        stages = [
            ("1. Raw ASR", [d["raw"] for d in processed_dicts]),
            ("2. + Tầng 1 (Lexicon/Rule)", [d["t1_lexicon"] for d in processed_dicts]),
            ("3. + Tầng 2 (Context ViT5)", [d["t2_context"] for d in processed_dicts]),
            ("4. + Tầng 3 (ViBERT Punct)", [d["t3_punct"] for d in processed_dicts]),
        ]

        ablation_results = {}
        for stage_name, hyps in stages:
            metrics, _ = self.evaluate_predictions(references, hyps, sample_ids=sample_ids)
            ablation_results[stage_name] = metrics

        return ablation_results

    def generate_markdown_report(
        self,
        metrics: ASREvalMetrics,
        ablation: Optional[Dict[str, ASREvalMetrics]] = None,
        sample_details: Optional[List[Dict[str, Any]]] = None,
        output_path: Optional[Path] = None
    ) -> str:
        """Tạo báo cáo chi tiết dạng Markdown chuyên nghiệp."""
        lines = [
            f"# Báo Cáo Đánh Giá Hiệu Năng ASR & Post-Processing",
            f"**Tác vụ:** Zalo AI Challenge 2026 — LifeMate",
            f"**Số lượng mẫu đánh giá:** {metrics.total_samples}",
            "",
            "## 1. Tổng quan Chỉ số Chính",
            "",
            "| Chỉ số | Kết quả | Mô tả |",
            "| :--- | :--- | :--- |",
            f"| **Word Error Rate (WER)** | **{metrics.wer}%** | Tỷ lệ lỗi cấp độ từ (càng thấp càng tốt) |",
            f"| **Character Error Rate (CER)** | **{metrics.cer}%** | Tỷ lệ lỗi cấp độ ký tự |",
            f"| **Token-F1 Score** | **{metrics.token_f1}%** | Độ chuẩn xác Token-F1 theo luật ZAC |",
            f"| **Exact Match (EM)** | **{metrics.exact_match}%** | Tỷ lệ câu nhận diện chính xác 100% |",
            f"| **Độ trễ trung bình** | **{metrics.avg_latency_ms} ms/mẫu** | Thời gian xử lý bình quân |",
            f"| **Throughput** | **{metrics.throughput_samples_per_sec} mẫu/giây** | Tốc độ xử lý song song trên GPU |",
            f"| **Chi tiết lỗi từ (S/D/I)** | {metrics.substitutions} / {metrics.deletions} / {metrics.insertions} | Thay thế (S), Bỏ sót (D), Chèn thêm (I) / Tổng {metrics.total_words} từ |",
            "",
        ]

        if ablation:
            lines.extend([
                "## 2. Phân Tích Đóng Góp Của Từng Tầng Hậu Xử Lý (Ablation Study)",
                "",
                "| Giai đoạn hậu xử lý | WER (%) | CER (%) | Token-F1 (%) | Exact Match (%) |",
                "| :--- | :---: | :---: | :---: | :---: |",
            ])
            for st_name, m in ablation.items():
                lines.append(f"| {st_name} | {m.wer}% | {m.cer}% | {m.token_f1}% | {m.exact_match}% |")
            lines.append("")

        if sample_details:
            # Lấy top 5 mẫu có WER cao nhất để phân tích lỗi
            worst_samples = sorted(sample_details, key=lambda x: x["wer"], reverse=True)[:5]
            lines.extend([
                "## 3. Phân Tích 5 Mẫu Lỗi Điển Hình (Cần Chú Ý)",
                "",
                "| ID | Gold Text | Predicted Text | WER | Token-F1 |",
                "| :--- | :--- | :--- | :---: | :---: |",
            ])
            for s in worst_samples:
                gold_esc = s["gold"].replace("|", "\\|")
                pred_esc = s["pred"].replace("|", "\\|")
                lines.append(f"| `{s['id']}` | {gold_esc} | {pred_esc} | {s['wer']}% | {s['f1']}% |")
            lines.append("")

        report_content = "\n".join(lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"[Đánh giá] Đã lưu báo cáo chi tiết tại: {output_path}")

        return report_content

