"""
batch_runner.py — Trình điều phối chạy xử lý ASR theo lô (Batch Processing Pipeline).
Tự động phối hợp:
1. UnifiedASRTranscriber (PhoWhisper / PhoASR / FasterWhisper theo lô).
2. CompositePostProcessor (Tầng 1 Lexicon -> Tầng 2 ViT5 -> Tầng 3 ViBERT).
3. ASREvaluator (Đánh giá chi tiết WER, CER, Token-F1, Latency, Ablation).
4. Tương thích định dạng cho ReAct Agent và file submit của ZAC 2026 LifeMate.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from src.asr.config import ASRConfig, get_asr_config
from src.asr.transcribe import UnifiedASRTranscriber
from src.asr.post_processor import CompositePostProcessor
from src.asr.evaluate import ASREvaluator


def extract_context_hints_from_item(item: Dict[str, Any]) -> List[str]:
    """
    Trích xuất an toàn các thực thể gợi ý từ item JSONL (context, user_profile, app names).
    TUYỆT ĐỐI KHÔNG đọc bất kỳ file nào từ state.json hay <split>_environment/ (tuân thủ .cursorrules).
    """
    hints = []
    context = item.get("context", {})
    if isinstance(context, dict):
        user_prof = context.get("user_profile", {})
        if isinstance(user_prof, dict):
            if user_prof.get("name"):
                hints.append(str(user_prof["name"]))
            if user_prof.get("city"):
                hints.append(str(user_prof["city"]))

        apps = context.get("installed_apps", {})
        if isinstance(apps, dict):
            hints.extend(list(apps.keys()))

    # Lấy thêm tên người liên hệ hoặc thực thể trong history nếu có
    history = item.get("history", [])
    if isinstance(history, list):
        for turn in history:
            text = turn.get("text") or turn.get("audio_text")
            if text and isinstance(text, str):
                # Quét từ khóa ngắn
                for token in text.split():
                    if token.istitle() and len(token) > 2 and token not in hints:
                        hints.append(token)
                        if len(hints) >= 15:
                            break

    return hints[:12]


class ASRBatchRunner:
    """
    Trình chạy Batch toàn diện cho tập dữ liệu (train, val_benchmark_200, public_test).
    """
    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or get_asr_config()

        print(self.config.summary())

        # Khởi tạo Transcriber & PostProcessor
        print("\n[ASR Pipeline] Đang khởi tạo ASR Transcriber...")
        self.transcriber = UnifiedASRTranscriber(self.config)

        print("[ASR Pipeline] Đang khởi tạo Composite Post-Processor (3 Tầng)...")
        self.post_processor = CompositePostProcessor(
            enable_t1=self.config.enable_lexicon_correction,
            enable_t2=self.config.enable_contextual_correction,
            enable_t3=self.config.enable_punctuation_restoration,
            context_model_name=self.config.context_model_name_or_path,
            punct_model_name=self.config.punctuation_model_name_or_path,
            device=self.config.device,
            batch_size=self.config.batch_size
        )

        self.evaluator = ASREvaluator(name=f"Evaluation-{self.config.asr_engine}")

    def run_on_dataset(
        self,
        input_jsonl_path: Path,
        output_jsonl_path: Path,
        audio_base_dir: Path,
        limit: Optional[int] = None,
        transcribe_history: bool = True
    ) -> Dict[str, Any]:
        """
        Chạy toàn bộ pipeline trên một file jsonl theo batch tối ưu cho Kaggle GPU.
        Hỗ trợ resume nếu file output đã có kết quả trước đó.
        """
        if not input_jsonl_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file input: {input_jsonl_path}")

        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Đọc dữ liệu
        items = []
        with open(input_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
                if limit and len(items) >= limit:
                    break

        print(f"\n[Batch Runner] Tổng số mẫu nạp: {len(items)} từ {input_jsonl_path.name}")

        # 2. Kiểm tra các mẫu đã xử lý (Resume support)
        processed_ids = set()
        if output_jsonl_path.exists():
            with open(output_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                            processed_ids.add(rec["id"])
                        except Exception:
                            pass
            if processed_ids:
                print(f"-> Đã tìm thấy {len(processed_ids)} mẫu đã xử lý trước đó. Bỏ qua để chạy tiếp.")

        remaining_items = [it for it in items if it["id"] not in processed_ids]
        print(f"-> Còn lại {len(remaining_items)} mẫu cần xử lý theo batch (batch_size={self.config.batch_size})...")

        # 3. Chạy inference theo batch
        gold_texts = []
        pred_dicts = []
        sample_ids = []

        batch_size = self.config.batch_size
        start_time = time.time()

        with open(output_jsonl_path, "a", encoding="utf-8") as out_f:
            for b_idx in range(0, len(remaining_items), batch_size):
                chunk = remaining_items[b_idx : b_idx + batch_size]
                
                # Gom đường dẫn audio và context_hints
                audio_paths = []
                hints_list = []
                valid_chunk = []

                for it in chunk:
                    rel_audio = it.get("audio")
                    if not rel_audio:
                        continue
                    full_audio_path = audio_base_dir / rel_audio
                    if not full_audio_path.exists():
                        # Thử tìm trực tiếp nếu audio_base_dir đã chứa rel_audio
                        alt_path = Path(rel_audio)
                        if alt_path.exists():
                            full_audio_path = alt_path
                        else:
                            print(f"[Cảnh báo] File audio không tồn tại: {full_audio_path}")
                            continue

                    audio_paths.append(full_audio_path)
                    hints_list.append(extract_context_hints_from_item(it))
                    valid_chunk.append(it)

                if not audio_paths:
                    continue

                # ASR Batch Inference trên GPU
                raw_transcripts = self.transcriber.transcribe_batch(
                    audio_paths=audio_paths,
                    context_hints_list=hints_list,
                    batch_size=batch_size
                )

                # Post-Processing 3 Tầng theo Batch
                processed_results = self.post_processor.process_batch(
                    texts=raw_transcripts,
                    context_hints_list=hints_list
                )

                # Ghi ra output và thu thập metrics nếu có gold text
                for it, p_res in zip(valid_chunk, processed_results):
                    sample_id = it["id"]
                    final_transcript = p_res["final"]

                    # Xử lý history audios nếu có yêu cầu
                    history_transcripts = []
                    if transcribe_history and "history" in it and isinstance(it["history"], list):
                        hist_audios = []
                        hist_indices = []
                        for h_idx, h_turn in enumerate(it["history"]):
                            h_rel = h_turn.get("audio")
                            if h_rel:
                                h_path = audio_base_dir / h_rel
                                if h_path.exists():
                                    hist_audios.append(h_path)
                                    hist_indices.append(h_idx)

                        if hist_audios:
                            h_raw = self.transcriber.transcribe_batch(hist_audios, batch_size=batch_size)
                            h_processed = self.post_processor.process_batch(h_raw)
                            for h_i, h_res in zip(hist_indices, h_processed):
                                history_transcripts.append({
                                    "history_index": h_i,
                                    "transcript": h_res["final"]
                                })

                    record = {
                        "id": sample_id,
                        "transcript": final_transcript,
                        "history_user_transcripts": history_transcripts,
                        "raw_transcript": p_res["raw"],
                        "t1_transcript": p_res["t1_lexicon"],
                        "t2_transcript": p_res["t2_context"],
                        "t3_transcript": p_res["t3_punct"]
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()

                    if "audio_text" in it and it["audio_text"]:
                        gold_texts.append(it["audio_text"])
                        pred_dicts.append(p_res)
                        sample_ids.append(sample_id)

                done_so_far = min(b_idx + batch_size, len(remaining_items))
                print(f"  -> Đã xử lý {done_so_far}/{len(remaining_items)} mẫu ({done_so_far*100//len(remaining_items)}%)...")

        elapsed_total = time.time() - start_time
        print(f"\n[Batch Runner] Hoàn thành xử lý {len(remaining_items)} mẫu trong {elapsed_total:.2f}s!")
        print(f"[Batch Runner] File kết quả đã ghi tại: {output_jsonl_path}")

        # 4. Đánh giá chi tiết nếu có gold label
        eval_result = {}
        if gold_texts:
            print("\n" + "=" * 65)
            print(" BẮT ĐẦU ĐÁNH GIÁ CHI TIẾT (WER, CER, TOKEN-F1, ABLATION)")
            print("=" * 65)

            # Đánh giá kết quả cuối cùng
            final_hyps = [d["final"] for d in pred_dicts]
            metrics, sample_details = self.evaluator.evaluate_predictions(
                references=gold_texts,
                hypotheses=final_hyps,
                sample_ids=sample_ids,
                elapsed_time_sec=elapsed_total
            )

            # Đánh giá bóc tách (Ablation study)
            ablation = self.evaluator.evaluate_ablation(
                references=gold_texts,
                processed_dicts=pred_dicts,
                sample_ids=sample_ids
            )

            # Xuất báo cáo Markdown & JSON
            report_md_path = output_jsonl_path.parent / f"{output_jsonl_path.stem}_eval_report.md"
            self.evaluator.generate_markdown_report(
                metrics=metrics,
                ablation=ablation,
                sample_details=sample_details,
                output_path=report_md_path
            )

            metrics_json_path = output_jsonl_path.parent / f"{output_jsonl_path.stem}_metrics.json"
            with open(metrics_json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "overall": asdict(metrics),
                    "ablation": {k: asdict(v) for k, v in ablation.items()}
                }, f, ensure_ascii=False, indent=2)

            # In tóm tắt ra console
            print("\n" + "-" * 65)
            print(" KẾT QUẢ TỔNG QUAN")
            print("-" * 65)
            print(f"  * WER:           {metrics.wer}% (S={metrics.substitutions}, D={metrics.deletions}, I={metrics.insertions})")
            print(f"  * CER:           {metrics.cer}%")
            print(f"  * Token-F1:      {metrics.token_f1}%")
            print(f"  * Exact Match:   {metrics.exact_match}%")
            print(f"  * Throughput:    {metrics.throughput_samples_per_sec} mẫu/giây")
            print("-" * 65)
            print(" SO SÁNH CÁC TẦNG HẬU XỬ LÝ (ABLATION):")
            for k, v in ablation.items():
                print(f"  * {k:<28}: WER={v.wer:>5.2f}% | CER={v.cer:>5.2f}% | Token-F1={v.token_f1:>5.2f}%")
            print("-" * 65)

            eval_result = {"metrics": metrics, "ablation": ablation}

        return eval_result

