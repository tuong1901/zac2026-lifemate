"""
compare_asr_postproc_llm.py — Kiểm thử đối chiếu trực tiếp tác động của ASR và Post-Processing tới LLM Agent:
1. Luồng 1 (ASR Raw -> LLM): Văn bản thô từ ASR đưa trực tiếp vào LLM sinh tool_calls & state_hash.
2. Luồng 2 (ASR + Post-Processing -> LLM): Văn bản ASR sau khi qua hậu xử lý đưa vào LLM.
3. Luồng 3 (Gold Text -> LLM): Dùng văn bản chuẩn của BTC để đối chứng trần điểm số (Upper bound).

TÍCH HỢP HỆ THỐNG BENCHMARK THỜI GIAN & ĐỘ TRỄ CHI TIẾT (LATENCY & SPEED BENCHMARK):
- Đo thời lượng audio (.wav) gốc
- Đo thời gian gỡ băng ASR (giây) và chỉ số RTF (Real-Time Factor)
- Đo thời gian xử lý của Post-Processing (mili-giây)
- Đo thời gian suy luận của LLM Agent (giây) và số lượng tool calls sinh ra
- Đo tổng thời gian End-to-End (E2E) từ khi nhận audio đến khi trả về kết quả
- Xuất bảng tổng hợp Markdown và JSON chi tiết
"""

import os
import sys
import copy
import json
import time
import random
import warnings
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import soundfile as sf

# Tắt cảnh báo không cần thiết từ transformers
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Đảm bảo in tiếng Việt chuẩn trên mọi console và Jupyter Notebook
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

SEED = 42
random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.rules.catalog_filter import CatalogFilter
from src.rules.schema_guardrail import SafeSchemaGuardrail
from src.agent.rag.exemplar_retriever import BM25ExemplarRetriever
from src.agent.rag.history_resolver import HistoryResolver
from src.agent.react_loop import ReActAgent
from src.agent.llm_client import get_llm_client
from src.asr.transcribe import UnifiedASRTranscriber
from src.asr.post_processor import LexiconPostProcessor
from src.asr.config import ASRConfig
from local_eval import load_fuzzy_map, evaluate_sample, compute_token_f1


def resolve_data_root(preferred_path: str = None) -> Path:
    candidates = [
        preferred_path,
        os.getenv("DATA_ROOT"),
        "/kaggle/input/datasets/n107hoangtuong/zalo-competion/train+public_test",
        "/kaggle/input/zalo-ai-challenge-2026/train+public_test",
        "/kaggle/input/train+public_test",
        "D:/Download/train+public_test/train+public_test",
        "C:/Users/Admin/Downloads/train+public_test",
        str(PROJECT_ROOT / "data"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError("Không tìm thấy DATA_ROOT! Hãy chỉ định --data_root hợp lệ.")


def run_pipeline_for_text(agent, item: Dict[str, Any], text: str, split: str, root: Path, desc: str = ""):
    from zaic_tools import ToolEnv

    run_item = copy.deepcopy(item)
    run_item["audio_text"] = text

    t_start = time.time()
    env = ToolEnv(split, run_item["id"], root=root)
    pred_calls, hash_sys = agent.run(env, run_item, text)
    elapsed = time.time() - t_start
    has_rejected = any("error" in c.get("result", {}) for c in env.log)

    return {
        "text": text,
        "pred_calls": pred_calls,
        "hash_sys": hash_sys,
        "has_rejected": has_rejected,
        "latency_sec": elapsed,
        "num_calls": len(pred_calls)
    }


def main():
    parser = argparse.ArgumentParser(description="So sánh tác động của ASR Raw vs ASR + Post-Processing tới LLM Agent kèm Benchmark Thời Gian")
    parser.add_argument("--data_root", type=str, default=None, help="Thư mục train+public_test")
    parser.add_argument("--eval_file", type=str, default="data/val_benchmark_200.jsonl", help="File dữ liệu kiểm thử (JSONL)")
    parser.add_argument("--sample_id", type=str, default=None, help="Chạy thử nghiệm trên 1 ID duy nhất (VD: train_1364)")
    parser.add_argument("--limit", type=int, default=5, help="Số lượng mẫu kiểm thử (Mặc định: 5 mẫu để xem nhanh)")
    parser.add_argument("--mode", choices=["server", "hf"], default="server", help="Chế độ LLM: server (vLLM/Ollama/ngrok) hoặc hf (HuggingFace local)")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="Base URL server LLM")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API Key nếu cần")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Tên model LLM")
    parser.add_argument("--asr_engine", choices=["phowhisper", "phoasr", "faster_whisper"], default="phowhisper", help="Engine ASR (Mặc định: phowhisper)")
    parser.add_argument("--asr_model", type=str, default=None, help="Tên hoặc checkpoint của model ASR")
    parser.add_argument("--report_file", type=str, default="data/compare_asr_postproc_report.md", help="Đường dẫn file báo cáo kết quả Markdown")
    parser.add_argument("--json_file", type=str, default="data/compare_asr_postproc_benchmark.json", help="Đường dẫn file JSON lưu chi tiết metrics")
    args = parser.parse_args()

    root = resolve_data_root(args.data_root)
    split_dir = root / "train"
    catalog_file = split_dir / "train_tools_catalog.json"
    cache_path = PROJECT_ROOT / "data" / "bm25_train_index.pkl"

    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv

    fuzzy_map = load_fuzzy_map(catalog_file)

    print("=" * 80, flush=True)
    print(" SO SÁNH ĐỐI CHIẾU: ASR RAW vs ASR + POST-PROCESSING -> LLM AGENT", flush=True)
    print(" (KÈM HỆ THỐNG ĐO LƯỜNG THỜI GIAN VÀ ĐỘ TRỄ CHI TIẾT)", flush=True)
    print("=" * 80, flush=True)
    print(f"  - LLM Mode:         {args.mode} ({args.model})", flush=True)
    if args.mode == "server":
        print(f"  - LLM Base URL:     {args.base_url}", flush=True)
    print(f"  - ASR Engine:       {args.asr_engine} ({args.asr_model or 'Default'})", flush=True)
    print(f"  - Data Root:        {root}", flush=True)
    print("-" * 80, flush=True)

    # 1. Khởi tạo LLM Client
    print("\n[1/4] Khởi tạo LLM Client...", flush=True)
    if args.mode == "hf":
        llm_client = get_llm_client(client_type="hf", model_name_or_path=args.model)
    else:
        llm_client = get_llm_client(
            client_type="openai",
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model
        )

    # 2. Khởi tạo Agent Modules
    print("[2/4] Nạp các module RAG, Guardrail & Catalog...", flush=True)
    catalog_filter = CatalogFilter(str(catalog_file))
    retriever = BM25ExemplarRetriever()
    if cache_path.exists():
        retriever.load_cache(str(cache_path))
    else:
        retriever.build_from_train_file(str(split_dir / "train.jsonl"))

    agent = ReActAgent(
        llm_client=llm_client,
        catalog_filter=catalog_filter,
        exemplar_retriever=retriever,
        history_resolver=HistoryResolver(),
        guardrail=SafeSchemaGuardrail(),
        max_steps=7
    )

    # 3. Khởi tạo ASR Engine & Post-Processor
    print(f"[3/4] Khởi tạo ASR Engine ({args.asr_engine})...", flush=True)
    asr_cfg = ASRConfig(
        asr_engine=args.asr_engine,
        model_name_or_path=args.asr_model or ("vinai/PhoWhisper-medium" if args.asr_engine == "phowhisper" else "Qualcomm-AI-Research/PhoASR-whisper-small"),
        device="auto"
    )
    transcriber = UnifiedASRTranscriber(asr_cfg)
    post_processor = LexiconPostProcessor()

    # 4. Nạp dữ liệu kiểm thử
    print(f"[4/4] Nạp dữ liệu kiểm thử từ {args.eval_file}...", flush=True)
    eval_path = Path(args.eval_file) if Path(args.eval_file).is_absolute() else PROJECT_ROOT / args.eval_file
    if not eval_path.exists():
        eval_path = root / "val_benchmark_200.jsonl"
    if not eval_path.exists():
        eval_path = split_dir / "train.jsonl"

    items = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            it = json.loads(line)
            if args.sample_id:
                if it["id"] == args.sample_id:
                    items.append(it)
                    break
            else:
                items.append(it)
                if args.limit and len(items) >= args.limit:
                    break

    print(f"-> Sẵn sàng kiểm thử trên {len(items)} mẫu.", flush=True)
    print("=" * 80, flush=True)

    # 5. Vòng lặp chạy so sánh chi tiết & đo thời gian
    results = []
    total_score_gold = 0.0
    total_score_raw = 0.0
    total_score_postproc = 0.0

    state_match_gold = 0
    state_match_raw = 0
    state_match_postproc = 0

    total_audio_duration = 0.0
    total_time_asr = 0.0
    total_time_postproc = 0.0
    total_time_llm_raw = 0.0
    total_time_llm_post = 0.0
    total_time_llm_gold = 0.0

    for idx, item in enumerate(items, 1):
        sample_id = item["id"]
        gold_text = item.get("audio_text", "")
        gold_calls = item.get("label", {}).get("tool_calls", [])

        print(f"\n[{idx}/{len(items)}] 🚀 BẮT ĐẦU XỬ LÝ MẪU: {sample_id}...", flush=True)

        # Đường dẫn file audio và đo thời lượng âm thanh
        rel_audio = item.get("audio", "")
        audio_path = split_dir / rel_audio
        if not audio_path.exists():
            audio_path = root / rel_audio

        audio_duration_sec = 0.0
        if audio_path.exists():
            try:
                info = sf.info(str(audio_path))
                audio_duration_sec = info.duration
            except Exception:
                pass
        total_audio_duration += audio_duration_sec

        # Bước 5.1: Gỡ băng âm thanh bằng ASR và đo độ trễ
        print(f"   [1/4] Gỡ băng âm thanh (File: {audio_duration_sec:.2f}s)...", flush=True)
        t_start_asr = time.time()
        if audio_path.exists():
            raw_text = transcriber.transcribe(audio_path)
        else:
            raw_text = gold_text
        t_asr = time.time() - t_start_asr
        total_time_asr += t_asr
        rtf = (t_asr / audio_duration_sec) if audio_duration_sec > 0 else 0.0

        # Bước 5.2: Đo thời gian Hậu xử lý (Post-Processing)
        t_start_post = time.time()
        postproc_text = post_processor.process(raw_text)
        t_post_ms = (time.time() - t_start_post) * 1000
        total_time_postproc += (t_post_ms / 1000)

        print(f"         * ASR Raw:     \"{raw_text}\" (Thời gian: {t_asr:.2f}s, RTF={rtf:.2f})", flush=True)
        print(f"         * + PostProc:  \"{postproc_text}\" (Thời gian: {t_post_ms:.2f}ms)", flush=True)
        print(f"         * Gold Label:  \"{gold_text}\"", flush=True)

        # Bước 5.3: Tính Gold State Hash chuẩn của BTC
        gold_env = ToolEnv("train", sample_id, root=root)
        for c in gold_calls:
            gold_env.call(c["name"], c.get("arguments") or {})
        hash_gold = gold_env.get_state_hash()

        # Bước 5.4: Chạy Luồng 1: ASR Raw -> LLM Agent
        print(f"   [2/4] Gọi LLM Agent với ASR Raw qua {args.base_url}...", flush=True)
        res_raw = run_pipeline_for_text(agent, item, raw_text, split="train", root=root, desc="Raw ASR")
        score_raw = evaluate_sample(res_raw["pred_calls"], gold_calls, res_raw["hash_sys"], hash_gold, fuzzy_map) * 100
        total_time_llm_raw += res_raw["latency_sec"]
        e2e_raw = t_asr + res_raw["latency_sec"]
        print(f"         -> Điểm Raw: {score_raw:.2f}/100 | Hash: {'✅' if res_raw['hash_sys'] == hash_gold else '❌'} | LLM: {res_raw['latency_sec']:.2f}s | E2E: {e2e_raw:.2f}s", flush=True)

        # Bước 5.5: Chạy Luồng 2: ASR + Post-Processing -> LLM Agent
        print(f"   [3/4] Gọi LLM Agent với Post-Processing...", flush=True)
        res_post = run_pipeline_for_text(agent, item, postproc_text, split="train", root=root, desc="ASR + PostProc")
        score_post = evaluate_sample(res_post["pred_calls"], gold_calls, res_post["hash_sys"], hash_gold, fuzzy_map) * 100
        total_time_llm_post += res_post["latency_sec"]
        e2e_post = t_asr + (t_post_ms / 1000) + res_post["latency_sec"]
        print(f"         -> Điểm PostProc: {score_post:.2f}/100 | Hash: {'✅' if res_post['hash_sys'] == hash_gold else '❌'} | LLM: {res_post['latency_sec']:.2f}s | E2E: {e2e_post:.2f}s", flush=True)

        # Bước 5.6: Chạy Luồng 3: Gold Text -> LLM (Chuẩn đối chứng)
        print(f"   [4/4] Gọi LLM Agent với Gold Text (Đối chứng)...", flush=True)
        res_gold = run_pipeline_for_text(agent, item, gold_text, split="train", root=root, desc="Gold Text")
        score_gold = evaluate_sample(res_gold["pred_calls"], gold_calls, res_gold["hash_sys"], hash_gold, fuzzy_map) * 100
        total_time_llm_gold += res_gold["latency_sec"]
        print(f"         -> Điểm Gold: {score_gold:.2f}/100 | Hash: {'✅' if res_gold['hash_sys'] == hash_gold else '❌'} | LLM: {res_gold['latency_sec']:.2f}s", flush=True)

        # Thống kê tích lũy điểm số
        total_score_raw += score_raw
        total_score_postproc += score_post
        total_score_gold += score_gold

        if res_raw["hash_sys"] == hash_gold:
            state_match_raw += 1
        if res_post["hash_sys"] == hash_gold:
            state_match_postproc += 1
        if res_gold["hash_sys"] == hash_gold:
            state_match_gold += 1

        rec = {
            "id": sample_id,
            "audio_duration_sec": round(audio_duration_sec, 2),
            "gold_text": gold_text,
            "raw_text": raw_text,
            "postproc_text": postproc_text,
            "gold_calls": gold_calls,
            "raw_calls": res_raw["pred_calls"],
            "postproc_calls": res_post["pred_calls"],
            "score_raw": round(score_raw, 2),
            "score_postproc": round(score_post, 2),
            "score_gold": round(score_gold, 2),
            "match_raw": res_raw["hash_sys"] == hash_gold,
            "match_postproc": res_post["hash_sys"] == hash_gold,
            "match_gold": res_gold["hash_sys"] == hash_gold,
            "timing": {
                "t_asr_sec": round(t_asr, 3),
                "rtf": round(rtf, 3),
                "t_postproc_ms": round(t_post_ms, 2),
                "t_llm_raw_sec": round(res_raw["latency_sec"], 2),
                "t_llm_post_sec": round(res_post["latency_sec"], 2),
                "t_llm_gold_sec": round(res_gold["latency_sec"], 2),
                "e2e_raw_sec": round(e2e_raw, 2),
                "e2e_post_sec": round(e2e_post, 2)
            }
        }
        results.append(rec)

    # 6. Bảng tổng kết chung cuộc kèm Benchmark thời gian
    n = max(len(items), 1)
    avg_raw = total_score_raw / n
    avg_post = total_score_postproc / n
    avg_gold = total_score_gold / n

    avg_audio = total_audio_duration / n
    avg_asr = total_time_asr / n
    avg_rtf = (avg_asr / avg_audio) if avg_audio > 0 else 0.0
    avg_post_ms = (total_time_postproc / n) * 1000
    avg_llm_raw = total_time_llm_raw / n
    avg_llm_post = total_time_llm_post / n
    avg_e2e_raw = avg_asr + avg_llm_raw
    avg_e2e_post = avg_asr + (avg_post_ms / 1000) + avg_llm_post

    print("\n" + "=" * 84, flush=True)
    print(" BẢNG TỔNG HỢP BENCHMARK: ĐIỂM SỐ & THỜI GIAN THỰC THI (ACCURACY & LATENCY)", flush=True)
    print("=" * 84, flush=True)
    print(f"{'Luồng thực thi':<30} | {'Khớp State':<12} | {'Điểm (/100)':<12} | {'Độ trễ TB (E2E)':<16} | {'Throughput'}")
    print("-" * 84, flush=True)
    print(f"{'1. ASR Raw -> LLM':<30} | {state_match_raw/n*100:>5.1f}% ({state_match_raw}/{n}) | {avg_raw:>8.2f} / 100 | {avg_e2e_raw:>7.2f}s / mẫu   | {1/avg_e2e_raw:>5.2f} mẫu/s")
    print(f"{'2. ASR + PostProc -> LLM':<30} | {state_match_postproc/n*100:>5.1f}% ({state_match_postproc}/{n}) | {avg_post:>8.2f} / 100 | {avg_e2e_post:>7.2f}s / mẫu   | {1/avg_e2e_post:>5.2f} mẫu/s")
    print(f"{'3. Gold Text -> LLM (Trần)':<30} | {state_match_gold/n*100:>5.1f}% ({state_match_gold}/{n}) | {avg_gold:>8.2f} / 100 | {total_time_llm_gold/n:>7.2f}s / mẫu   | {n/total_time_llm_gold:>5.2f} mẫu/s")
    print("-" * 84, flush=True)
    print("⏱️ CHI TIẾT THỜI GIAN BÌNH QUÂN TỪNG CÔNG ĐOẠN:", flush=True)
    print(f"  * Thời lượng Audio trung bình:       {avg_audio:.2f}s", flush=True)
    print(f"  * Thời gian gỡ băng ASR trung bình:  {avg_asr:.3f}s (RTF = {avg_rtf:.2f} — Nhanh gấp {1/avg_rtf:.1f}x thời gian thực)" if avg_rtf > 0 else f"  * Thời gian gỡ băng ASR: {avg_asr:.3f}s", flush=True)
    print(f"  * Thời gian Post-Processing:         {avg_post_ms:.2f}ms (Gần như 0s, không tốn tài nguyên)", flush=True)
    print(f"  * Thời gian suy luận LLM Agent:      {avg_llm_post:.2f}s / lượt", flush=True)
    print("=" * 84, flush=True)

    diff = avg_post - avg_raw
    if diff > 0:
        print(f"🎉 KẾT LUẬN: Post-Processing GIÚP TĂNG +{diff:.2f} ĐIỂM với độ trễ cộng thêm chỉ {avg_post_ms:.2f}ms!", flush=True)
    elif diff < 0:
        print(f"⚠️ KẾT LUẬN: Post-Processing LÀM GIẢM {diff:.2f} ĐIỂM của LLM Agent (Cần bỏ các từ đồng nghĩa)!", flush=True)
    else:
        print("ℹ️ KẾT LUẬN: Điểm số ngang bằng nhau giữa Raw và Post-Processing.", flush=True)

    # 7. Xuất file JSON metrics đầy đủ
    json_path = Path(args.json_file) if Path(args.json_file).is_absolute() else PROJECT_ROOT / args.json_file
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_samples": n,
                "score_raw": round(avg_raw, 2),
                "score_postproc": round(avg_post, 2),
                "score_gold": round(avg_gold, 2),
                "state_match_raw_pct": round(state_match_raw / n * 100, 1),
                "state_match_postproc_pct": round(state_match_postproc / n * 100, 1),
                "avg_audio_duration_sec": round(avg_audio, 2),
                "avg_asr_latency_sec": round(avg_asr, 3),
                "avg_rtf": round(avg_rtf, 3),
                "avg_postproc_latency_ms": round(avg_post_ms, 2),
                "avg_e2e_latency_sec": round(avg_e2e_post, 2),
            },
            "samples": results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[Dữ liệu JSON] Đã lưu file benchmark JSON tại: {json_path}", flush=True)

    # 8. Xuất báo cáo Markdown
    rep_path = Path(args.report_file) if Path(args.report_file).is_absolute() else PROJECT_ROOT / args.report_file
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Benchmark Đối Chiếu: ASR Raw vs ASR + PostProc -> LLM Agent\n\n")
        f.write(f"- **Tổng số mẫu kiểm thử:** {n}\n")
        f.write(f"- **Mô hình LLM:** `{args.model}`\n")
        f.write(f"- **Engine ASR:** `{args.asr_engine}`\n\n")
        f.write("## 1. Bảng Tổng Hợp Độ Chính Xác & Thời Gian Thực Thi\n\n")
        f.write("| Luồng thực thi | Khớp State Hash | Điểm số (/100) | Độ trễ TB (E2E) | Tốc độ xử lý |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **1. ASR Raw $\\rightarrow$ LLM** | {state_match_raw/n*100:.1f}% ({state_match_raw}/{n}) | **{avg_raw:.2f}** | {avg_e2e_raw:.2f}s | {1/avg_e2e_raw:.2f} mẫu/s |\n")
        f.write(f"| **2. ASR + PostProc $\\rightarrow$ LLM** | {state_match_postproc/n*100:.1f}% ({state_match_postproc}/{n}) | **{avg_post:.2f}** | {avg_e2e_post:.2f}s | {1/avg_e2e_post:.2f} mẫu/s |\n")
        f.write(f"| **3. Gold Text $\\rightarrow$ LLM (Upper bound)** | {state_match_gold/n*100:.1f}% ({state_match_gold}/{n}) | **{avg_gold:.2f}** | {total_time_llm_gold/n:.2f}s | {n/total_time_llm_gold:.2f} mẫu/s |\n\n")
        f.write("## 2. Chi Tiết Độ Trễ Từng Thành Phần\n\n")
        f.write(f"- **Audio Duration trung bình:** {avg_audio:.2f}s\n")
        f.write(f"- **ASR Transcription trung bình:** {avg_asr:.3f}s (RTF = {avg_rtf:.2f})\n")
        f.write(f"- **Post-Processing trung bình:** {avg_post_ms:.2f}ms\n")
        f.write(f"- **LLM Agent trung bình:** {avg_llm_post:.2f}s\n\n")
        f.write("## 3. Chi Tiết Từng Mẫu\n\n")
        for r in results:
            f.write(f"### Mẫu: `{r['id']}` (Audio: {r['audio_duration_sec']}s)\n")
            f.write(f"- **Gold:** `{r['gold_text']}` (Điểm: {r['score_gold']:.1f})\n")
            f.write(f"- **Raw:** `{r['raw_text']}` (Điểm: {r['score_raw']:.1f}, Hash: {'✅' if r['match_raw'] else '❌'}, E2E: {r['timing']['e2e_raw_sec']}s)\n")
            f.write(f"- **PostProc:** `{r['postproc_text']}` (Điểm: {r['score_postproc']:.1f}, Hash: {'✅' if r['match_postproc'] else '❌'}, E2E: {r['timing']['e2e_post_sec']}s)\n\n")

    print(f"[Báo cáo] Đã lưu file Markdown chi tiết tại: {rep_path}\n", flush=True)


if __name__ == "__main__":
    main()
