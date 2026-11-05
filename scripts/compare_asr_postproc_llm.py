"""
compare_asr_postproc_llm.py — Kiểm thử đối chiếu trực tiếp tác động của ASR và Post-Processing tới LLM Agent:
1. Luồng 1 (ASR Raw -> LLM): Văn bản thô từ ASR đưa trực tiếp vào LLM sinh tool_calls & state_hash.
2. Luồng 2 (ASR + Post-Processing -> LLM): Văn bản ASR sau khi qua hậu xử lý đưa vào LLM.
3. Luồng 3 (Gold Text -> LLM): Dùng văn bản chuẩn của BTC để đối chứng trần điểm số (Upper bound).

HỖ TRỢ TOÀN DIỆN CHO MÔI TRƯỜNG KAGGLE / LOCAL:
- Đọc cấu hình linh hoạt từ .env (ASR_MODEL, LLM_BASE_URL, CONCURRENCY, v.v.).
- Tối ưu GPU BATCHING cho ASR: Gom nhiều file audio vào 1 lượt gỡ băng (batch_size=16/32).
- Tối ưu CPU/RAM & vLLM: Hỗ trợ chạy SONG SONG (CONCURRENCY) qua ThreadPoolExecutor (tận dụng Continuous Batching).
- Tùy chọn luồng kiểm thử: --eval_stream [all, postproc, postproc_gold, raw, gold] để tiết kiệm thời gian.
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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional
import soundfile as sf
from tqdm import tqdm

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
from src.asr.post_processor import get_post_processor
from src.asr.config import ASRConfig, get_asr_config, get_llm_config
from local_eval import load_fuzzy_map, evaluate_sample


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
    raise FileNotFoundError("Không tìm thấy DATA_ROOT! Hãy chỉ định --data_root hợp lệ hoặc cấu hình trong .env.")


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
    llm_cfg = get_llm_config()
    asr_cfg_env = get_asr_config()

    parser = argparse.ArgumentParser(description="So sánh tác động của ASR Raw vs ASR + Post-Processing tới LLM Agent (Kèm Batching & Concurrency)")
    parser.add_argument("--data_root", type=str, default=None, help="Thư mục train+public_test (mặc định lấy từ .env)")
    parser.add_argument("--eval_file", type=str, default=asr_cfg_env.benchmark_file, help="File dữ liệu kiểm thử (JSONL)")
    parser.add_argument("--sample_id", type=str, default=None, help="Chạy thử nghiệm trên 1 ID duy nhất (VD: train_1364)")
    parser.add_argument("--limit", type=int, default=None, help="Số lượng mẫu kiểm thử (Bỏ trống để chạy toàn bộ file)")
    parser.add_argument("--mode", choices=["server", "hf"], default=llm_cfg.mode, help=f"Chế độ LLM: server hoặc hf (Mặc định: {llm_cfg.mode})")
    parser.add_argument("--base_url", type=str, default=llm_cfg.base_url, help=f"Base URL server LLM (Mặc định: {llm_cfg.base_url})")
    parser.add_argument("--api_key", type=str, default=llm_cfg.api_key, help="API Key nếu cần")
    parser.add_argument("--model", type=str, default=llm_cfg.model, help=f"Tên model LLM (Mặc định: {llm_cfg.model})")
    parser.add_argument("--asr_engine", choices=["phowhisper", "phoasr", "faster_whisper", "whisper"], default=asr_cfg_env.asr_engine, help="Engine ASR")
    parser.add_argument("--asr_model", type=str, default=asr_cfg_env.model_name_or_path, help="Tên hoặc checkpoint model ASR")
    parser.add_argument("--asr_batch_size", type=int, default=asr_cfg_env.batch_size, help="Batch size xử lý âm thanh trên GPU (Mặc định: 16)")
    parser.add_argument("--postproc_mode", choices=["aggressive", "lexicon", "off"], default=llm_cfg.postproc_mode, help="Chế độ Hậu xử lý ASR")
    parser.add_argument("--concurrency", "--workers", dest="concurrency", type=int, default=llm_cfg.concurrency,
                        help=f"Số luồng song song gọi LLM (Mặc định: {llm_cfg.concurrency})")
    parser.add_argument("--eval_stream", choices=["all", "postproc", "postproc_gold", "raw", "gold"], default=llm_cfg.eval_streams,
                        help="Luồng kiểm thử: 'all' (cả 3), 'postproc' (chỉ PostProc), 'postproc_gold', 'raw', 'gold'")
    parser.add_argument("--batch_asr", action="store_true", default=llm_cfg.batch_asr,
                        help="Gom lô toàn bộ file audio gỡ băng 1 lần trên GPU trước khi gọi LLM (nhanh gấp 5-10x)")
    parser.add_argument("--report_file", type=str, default="data/compare_asr_postproc_report.md", help="Đường dẫn file báo cáo Markdown")
    parser.add_argument("--json_file", type=str, default="data/compare_asr_postproc_benchmark.json", help="Đường dẫn file JSON metrics")
    args = parser.parse_args()

    root = resolve_data_root(args.data_root)
    split_dir = root / "train"
    catalog_file = split_dir / "train_tools_catalog.json"
    cache_path = PROJECT_ROOT / "data" / "bm25_train_index.pkl"

    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv

    fuzzy_map = load_fuzzy_map(catalog_file)

    print("=" * 84, flush=True)
    print(" SO SÁNH ĐỐI CHIẾU: ASR RAW vs ASR + POST-PROCESSING -> LLM AGENT", flush=True)
    print(" (TÍCH HỢP GPU BATCH ASR & CONCURRENT CONTINUOUS BATCHING)", flush=True)
    print("=" * 84, flush=True)
    print(f"  - LLM Mode:         {args.mode} ({args.model})", flush=True)
    if args.mode == "server":
        print(f"  - LLM Base URL:     {args.base_url}", flush=True)
    print(f"  - LLM Concurrency:  {args.concurrency} workers (Song song tối ưu RAM/vLLM)", flush=True)
    print(f"  - Eval Streams:     {args.eval_stream.upper()}", flush=True)
    print(f"  - ASR Engine:       {args.asr_engine} ({args.asr_model}) [Batch Size: {args.asr_batch_size}]", flush=True)
    print(f"  - Batch ASR Pass:   {'BẬT (Gom lô GPU)' if args.batch_asr else 'TẮT (Gỡ băng từng file)'}", flush=True)
    print(f"  - PostProc Mode:    {args.postproc_mode.upper()}", flush=True)
    print(f"  - Data Root:        {root}", flush=True)
    print("-" * 84, flush=True)

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
        model_name_or_path=args.asr_model,
        batch_size=args.asr_batch_size,
        device="auto"
    )
    transcriber = UnifiedASRTranscriber(asr_cfg)
    post_processor = get_post_processor(mode=args.postproc_mode, llm_client=llm_client)
    print(f"[3/4] Khởi tạo Post-Processor thành công ({args.postproc_mode})...", flush=True)

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

    total_samples = len(items)
    print(f"-> Sẵn sàng kiểm thử trên {total_samples} mẫu.", flush=True)
    print("=" * 84, flush=True)

    # 5. TIỀN XỬ LÝ ASR: Gom lô toàn bộ file audio gỡ băng trước bằng GPU
    precomputed_asr: Dict[str, Dict[str, Any]] = {}
    total_audio_duration = 0.0
    total_time_asr = 0.0
    total_time_postproc = 0.0

    audio_paths_to_batch = []
    items_with_audio = []

    for item in items:
        rel_audio = item.get("audio", "")
        audio_path = split_dir / rel_audio
        if not audio_path.exists():
            audio_path = root / rel_audio

        dur = 0.0
        if audio_path.exists():
            try:
                info = sf.info(str(audio_path))
                dur = info.duration
            except Exception:
                pass
            audio_paths_to_batch.append(audio_path)
            items_with_audio.append(item)
        else:
            precomputed_asr[item["id"]] = {
                "raw_text": item.get("audio_text", ""),
                "postproc_text": item.get("audio_text", ""),
                "audio_duration_sec": 0.0,
                "t_asr": 0.0,
                "rtf": 0.0,
                "t_post_ms": 0.0
            }
        total_audio_duration += dur

    if args.batch_asr and audio_paths_to_batch:
        print(f"\n⚡ [GPU Batch ASR] Đang gỡ băng hàng loạt {len(audio_paths_to_batch)} file audio (Batch size: {args.asr_batch_size})...", flush=True)
        t_asr_start = time.time()
        raw_transcripts = transcriber.transcribe_batch(audio_paths_to_batch, batch_size=args.asr_batch_size)
        t_asr_total = time.time() - t_asr_start
        total_time_asr = t_asr_total

        avg_asr_per_sample = t_asr_total / len(audio_paths_to_batch)
        print(f"   -> Hoàn thành gỡ băng {len(audio_paths_to_batch)} file trong {t_asr_total:.2f}s (TB: {avg_asr_per_sample:.3f}s / file)!", flush=True)

        for it, a_path, raw_txt in zip(items_with_audio, audio_paths_to_batch, raw_transcripts):
            dur = 0.0
            try:
                info = sf.info(str(a_path))
                dur = info.duration
            except Exception:
                pass

            t_post_start = time.time()
            post_txt = post_processor.process(raw_txt) if args.postproc_mode != "off" else raw_txt
            t_post_ms = (time.time() - t_post_start) * 1000
            total_time_postproc += (t_post_ms / 1000)

            rtf = (avg_asr_per_sample / dur) if dur > 0 else 0.0
            precomputed_asr[it["id"]] = {
                "raw_text": raw_txt,
                "postproc_text": post_txt,
                "audio_duration_sec": dur,
                "t_asr": avg_asr_per_sample,
                "rtf": rtf,
                "t_post_ms": t_post_ms
            }

    # 6. HÀM THỰC THI 1 MẪU
    def process_sample(item: Dict[str, Any], idx_display: int) -> Dict[str, Any]:
        sample_id = item["id"]
        gold_text = item.get("audio_text", "")
        gold_calls = item.get("label", {}).get("tool_calls", [])

        # Lấy kết quả ASR
        if sample_id in precomputed_asr:
            asr_info = precomputed_asr[sample_id]
            raw_text = asr_info["raw_text"]
            postproc_text = asr_info["postproc_text"]
            audio_duration_sec = asr_info["audio_duration_sec"]
            t_asr = asr_info["t_asr"]
            rtf = asr_info["rtf"]
            t_post_ms = asr_info["t_post_ms"]
        else:
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

            t0 = time.time()
            raw_text = transcriber.transcribe(audio_path) if audio_path.exists() else gold_text
            t_asr = time.time() - t0
            rtf = (t_asr / audio_duration_sec) if audio_duration_sec > 0 else 0.0

            t_p0 = time.time()
            postproc_text = post_processor.process(raw_text) if args.postproc_mode != "off" else raw_text
            t_post_ms = (time.time() - t_p0) * 1000

        # Tính Gold State Hash chuẩn
        gold_env = ToolEnv("train", sample_id, root=root)
        for c in gold_calls:
            gold_env.call(c["name"], c.get("arguments") or {})
        hash_gold = gold_env.get_state_hash()

        # Thực thi các luồng theo cấu hình args.eval_stream
        res_raw = {"latency_sec": 0.0, "pred_calls": [], "hash_sys": None}
        score_raw = 0.0
        match_raw = False

        res_post = {"latency_sec": 0.0, "pred_calls": [], "hash_sys": None}
        score_post = 0.0
        match_post = False

        res_gold = {"latency_sec": 0.0, "pred_calls": [], "hash_sys": None}
        score_gold = 0.0
        match_gold = False

        # Luồng 1: ASR Raw
        if args.eval_stream in ("all", "raw"):
            res_raw = run_pipeline_for_text(agent, item, raw_text, split="train", root=root, desc="Raw")
            score_raw = evaluate_sample(res_raw["pred_calls"], gold_calls, res_raw["hash_sys"], hash_gold, fuzzy_map) * 100
            match_raw = (res_raw["hash_sys"] == hash_gold)

        # Luồng 2: ASR + PostProc
        if args.eval_stream in ("all", "postproc", "postproc_gold"):
            res_post = run_pipeline_for_text(agent, item, postproc_text, split="train", root=root, desc="PostProc")
            score_post = evaluate_sample(res_post["pred_calls"], gold_calls, res_post["hash_sys"], hash_gold, fuzzy_map) * 100
            match_post = (res_post["hash_sys"] == hash_gold)

        # Luồng 3: Gold Text
        if args.eval_stream in ("all", "gold", "postproc_gold"):
            res_gold = run_pipeline_for_text(agent, item, gold_text, split="train", root=root, desc="Gold")
            score_gold = evaluate_sample(res_gold["pred_calls"], gold_calls, res_gold["hash_sys"], hash_gold, fuzzy_map) * 100
            match_gold = (res_gold["hash_sys"] == hash_gold)

        e2e_raw = t_asr + res_raw["latency_sec"]
        e2e_post = t_asr + (t_post_ms / 1000) + res_post["latency_sec"]

        return {
            "id": sample_id,
            "idx": idx_display,
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
            "match_raw": match_raw,
            "match_postproc": match_post,
            "match_gold": match_gold,
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

    # 7. ĐIỀU PHỐI CHẠY (TUẦN TỰ HOẶC SONG SONG)
    results = []
    total_score_gold = 0.0
    total_score_raw = 0.0
    total_score_postproc = 0.0

    state_match_gold = 0
    state_match_raw = 0
    state_match_postproc = 0

    total_time_llm_raw = 0.0
    total_time_llm_post = 0.0
    total_time_llm_gold = 0.0

    print_lock = threading.Lock()
    completed_count = 0

    if args.concurrency > 1 and total_samples > 1:
        print(f"\n🚀 Đang chạy {total_samples} mẫu song song với {args.concurrency} worker threads...", flush=True)
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_idx = {executor.submit(process_sample, it, i): i for i, it in enumerate(items, 1)}

            for fut in as_completed(future_to_idx):
                res = fut.result()
                with print_lock:
                    completed_count += 1
                    results.append(res)
                    total_score_raw += res["score_raw"]
                    total_score_postproc += res["score_postproc"]
                    total_score_gold += res["score_gold"]

                    if res["match_raw"]:
                        state_match_raw += 1
                    if res["match_postproc"]:
                        state_match_postproc += 1
                    if res["match_gold"]:
                        state_match_gold += 1

                    total_time_llm_raw += res["timing"]["t_llm_raw_sec"]
                    total_time_llm_post += res["timing"]["t_llm_post_sec"]
                    total_time_llm_gold += res["timing"]["t_llm_gold_sec"]

                    cur_avg_post = total_score_postproc / completed_count
                    cur_match_rate = (state_match_postproc / completed_count) * 100
                    status_str = f"Post: {res['score_postproc']:.1f} ({'✅' if res['match_postproc'] else '❌'})"
                    if args.eval_stream in ("all", "postproc_gold"):
                        status_str += f" | Gold: {res['score_gold']:.1f}"

                    print(f"[{completed_count:03d}/{total_samples:03d}] {res['id']:<10} | {status_str} | TB: {cur_avg_post:.2f}/100 ({cur_match_rate:.1f}%)", flush=True)
    else:
        # Chạy tuần tự với log chi tiết từng bước
        for idx, item in enumerate(items, 1):
            sample_id = item["id"]
            print(f"\n[{idx}/{total_samples}] 🚀 BẮT ĐẦU XỬ LÝ MẪU: {sample_id}...", flush=True)
            res = process_sample(item, idx)
            results.append(res)
            completed_count += 1

            total_score_raw += res["score_raw"]
            total_score_postproc += res["score_postproc"]
            total_score_gold += res["score_gold"]

            if res["match_raw"]:
                state_match_raw += 1
            if res["match_postproc"]:
                state_match_postproc += 1
            if res["match_gold"]:
                state_match_gold += 1

            total_time_llm_raw += res["timing"]["t_llm_raw_sec"]
            total_time_llm_post += res["timing"]["t_llm_post_sec"]
            total_time_llm_gold += res["timing"]["t_llm_gold_sec"]

            print(f"   [1/3] ASR Raw:     \"{res['raw_text']}\" (Thời gian: {res['timing']['t_asr_sec']}s, RTF={res['timing']['rtf']})", flush=True)
            print(f"   [2/3] + PostProc:  \"{res['postproc_text']}\" (Thời gian: {res['timing']['t_postproc_ms']}ms)", flush=True)
            print(f"   [3/3] Gold Label:  \"{res['gold_text']}\"", flush=True)
            if args.eval_stream in ("all", "raw"):
                print(f"         -> Điểm Raw: {res['score_raw']:.2f}/100 | Hash: {'✅' if res['match_raw'] else '❌'} | E2E: {res['timing']['e2e_raw_sec']}s", flush=True)
            if args.eval_stream in ("all", "postproc", "postproc_gold"):
                print(f"         -> Điểm Post: {res['score_postproc']:.2f}/100 | Hash: {'✅' if res['match_postproc'] else '❌'} | E2E: {res['timing']['e2e_post_sec']}s", flush=True)
            if args.eval_stream in ("all", "gold", "postproc_gold"):
                print(f"         -> Điểm Gold: {res['score_gold']:.2f}/100 | Hash: {'✅' if res['match_gold'] else '❌'}", flush=True)

    # 8. BẢNG TỔNG HỢP CUỐI CÙNG
    results.sort(key=lambda r: r.get("idx", 0))
    n = max(total_samples, 1)
    avg_raw = total_score_raw / n
    avg_post = total_score_postproc / n
    avg_gold = total_score_gold / n

    avg_audio = total_audio_duration / n
    avg_asr = total_time_asr / n if total_time_asr > 0 else (sum(r["timing"]["t_asr_sec"] for r in results) / n)
    avg_rtf = (avg_asr / avg_audio) if avg_audio > 0 else 0.0
    avg_post_ms = (total_time_postproc / n) * 1000 if total_time_postproc > 0 else (sum(r["timing"]["t_postproc_ms"] for r in results) / n)
    avg_llm_raw = total_time_llm_raw / n
    avg_llm_post = total_time_llm_post / n
    avg_llm_gold = total_time_llm_gold / n
    avg_e2e_raw = avg_asr + avg_llm_raw
    avg_e2e_post = avg_asr + (avg_post_ms / 1000) + avg_llm_post

    print("\n" + "=" * 84, flush=True)
    print(" BẢNG TỔNG HỢP BENCHMARK: ĐIỂM SỐ & THỜI GIAN THỰC THI (ACCURACY & LATENCY)", flush=True)
    print("=" * 84, flush=True)
    print(f"{'Luồng thực thi':<30} | {'Khớp State':<12} | {'Điểm (/100)':<12} | {'Độ trễ TB (E2E)':<16} | {'Throughput'}")
    print("-" * 84, flush=True)
    if args.eval_stream in ("all", "raw"):
        print(f"{'1. ASR Raw -> LLM':<30} | {state_match_raw/n*100:>5.1f}% ({state_match_raw}/{n}) | {avg_raw:>8.2f} / 100 | {avg_e2e_raw:>7.2f}s / mẫu   | {1/avg_e2e_raw:>5.2f} mẫu/s")
    if args.eval_stream in ("all", "postproc", "postproc_gold"):
        print(f"{'2. ASR + PostProc -> LLM':<30} | {state_match_postproc/n*100:>5.1f}% ({state_match_postproc}/{n}) | {avg_post:>8.2f} / 100 | {avg_e2e_post:>7.2f}s / mẫu   | {1/avg_e2e_post:>5.2f} mẫu/s")
    if args.eval_stream in ("all", "gold", "postproc_gold"):
        print(f"{'3. Gold Text -> LLM (Trần)':<30} | {state_match_gold/n*100:>5.1f}% ({state_match_gold}/{n}) | {avg_gold:>8.2f} / 100 | {avg_llm_gold:>7.2f}s / mẫu   | {1/avg_llm_gold:>5.2f} mẫu/s")
    print("-" * 84, flush=True)
    print("⏱️ CHI TIẾT THỜI GIAN BÌNH QUÂN TỪNG CÔNG ĐOẠN:", flush=True)
    print(f"  * Thời lượng Audio trung bình:       {avg_audio:.2f}s", flush=True)
    print(f"  * Thời gian gỡ băng ASR trung bình:  {avg_asr:.3f}s (RTF = {avg_rtf:.2f} — Nhanh gấp {1/avg_rtf:.1f}x thời gian thực)" if avg_rtf > 0 else f"  * Thời gian gỡ băng ASR: {avg_asr:.3f}s", flush=True)
    print(f"  * Thời gian Post-Processing:         {avg_post_ms:.2f}ms (Gần như 0s, không tốn tài nguyên)", flush=True)
    print(f"  * Thời gian suy luận LLM Agent:      {avg_llm_post:.2f}s / lượt", flush=True)
    print("=" * 84, flush=True)

    if args.eval_stream == "all":
        diff = avg_post - avg_raw
        if diff > 0:
            print(f"🎉 KẾT LUẬN: Post-Processing GIÚP TĂNG +{diff:.2f} ĐIỂM với độ trễ cộng thêm chỉ {avg_post_ms:.2f}ms!", flush=True)
        elif diff < 0:
            print(f"⚠️ KẾT LUẬN: Post-Processing LÀM GIẢM {diff:.2f} ĐIỂM của LLM Agent (Cần bỏ các từ đồng nghĩa)!", flush=True)

    # 9. Xuất file JSON metrics đầy đủ
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

    # 10. Xuất báo cáo Markdown
    rep_path = Path(args.report_file) if Path(args.report_file).is_absolute() else PROJECT_ROOT / args.report_file
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Benchmark Đối Chiếu: ASR Raw vs ASR + PostProc -> LLM Agent\n\n")
        f.write(f"- **Tổng số mẫu kiểm thử:** {n}\n")
        f.write(f"- **Mô hình LLM:** `{args.model}` ({args.mode})\n")
        f.write(f"- **Engine ASR:** `{args.asr_engine}` (`{args.asr_model}`)\n")
        f.write(f"- **Concurrency:** {args.concurrency} workers\n\n")
        f.write("## 1. Bảng Tổng Hợp Độ Chính Xác & Thời Gian Thực Thi\n\n")
        f.write("| Luồng thực thi | Khớp State Hash | Điểm số (/100) | Độ trễ TB (E2E) | Tốc độ xử lý |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        if args.eval_stream in ("all", "raw"):
            f.write(f"| **1. ASR Raw $\\rightarrow$ LLM** | {state_match_raw/n*100:.1f}% ({state_match_raw}/{n}) | **{avg_raw:.2f}** | {avg_e2e_raw:.2f}s | {1/avg_e2e_raw:.2f} mẫu/s |\n")
        if args.eval_stream in ("all", "postproc", "postproc_gold"):
            f.write(f"| **2. ASR + PostProc $\\rightarrow$ LLM** | {state_match_postproc/n*100:.1f}% ({state_match_postproc}/{n}) | **{avg_post:.2f}** | {avg_e2e_post:.2f}s | {1/avg_e2e_post:.2f} mẫu/s |\n")
        if args.eval_stream in ("all", "gold", "postproc_gold"):
            f.write(f"| **3. Gold Text $\\rightarrow$ LLM (Upper bound)** | {state_match_gold/n*100:.1f}% ({state_match_gold}/{n}) | **{avg_gold:.2f}** | {avg_llm_gold:.2f}s | {1/avg_llm_gold:.2f} mẫu/s |\n\n")

    print(f"[Báo cáo] Đã lưu file Markdown chi tiết tại: {rep_path}\n", flush=True)


if __name__ == "__main__":
    main()
