"""
compare_asr_postproc_llm.py — Kiểm thử đối chiếu trực tiếp tác động của ASR và Post-Processing tới LLM Agent:
1. Luồng 1 (ASR Raw -> LLM): Văn bản thô từ ASR đưa trực tiếp vào LLM sinh tool_calls & state_hash.
2. Luồng 2 (ASR + Post-Processing -> LLM): Văn bản ASR sau khi qua hậu xử lý đưa vào LLM.
3. Luồng 3 (Gold Text -> LLM): Dùng văn bản chuẩn của BTC để đối chứng trần điểm số (Upper bound).

Đo lường chính xác theo công thức chấm điểm chính thức của ZAC 2026 LifeMate:
- Khớp State Hash (Cổng sinh tử)
- F1-Score của Tool Calls và Tham số (Fuzzy & Non-fuzzy)
- Điểm số tổng hợp trên thang điểm 100.0 của BTC
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

    # Khởi tạo ToolEnv độc lập cho lượt chạy này
    env = ToolEnv(split, run_item["id"], root=root)
    pred_calls, hash_sys = agent.run(env, run_item, text)
    has_rejected = any("error" in c.get("result", {}) for c in env.log)

    return {
        "text": text,
        "pred_calls": pred_calls,
        "hash_sys": hash_sys,
        "has_rejected": has_rejected
    }


def main():
    parser = argparse.ArgumentParser(description="So sánh tác động của ASR Raw vs ASR + Post-Processing tới LLM Agent")
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
    args = parser.parse_args()

    root = resolve_data_root(args.data_root)
    split_dir = root / "train"
    catalog_file = split_dir / "train_tools_catalog.json"
    cache_path = PROJECT_ROOT / "data" / "bm25_train_index.pkl"

    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv

    fuzzy_map = load_fuzzy_map(catalog_file)

    print("=" * 72, flush=True)
    print(" SO SÁNH ĐỐI CHIẾU: ASR RAW vs ASR + POST-PROCESSING -> LLM AGENT", flush=True)
    print("=" * 72, flush=True)
    print(f"  - LLM Mode:         {args.mode} ({args.model})", flush=True)
    if args.mode == "server":
        print(f"  - LLM Base URL:     {args.base_url}", flush=True)
    print(f"  - ASR Engine:       {args.asr_engine} ({args.asr_model or 'Default'})", flush=True)
    print(f"  - Data Root:        {root}", flush=True)
    print("-" * 72, flush=True)

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
    print("=" * 72, flush=True)

    # 5. Vòng lặp chạy so sánh chi tiết
    results = []
    total_score_gold = 0.0
    total_score_raw = 0.0
    total_score_postproc = 0.0

    state_match_gold = 0
    state_match_raw = 0
    state_match_postproc = 0

    for idx, item in enumerate(items, 1):
        sample_id = item["id"]
        gold_text = item.get("audio_text", "")
        gold_calls = item.get("label", {}).get("tool_calls", [])

        print(f"\n[{idx}/{len(items)}] 🚀 BẮT ĐẦU XỬ LÝ MẪU: {sample_id}...", flush=True)

        # Audio file path
        rel_audio = item.get("audio", "")
        audio_path = split_dir / rel_audio
        if not audio_path.exists():
            audio_path = root / rel_audio

        # Bước 5.1: Gỡ băng âm thanh bằng ASR
        print(f"   [1/4] Gỡ băng âm thanh từ file .wav...", flush=True)
        t_asr = time.time()
        if audio_path.exists():
            raw_text = transcriber.transcribe(audio_path)
        else:
            raw_text = gold_text  # Fallback nếu thiếu audio
        postproc_text = post_processor.process(raw_text)
        print(f"         * ASR Raw:     \"{raw_text}\" ({time.time() - t_asr:.2f}s)", flush=True)
        print(f"         * + PostProc:  \"{postproc_text}\"", flush=True)
        print(f"         * Gold Label:  \"{gold_text}\"", flush=True)

        # Bước 5.2: Tính Gold State Hash chuẩn của BTC
        gold_env = ToolEnv("train", sample_id, root=root)
        for c in gold_calls:
            gold_env.call(c["name"], c.get("arguments") or {})
        hash_gold = gold_env.get_state_hash()

        # Bước 5.3: Chạy Luồng 1: ASR Raw -> LLM Agent
        print(f"   [2/4] Gọi LLM Agent với ASR Raw qua {args.base_url}...", flush=True)
        t_llm1 = time.time()
        res_raw = run_pipeline_for_text(agent, item, raw_text, split="train", root=root, desc="Raw ASR")
        score_raw = evaluate_sample(res_raw["pred_calls"], gold_calls, res_raw["hash_sys"], hash_gold, fuzzy_map) * 100
        print(f"         -> Điểm Raw: {score_raw:.2f}/100 | Khớp Hash: {'✅' if res_raw['hash_sys'] == hash_gold else '❌'} ({time.time() - t_llm1:.2f}s)", flush=True)

        # Bước 5.4: Chạy Luồng 2: ASR + Post-Processing -> LLM Agent
        print(f"   [3/4] Gọi LLM Agent với Post-Processing...", flush=True)
        t_llm2 = time.time()
        res_post = run_pipeline_for_text(agent, item, postproc_text, split="train", root=root, desc="ASR + PostProc")
        score_post = evaluate_sample(res_post["pred_calls"], gold_calls, res_post["hash_sys"], hash_gold, fuzzy_map) * 100
        print(f"         -> Điểm PostProc: {score_post:.2f}/100 | Khớp Hash: {'✅' if res_post['hash_sys'] == hash_gold else '❌'} ({time.time() - t_llm2:.2f}s)", flush=True)

        # Bước 5.5: Chạy Luồng 3: Gold Text -> LLM (Chuẩn đối chứng)
        print(f"   [4/4] Gọi LLM Agent với Gold Text (Đối chứng)...", flush=True)
        t_llm3 = time.time()
        res_gold = run_pipeline_for_text(agent, item, gold_text, split="train", root=root, desc="Gold Text")
        score_gold = evaluate_sample(res_gold["pred_calls"], gold_calls, res_gold["hash_sys"], hash_gold, fuzzy_map) * 100
        print(f"         -> Điểm Gold: {score_gold:.2f}/100 | Khớp Hash: {'✅' if res_gold['hash_sys'] == hash_gold else '❌'} ({time.time() - t_llm3:.2f}s)", flush=True)

        # Thống kê tích lũy
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
            "gold_text": gold_text,
            "raw_text": raw_text,
            "postproc_text": postproc_text,
            "gold_calls": gold_calls,
            "raw_calls": res_raw["pred_calls"],
            "postproc_calls": res_post["pred_calls"],
            "score_raw": score_raw,
            "score_postproc": score_post,
            "score_gold": score_gold,
            "match_raw": res_raw["hash_sys"] == hash_gold,
            "match_postproc": res_post["hash_sys"] == hash_gold,
            "match_gold": res_gold["hash_sys"] == hash_gold,
        }
        results.append(rec)

    # 6. Bảng tổng kết chung cuộc
    n = max(len(items), 1)
    avg_raw = total_score_raw / n
    avg_post = total_score_postproc / n
    avg_gold = total_score_gold / n

    print("\n" + "=" * 76, flush=True)
    print(" BẢNG TỔNG KẾT SO SÁNH HIỆU NĂNG LLM AGENT TRÊN TOÀN BỘ MẪU", flush=True)
    print("=" * 76, flush=True)
    print(f"{'Luồng thực thi':<32} | {'Tỷ lệ Khớp State':<18} | {'ĐIỂM TRUNG BÌNH (/100)'}", flush=True)
    print("-" * 76, flush=True)
    print(f"{'1. ASR Raw -> LLM':<32} | {state_match_raw/n*100:>6.1f}% ({state_match_raw}/{n})     | {avg_raw:>8.4f} / 100", flush=True)
    print(f"{'2. ASR + PostProc -> LLM':<32} | {state_match_postproc/n*100:>6.1f}% ({state_match_postproc}/{n})     | {avg_post:>8.4f} / 100", flush=True)
    print(f"{'3. Gold Text -> LLM (Trần tối đa)':<32} | {state_match_gold/n*100:>6.1f}% ({state_match_gold}/{n})     | {avg_gold:>8.4f} / 100", flush=True)
    print("=" * 76, flush=True)

    diff = avg_post - avg_raw
    if diff > 0:
        print(f"🎉 KẾT LUẬN: Post-Processing GIÚP TĂNG +{diff:.2f} ĐIỂM cho LLM Agent!", flush=True)
    elif diff < 0:
        print(f"⚠️ KẾT LUẬN: Post-Processing LÀM GIẢM {diff:.2f} ĐIỂM của LLM Agent!", flush=True)
    else:
        print("ℹ️ KẾT LUẬN: Điểm số ngang bằng nhau giữa Raw và Post-Processing.", flush=True)

    # 7. Xuất báo cáo Markdown
    rep_path = Path(args.report_file) if Path(args.report_file).is_absolute() else PROJECT_ROOT / args.report_file
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Đối Chiếu: ASR Raw vs ASR + PostProc -> LLM Agent\n\n")
        f.write(f"- **Tổng số mẫu:** {n}\n")
        f.write(f"- **Mô hình LLM:** {args.model}\n")
        f.write(f"- **Engine ASR:** {args.asr_engine}\n\n")
        f.write("## 1. Bảng Tổng Hợp Chỉ Số\n\n")
        f.write("| Luồng thực thi | Khớp State Hash (%) | Điểm số trung bình (/100) |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| **1. ASR Raw $\\rightarrow$ LLM** | {state_match_raw/n*100:.1f}% ({state_match_raw}/{n}) | **{avg_raw:.4f}** |\n")
        f.write(f"| **2. ASR + PostProc $\\rightarrow$ LLM** | {state_match_postproc/n*100:.1f}% ({state_match_postproc}/{n}) | **{avg_post:.4f}** |\n")
        f.write(f"| **3. Gold Text $\\rightarrow$ LLM (Upper bound)** | {state_match_gold/n*100:.1f}% ({state_match_gold}/{n}) | **{avg_gold:.4f}** |\n\n")
        f.write("## 2. Chi Tiết Từng Mẫu Kiểm Thử\n\n")
        for r in results:
            f.write(f"### Mẫu: `{r['id']}`\n")
            f.write(f"- **Gold:** `{r['gold_text']}` (Điểm: {r['score_gold']:.1f})\n")
            f.write(f"- **Raw:** `{r['raw_text']}` (Điểm: {r['score_raw']:.1f}, State: {'✅' if r['match_raw'] else '❌'})\n")
            f.write(f"- **PostProc:** `{r['postproc_text']}` (Điểm: {r['score_postproc']:.1f}, State: {'✅' if r['match_postproc'] else '❌'})\n\n")

    print(f"\n[Báo cáo] Đã lưu file Markdown chi tiết tại: {rep_path}\n", flush=True)


if __name__ == "__main__":
    main()
