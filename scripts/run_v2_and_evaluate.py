"""
scripts/run_v2_and_evaluate.py — Bộ Thực thi Suy Luận và Đánh Giá Kiểm Định cho Version 2 (Two-Stage Architecture).
Tối ưu hóa riêng cho ZAC 2026:
- Chạy Pipeline 2 Giai Đoạn (Stage 1: Intent & Router -> Stage 2: Targeted ReAct Planner).
- Tự động ghi nhận thời gian từng giai đoạn (Stage 1 Latency, Stage 2 Latency, Total Latency).
- Tự động kiểm định BTC Schema Validation và đóng gói submission_v2.zip.
"""

import os
import sys
import json
import time
import zipfile
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
from tqdm import tqdm

# Đảm bảo in tiếng Việt chuẩn trên Windows / Linux console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.rules.catalog_filter import CatalogFilter
from src.rules.schema_guardrail import SafeSchemaGuardrail
from src.agent.v2.two_stage_agent import TwoStageReActAgent
from src.agent.llm_client import get_llm_client
from src.asr.config import get_llm_config, get_asr_config
from local_eval import load_fuzzy_map, evaluate_sample


def resolve_data_root(custom_path: Optional[str] = None) -> Path:
    candidates = [
        custom_path,
        os.getenv("DATA_ROOT"),
        str(PROJECT_ROOT / "data"),
        "/kaggle/input/datasets/n107hoangtuong/zalo-competion/train+public_test",
        "/kaggle/input/zalo-ai-challenge-2026/train+public_test",
        "D:/Download/train+public_test/train+public_test",
    ]
    for c in candidates:
        if c and Path(c).exists() and (Path(c) / "public_test").exists():
            return Path(c)
    if (PROJECT_ROOT / "data" / "public_test").exists():
        return PROJECT_ROOT / "data"
    raise FileNotFoundError(
        "Không tìm thấy thư mục chứa 'public_test'! Vui lòng truyền đường dẫn qua --data_root."
    )


def validate_submission_file(sub_path: Path, split_dir: Path, catalog_path: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(split_dir))
    try:
        from zaic_tools import ToolEnv, schema_validate
    except ImportError as e:
        print(f"[Cảnh báo] Không thể nạp zaic_tools từ {split_dir}: {e}")
        return {"error": str(e)}

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    stats = {
        "total_lines": 0,
        "valid_lines": 0,
        "schema_errors": 0,
        "error_details": [],
        "execute_calls": 0,
        "ask_missing_calls": 0,
        "out_of_scope_calls": 0,
        "unique_tools_used": set(),
        "valid_hash_count": 0,
    }

    with open(sub_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            stats["total_lines"] += 1
            try:
                row = json.loads(line)
            except Exception as e:
                stats["schema_errors"] += 1
                stats["error_details"].append(f"Dòng {idx}: JSON không hợp lệ ({e})")
                continue

            sid = row.get("id")
            calls = row.get("tool_calls", [])
            shash = row.get("state_hash", "")

            if isinstance(shash, str) and len(shash) in (32, 64):
                stats["valid_hash_count"] += 1

            if not calls:
                stats["out_of_scope_calls"] += 1
                stats["valid_lines"] += 1
                continue

            is_valid = True
            for c in calls:
                name = c.get("name")
                args = c.get("arguments", {})
                stats["unique_tools_used"].add(name)
                if name == "sys.ask_missing_info":
                    stats["ask_missing_calls"] += 1
                else:
                    stats["execute_calls"] += 1

                val_ok = schema_validate(catalog, name, args)
                if not val_ok:
                    is_valid = False
                    stats["schema_errors"] += 1
                    stats["error_details"].append(
                        f"Dòng {idx} ({sid}): Lỗi schema tool '{name}' với args {json.dumps(args, ensure_ascii=False)}"
                    )

            if is_valid:
                stats["valid_lines"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="ZAC 2026 — V2 Two-Stage Inference & Evaluation Pipeline")
    parser.add_argument("--data_root", type=str, default=None, help="Đường dẫn đến thư mục chứa train và public_test")
    parser.add_argument("--split", type=str, default="public_test", choices=["public_test", "train", "val"])
    parser.add_argument("--output", type=str, default=None, help="File kết quả submission_v2.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Số mẫu chạy thử nghiệm (None = toàn bộ)")
    parser.add_argument("--asr_file", type=str, default=None, help="Đường dẫn file ASR transcript có sẵn (ví dụ: output/public_test_transcripts.jsonl)")
    parser.add_argument("--asr_model", type=str, default=None, help="Đường dẫn thư mục hoặc tên model ASR (vd: /kaggle/working/models/phowhisper-medium-finetuned)")
    parser.add_argument("--workers", type=int, default=2, help="Số luồng đồng thời (khuyến nghị 2-3 trên Colab T4)")
    parser.add_argument("--llm_endpoint", type=str, default=None, help="URL OpenAI API / vLLM / Colab ngrok")
    parser.add_argument("--llm_model", type=str, default=None, help="Tên model")
    parser.add_argument("--llm_api_key", type=str, default=None, help="API key")
    parser.add_argument("--max_tokens", type=int, default=256, help="Giới hạn max_tokens (mặc định 256)")
    parser.add_argument("--time_budget", type=float, default=45.0, help="Thời gian tối đa cho 1 mẫu (giây)")
    parser.add_argument("--max_steps", type=int, default=4, help="Số bước ReAct tối đa ở Stage 2")
    parser.add_argument("--resume", action="store_true", default=False, help="Chạy tiếp từ file kết quả đã có, bỏ qua các mẫu đã hoàn thành")
    parser.add_argument("--zip_output", type=str, default="submission_v2.zip", help="File zip nộp bài BTC")
    parser.add_argument("--quiet", action="store_true", help="Giảm log chi tiết từng bước")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    split_dir = data_root / args.split
    input_file = split_dir / f"{args.split}.jsonl"
    catalog_file = split_dir / f"{args.split}_tools_catalog.json"

    if not catalog_file.exists():
        fallback_cat = data_root / "train" / "train_tools_catalog.json"
        if fallback_cat.exists():
            catalog_file = fallback_cat
        else:
            catalog_file = PROJECT_ROOT / "data" / "train" / "train_tools_catalog.json"

    out_file = Path(args.output) if args.output else PROJECT_ROOT / "submission_v2.jsonl"
    if "KAGGLE_WORKING_DIR" in os.environ or Path("/kaggle/working").exists():
        out_file = Path("/kaggle/working") / "submission_v2.jsonl"
        args.zip_output = str(Path("/kaggle/working") / args.zip_output)

    print("=" * 75)
    print("      ZAC 2026 — KIẾN TRÚC HAI GIAI ĐOẠN (V2 TWO-STAGE PIPELINE)")
    print("=" * 75)
    print(f" Thư mục Data Root: {data_root}")
    print(f" Tập dữ liệu:       {args.split} ({input_file})")
    print(f" File Output:       {out_file}")
    print(f" Tool Catalog:      {catalog_file}")

    # Nạp dữ liệu kiểm thử
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if args.limit and args.limit > 0:
        samples = samples[:args.limit]
    print(f"\n📋 Đã nạp {len(samples)} mẫu kiểm thử từ {input_file.name}")

    # Nạp LLM Client
    llm_cfg = get_llm_config()
    endpoint = args.llm_endpoint or llm_cfg.endpoint
    model_name = args.llm_model or llm_cfg.model
    api_key = args.llm_api_key or llm_cfg.api_key
    print(f"🤖 Kết nối LLM tại: {endpoint} [Mô hình: {model_name}]")
    client = get_llm_client(
        base_url=endpoint,
        model=model_name,
        api_key=api_key,
        max_tokens=args.max_tokens,
        timeout=args.time_budget or 60.0,
    )

    # Nạp ToolEnv và Schema Guardrail
    sys.path.insert(0, str(split_dir))
    try:
        from zaic_tools import ToolEnv
    except ImportError:
        class DummyEnv:
            def __init__(self, sp, sid):
                self.calls = []
                self.hash = "0" * 32
            def call(self, name, args):
                self.calls.append({"name": name, "arguments": args})
                return {"ok": True}
            def get_tool_calls(self):
                return self.calls
            def get_state_hash(self):
                return self.hash
        ToolEnv = DummyEnv

    catalog_filter = CatalogFilter(str(catalog_file))
    guardrail = SafeSchemaGuardrail(str(catalog_file))
    fuzzy_map = load_fuzzy_map(catalog_file) if catalog_file.exists() else {}
    train_cat = data_root / "train" / "train_tools_catalog.json"
    if train_cat.exists() and train_cat != catalog_file:
        fuzzy_map.update(load_fuzzy_map(train_cat))

    # 2. Xử lý nạp ASR Transcripts
    asr_map = {}
    asr_candidates = [
        args.asr_file,
        Path("/kaggle/working/output") / f"{args.split}_transcripts.jsonl",
        PROJECT_ROOT / "output" / f"{args.split}_transcripts.jsonl",
        PROJECT_ROOT / "MultiASR_Results" / f"{args.split}_phowhisper_medium_for_agent.jsonl",
        data_root / f"{args.split}_phowhisper_medium_for_agent.jsonl",
        data_root / f"{args.split}_asr.jsonl",
    ]
    for c in asr_candidates:
        if c and Path(c).exists():
            print(f"🎤 Tự động tìm thấy file ASR transcript: {c}")
            with open(c, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        txt = rec.get("transcript") or rec.get("audio_text") or rec.get("text", "")
                        if rec.get("id") and txt:
                            asr_map[rec["id"]] = txt
            break

    for s in samples:
        sid = s.get("id")
        if sid in asr_map:
            s["audio_text"] = asr_map[sid]

    # Kiểm tra nhu cầu gỡ băng ASR tự động nếu còn mẫu thiếu audio_text
    missing_samples = [s for s in samples if not s.get("audio_text")]
    if missing_samples:
        print(f"\n🎙️ [Tự động nhận diện Giọng nói] Có {len(missing_samples)}/{len(samples)} mẫu chưa có trường 'audio_text'.")
        print("   Đang khởi động UnifiedASRTranscriber để gỡ băng trực tiếp...")
        try:
            from src.asr.transcribe import UnifiedASRTranscriber
            from src.asr.post_processor import LexiconPostProcessor
            asr_cfg = get_asr_config()
            if args.asr_model:
                asr_cfg.model_name_or_path = args.asr_model
                if Path(args.asr_model).exists():
                    print(f"🎯 Sử dụng mô hình ASR Fine-tuned từ: {args.asr_model}")
            transcriber = UnifiedASRTranscriber(asr_cfg)
            post_proc = LexiconPostProcessor()
            audio_paths = []
            for s in missing_samples:
                raw_audio = s.get("audio", f"audio/{s['id']}/{s['id']}.wav")
                p = split_dir / raw_audio
                if not p.exists():
                    p = data_root / raw_audio
                audio_paths.append(str(p) if p.exists() else "")

            valid_indices = [i for i, p in enumerate(audio_paths) if p]
            valid_audios = [audio_paths[i] for i in valid_indices]
            if valid_audios:
                print(f"   Đang xử lý {len(valid_audios)} file âm thanh bằng ASR GPU...")
                trans_list = transcriber.transcribe_batch(valid_audios)
                for idx, t_raw in zip(valid_indices, trans_list):
                    clean_t = post_proc.process(t_raw)
                    missing_samples[idx]["audio_text"] = clean_t
                print("   ✅ Hoàn tất gỡ băng ASR tự động!")
                for s in missing_samples:
                    if s.get("audio_text"):
                        asr_map[s["id"]] = s["audio_text"]

                # Lưu ngay bản gỡ băng xuống đĩa để tái sử dụng vĩnh viễn cho mọi lần chạy sau
                save_cache = Path("/kaggle/working/output") / f"{args.split}_transcripts.jsonl"
                if not Path("/kaggle/working").exists():
                    save_cache = PROJECT_ROOT / "output" / f"{args.split}_transcripts.jsonl"
                save_cache.parent.mkdir(parents=True, exist_ok=True)
                with open(save_cache, "w", encoding="utf-8") as sc_f:
                    for s in samples:
                        if s.get("audio_text"):
                            sc_f.write(json.dumps({"id": s["id"], "transcript": s["audio_text"]}, ensure_ascii=False) + "\n")
                print(f"💾 [Cache ASR] Đã lưu vĩnh viễn {len(asr_map)} bản gỡ băng vào đĩa: {save_cache}")
        except Exception as asr_err:
            print(f"   ⚠️ Lỗi khởi động ASR ({asr_err}). Sẽ dùng fallback từ context.")
            for s in missing_samples:
                if not s.get("audio_text"):
                    s["audio_text"] = f"Lệnh điều khiển thiết bị {s.get('id', '')}"

    completed_map = {}
    if args.resume and out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        rid = row.get("id")
                        if rid:
                            completed_map[rid] = row
                    except Exception:
                        pass
        print(f"♻️ [Resume Checkpoint] Đã phát hiện và nạp {len(completed_map)} mẫu đã hoàn thành trước đó.")

    samples_to_run = [s for s in samples if s.get("id") not in completed_map]
    print(f"\n⚡ Bắt đầu chạy V2 Hai Giai Đoạn trên {len(samples_to_run)}/{len(samples)} mẫu với {args.workers} luồng song song...\n")

    results = [
        {"id": r["id"], "tool_calls": r.get("tool_calls", []), "state_hash": r.get("state_hash", "0" * 32), "user_text": "", "stats": {}, "eval": None}
        for r in completed_map.values()
    ]
    gold_scores = []
    stage1_times = []
    stage2_times = []
    total_times = []
    tools_count_list = []
    lock = threading.Lock()

    def process_sample(item):
        sid = item.get("id")
        user_text = item.get("audio_text", "")
        env = ToolEnv(args.split, sid)

        agent = TwoStageReActAgent(
            llm_client=client,
            catalog_filter=catalog_filter,
            guardrail=guardrail,
            max_steps=args.max_steps,
            time_budget=args.time_budget,
            verbose=(not args.quiet),
        )

        tool_calls, state_hash = agent.run(env, item, user_text)
        stats = agent.last_run_stats

        # Đánh giá điểm nếu mẫu có nhãn chuẩn (Gold Label)
        eval_res = None
        if "label" in item:
            eval_res = evaluate_sample(item, tool_calls, state_hash, fuzzy_map)

        return {
            "id": sid,
            "tool_calls": tool_calls,
            "state_hash": state_hash,
            "user_text": user_text,
            "stats": stats,
            "eval": eval_res,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_sample, s): s.get("id") for s in samples_to_run}
        for future in tqdm(as_completed(futures), total=len(samples_to_run), desc="V2 Two-Stage Inference"):
            sid = futures[future]
            try:
                res = future.result()
                with lock:
                    results.append(res)
                    st = res["stats"]
                    stage1_times.append(st.get("stage1_time", 0.0))
                    stage2_times.append(st.get("stage2_time", 0.0))
                    total_times.append(st.get("total_time", 0.0))
                    tools_count_list.append(st.get("num_tools_stage2", 0))

                    if res["eval"]:
                        gold_scores.append(res["eval"]["final_score"])

                    # Log trực quan thông tin mẫu vừa xử lý
                    print(f"\n  -------------------------------------------------------------")
                    print(f"  [Sample: {res['id']}] Giọng nói: \"{res['user_text']}\"")
                    print(f"    - Thời gian: S1: {st.get('stage1_time', 0.0):.2f}s | S2: {st.get('stage2_time', 0.0):.2f}s | Tổng: {st.get('total_time', 0.0):.2f}s")
                    print(f"    - Apps chọn: {st.get('selected_apps', [])} -> Mở rộng: {st.get('expanded_apps', [])} ({st.get('num_tools_stage2', 0)} tools)")
                    print(f"    - Công cụ gọi: {json.dumps(res['tool_calls'], ensure_ascii=False)}")
                    print(f"    - Mã băm: {res['state_hash']}")
                    if res["eval"]:
                        ev = res["eval"]
                        print(f"    - 🎯 Điểm BTC Gold: {ev['final_score']:.4f} (Tool Match: {ev['tool_match']:.2f}, State Match: {ev['state_match']})")
                    print(f"  -------------------------------------------------------------")
            except Exception as e:
                print(f"  [LỖI nghiêm trọng ở mẫu {sid}]: {e}")
                results.append({
                    "id": sid,
                    "tool_calls": [],
                    "state_hash": "0" * 32,
                    "error": str(e),
                })

    # Sắp xếp kết quả theo đúng thứ tự ban đầu của file test
    id_to_res = {r["id"]: r for r in results}
    ordered_results = [id_to_res.get(s["id"], {"id": s["id"], "tool_calls": [], "state_hash": "0" * 32}) for s in samples]

    # Ghi file submission_v2.jsonl
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for r in ordered_results:
            line_obj = {
                "id": r["id"],
                "tool_calls": r["tool_calls"],
                "state_hash": r["state_hash"],
            }
            f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")

    print(f"\n💾 Đã lưu kết quả suy luận V2 vào: {out_file} ({len(ordered_results)} dòng)")

    # Kiểm tra tính hợp lệ bằng bộ Schema Validator của BTC
    print("\n🔍 Đang chạy kiểm định chuẩn BTC (Schema Validation & Health Check)...")
    val_report = validate_submission_file(out_file, split_dir, catalog_file)

    print("\n" + "=" * 75)
    print("      BÁO CÁO TỔNG KẾT V2 KIẾN TRÚC HAI GIAI ĐOẠN")
    print("=" * 75)
    if "error" not in val_report:
        print(f"  Tổng số dòng:              {val_report['total_lines']}")
        print(f"  Số dòng hợp lệ 100%:       {val_report['valid_lines']} ({val_report['valid_lines']/max(1, val_report['total_lines'])*100:.1f}%)")
        print(f"  Lỗi vi phạm Schema:        {val_report['schema_errors']}")
        print(f"  Số lệnh thực thi (Execute):{val_report['execute_calls']}")
        print(f"  Số lệnh hỏi lại (Ask Info):{val_report['ask_missing_calls']}")
        print(f"  Số lệnh bỏ qua (No-Op):    {val_report['out_of_scope_calls']}")
        print(f"  Số tools độc lập đã dùng:  {len(val_report['unique_tools_used'])}")
        print(f"  Mã băm hợp lệ (32 ký tự):  {val_report['valid_hash_count']}/{val_report['total_lines']}")
    
    # Thống kê hiệu năng thời gian V2
    if total_times:
        avg_s1 = sum(stage1_times) / len(stage1_times)
        avg_s2 = sum(stage2_times) / len(stage2_times)
        avg_tot = sum(total_times) / len(total_times)
        avg_tools = sum(tools_count_list) / len(tools_count_list)
        print("\n  ⏱️ THỐNG KÊ HIỆU NĂNG THỜI GIAN & CONTEXT:")
        print(f"  - Thời gian TB Giai đoạn 1 (Router): {avg_s1:.2f}s")
        print(f"  - Thời gian TB Giai đoạn 2 (ReAct):  {avg_s2:.2f}s")
        print(f"  - Tổng thời gian TB mỗi mẫu:         {avg_tot:.2f}s (Quy định BTC: <= 30.0s)")
        print(f"  - Số công cụ TB nạp vào Stage 2:     {avg_tools:.1f} tools (So với gốc: 140 tools)")

    if gold_scores:
        print(f"\n  🎯 ĐIỂM TRUNG BÌNH GOLD LABEL: {sum(gold_scores) / len(gold_scores):.4f}")

    # Đóng gói zip nộp bài
    zip_path = Path(args.zip_output)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_file, arcname="submission.jsonl")
    print(f"\n📦 Đã đóng gói file nộp bài chuẩn BTC: {zip_path.resolve()} ({zip_path.stat().st_size / 1024:.1f} KB)")
    print("=" * 75)


if __name__ == "__main__":
    main()

