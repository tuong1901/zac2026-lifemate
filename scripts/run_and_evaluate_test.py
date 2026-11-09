"""
run_and_evaluate_test.py — Script trọn gói Chạy Suy luận và Đánh giá Tính Hợp lệ trên tập Test (Public Test).
Dành riêng cho cuộc thi Zalo AI Challenge 2026 (LifeMate):
1. Đọc dữ liệu từ data/public_test (gồm audio và context của 634 mẫu).
2. Tự động gỡ băng ASR (hoặc nạp transcript ASR sẵn có) + Hậu xử lý âm vị.
3. Chạy ReAct Agent tương tác an toàn với ToolEnv("public_test", sample_id).
4. Xuất file submission.jsonl và tự động đóng gói submission.zip theo chuẩn nộp bài của BTC.
5. CHẤM KIỂM ĐỊNH TỰ ĐỘNG THEO CHUẨN BTC (Official Schema Validation & Health Check):
   - Quét từng tool_call qua bộ quy tắc từ chối của BTC (zaic_tools.schema_validate).
   - Kiểm tra mã băm trạng thái state_hash.
   - Thống kê tỷ lệ phân bổ lệnh: Execute Tools, Ask Missing Info, Out-of-Scope ([]).
   - Đánh giá độ sẵn sàng nộp bài (Submission Readiness).
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
from src.agent.rag.exemplar_retriever import BM25ExemplarRetriever
from src.agent.rag.history_resolver import HistoryResolver
from src.agent.react_loop import ReActAgent
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
    # Thử chính thư mục data nếu chứa public_test
    if (PROJECT_ROOT / "data" / "public_test").exists():
        return PROJECT_ROOT / "data"
    raise FileNotFoundError(
        "Không tìm thấy thư mục chứa 'public_test'! Vui lòng truyền đường dẫn qua --data_root."
    )


def validate_submission_file(sub_path: Path, split_dir: Path, catalog_path: Path) -> Dict[str, Any]:
    """
    Kiểm tra toàn diện tính hợp lệ của submission theo đúng bộ quy tắc và tool của BTC.
    """
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
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            stats["total_lines"] += 1
            try:
                row = json.loads(line)
            except Exception as exc:
                stats["schema_errors"] += 1
                stats["error_details"].append(f"Dòng {line_no}: Sai định dạng JSON - {exc}")
                continue

            sample_id = row.get("id")
            tool_calls = row.get("tool_calls", [])
            state_hash = row.get("state_hash", "")

            # Kiểm tra state_hash (phải là chuỗi hex sha256 64 ký tự)
            if state_hash and len(state_hash) == 64:
                stats["valid_hash_count"] += 1

            # Phân loại hành vi
            if not tool_calls:
                stats["out_of_scope_calls"] += 1
            elif any(c.get("name") == "sys.ask_missing_info" for c in tool_calls):
                stats["ask_missing_calls"] += 1
            else:
                stats["execute_calls"] += 1

            for c in tool_calls:
                tname = c.get("name")
                if tname:
                    stats["unique_tools_used"].add(tname)

            # Kiểm tra Schema với zaic_tools của BTC
            try:
                env = ToolEnv("public_test", sample_id, root=split_dir.parent)
                allowed = env.allowed
            except Exception:
                allowed = None

            call_results = schema_validate(tool_calls, catalog, allowed)
            has_err = False
            for c_idx, res in enumerate(call_results):
                if not res.get("ok"):
                    has_err = True
                    stats["schema_errors"] += 1
                    err_msg = f"Dòng {line_no} [{sample_id}] Call {c_idx} '{res.get('name')}': {res.get('error')}"
                    stats["error_details"].append(err_msg)

            if not has_err:
                stats["valid_lines"] += 1

    return stats


def main():
    llm_cfg = get_llm_config()
    asr_cfg = get_asr_config()

    parser = argparse.ArgumentParser(description="Chạy Pipeline và Đánh giá trên tập Public Test (LifeMate)")
    parser.add_argument("--data_root", type=str, default=None, help="Đường dẫn thư mục chứa public_test và train")
    parser.add_argument("--eval_file", type=str, default=None, help="Đường dẫn file dữ liệu cần chạy (mặc định: public_test.jsonl hoặc val_benchmark_200.jsonl)")
    parser.add_argument("--limit", type=int, default=None, help="Số lượng mẫu chạy thử (mặc định: toàn bộ)")
    parser.add_argument("--sample_ids", type=str, default=None, help="Chạy các ID cụ thể, phân cách bằng dấu phẩy (vd: public_0002,public_0003)")
    parser.add_argument("--output", type=str, default="submission.jsonl", help="Tên file kết quả đầu ra (mặc định: submission.jsonl)")
    parser.add_argument("--workers", "--concurrency", dest="concurrency", type=int, default=2, help="Số luồng chạy song song (khuyên dùng: 2 khi kết nối Colab T4 ngrok)")
    parser.add_argument("--time_budget", type=float, default=60.0, help="Ngân sách thời gian tối đa cho mỗi mẫu (giây, mặc định: 60.0s khi qua mạng)")
    parser.add_argument("--asr_file", type=str, default=None, help="Đường dẫn file ASR transcript có sẵn (nếu có)")
    parser.add_argument("--mode", choices=["server", "hf"], default=llm_cfg.mode, help="Chế độ LLM: server hoặc hf")
    parser.add_argument("--base_url", type=str, default=llm_cfg.base_url, help="URL API server LLM")
    parser.add_argument("--model", type=str, default=llm_cfg.model, help="Tên mô hình LLM")
    parser.add_argument("--max_tokens", type=int, default=256, help="Số token tối đa sinh ra (mặc định: 256 để khớp giới hạn server)")
    parser.add_argument("--validate_only", action="store_true", help="Chỉ kiểm tra và chấm điểm file submission hiện có mà không chạy lại agent")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    public_dir = data_root / "public_test"
    if args.eval_file:
        test_jsonl = Path(args.eval_file)
        if not test_jsonl.is_absolute():
            test_jsonl = PROJECT_ROOT / test_jsonl
        if not test_jsonl.exists():
            test_jsonl = data_root / args.eval_file
    else:
        test_jsonl = public_dir / "public_test.jsonl"

    catalog_path = public_dir / "public_test_tools_catalog.json"
    if not catalog_path.exists():
        catalog_path = data_root / "train" / "train_tools_catalog.json"

    fuzzy_map = load_fuzzy_map(catalog_path) if catalog_path.exists() else {}
    train_cat = data_root / "train" / "train_tools_catalog.json"
    if train_cat.exists() and train_cat != catalog_path:
        fuzzy_map.update(load_fuzzy_map(train_cat))

    output_path = Path(args.output).resolve()

    print("=" * 75)
    print("      ZAC 2026 — HỆ THỐNG SUY LUẬN & CHẤM ĐIỂM TRỰC TIẾP")
    print("=" * 75)
    print(f" Thư mục Data Root: {data_root}")
    print(f" File Dữ liệu Test:  {test_jsonl}")
    print(f" File Output:        {output_path}")

    # Chế độ 1: Chỉ validate file đã có
    if args.validate_only:
        if not output_path.exists():
            print(f"\n[Lỗi] Không tìm thấy file {output_path} để kiểm tra!")
            return
        print(f"\n🔍 Đang chạy kiểm tra Schema và Chấm tính hợp lệ cho file: {output_path.name}...")
        report = validate_submission_file(output_path, public_dir, catalog_path)
        print_report(report, output_path)
        return

    # Chế độ 2: Chạy suy luận toàn diện
    sys.path.insert(0, str(public_dir))
    from zaic_tools import ToolEnv

    # 1. Đọc danh sách mẫu
    requested_ids = set(x.strip() for x in args.sample_ids.split(",")) if args.sample_ids else None
    items = []
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            if requested_ids and item["id"] not in requested_ids:
                continue
            if args.limit and len(items) >= args.limit:
                break
            items.append(item)

    print(f"\n📋 Đã nạp {len(items)} mẫu kiểm thử từ {test_jsonl.name}")

    # 2. Xử lý ASR Transcript
    asr_map = {}
    if args.asr_file and Path(args.asr_file).exists():
        print(f"🎤 Đang nạp transcript ASR từ: {args.asr_file}")
        with open(args.asr_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    asr_map[rec["id"]] = rec.get("transcript", "")
    else:
        # Tự động tìm transcript có sẵn trong thư mục hoặc fallback sang audio_text nếu có
        asr_candidates = [
            PROJECT_ROOT / "MultiASR_Results" / "public_test_phowhisper_medium_for_agent.jsonl",
            data_root / "public_test_phowhisper_medium_for_agent.jsonl",
            data_root / "public_test_asr.jsonl",
        ]
        for c in asr_candidates:
            if c.exists():
                print(f"🎤 Tự động tìm thấy file ASR có sẵn: {c.name}")
                with open(c, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rec = json.loads(line)
                            asr_map[rec["id"]] = rec.get("transcript", "")
                break

        if not asr_map:
            audio_dir = public_dir / "audio"
            if audio_dir.exists():
                print(f"🎤 Chưa có file transcript ASR, tự động gỡ băng {len(items)} file audio bằng ASR GPU...")
                try:
                    from src.asr.transcribe import UnifiedASRTranscriber
                    from src.asr.post_processor import LexiconPostProcessor
                    transcriber = UnifiedASRTranscriber(asr_cfg)
                    post_proc = LexiconPostProcessor()
                    audio_paths = []
                    for it in items:
                        rel_audio = it.get("audio", f"audio/{it['id']}/{it['id']}.wav")
                        full_path = public_dir / rel_audio
                        if not full_path.exists():
                            full_path = data_root / rel_audio
                        audio_paths.append(full_path)

                    raw_texts = transcriber.transcribe_batch(audio_paths, batch_size=asr_cfg.batch_size or 8)
                    for it, raw_t in zip(items, raw_texts):
                        clean_t = post_proc.process(raw_t)
                        asr_map[it["id"]] = clean_t
                    print(f"✅ Gỡ băng thành công toàn bộ {len(asr_map)} mẫu!")
                except Exception as e:
                    print(f"⚠️ Không thể chạy ASR tự động: {e}. Vui lòng cung cấp --asr_file.")

    # 3. Khởi tạo LLM Client và Agent
    print(f"\n🤖 Đang kết nối LLM ({args.mode}) tại {args.base_url} [Mô hình: {args.model}]...")
    llm_client = get_llm_client(
        client_type=args.mode,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout=args.time_budget or 60.0
    )

    catalog_filter = CatalogFilter(str(catalog_path))
    retriever = BM25ExemplarRetriever()
    train_file = data_root / "train" / "train.jsonl"
    if train_file.exists():
        print(f"📚 Đang nạp {train_file.name} để xây dựng bộ nhớ mẫu BM25...")
        retriever.build_from_train_file(str(train_file))

    agent = ReActAgent(
        llm_client=llm_client,
        catalog_filter=catalog_filter,
        exemplar_retriever=retriever if len(retriever.corpus) > 0 else None,
        history_resolver=HistoryResolver(),
        guardrail=SafeSchemaGuardrail(),
        max_steps=7,
        resolve_lock_ids=True,
        time_budget=args.time_budget
    )

    # 4. Chạy suy luận (Hỗ trợ đa luồng Concurrency)
    print(f"\n⚡ Bắt đầu chạy suy luận trên {len(items)} mẫu với {args.concurrency} luồng...")
    results_map = {}
    write_lock = threading.Lock()
    t_start = time.time()

    completed_counter = [0]
    gold_scores = []
    state_matches = 0
    trace_path = output_path.with_name(f"{output_path.stem}_trace.jsonl")

    def process_sample(item):
        sample_id = item["id"]
        is_gold_eval = "label" in item
        split = "train" if is_gold_eval else "public_test"

        # Khôi phục câu khẩu lệnh (từ ASR map hoặc item)
        user_text = asr_map.get(sample_id, item.get("audio_text", ""))
        if not user_text:
            print(f"⚠️ Cảnh báo: Mẫu {sample_id} không có transcript ASR! Vui lòng chạy ASR trước hoặc truyền --asr_file.")
            user_text = ""

        env = ToolEnv(split, sample_id, root=data_root)
        t0 = time.time()
        pred_calls, hash_sys = agent.run(env, item, user_text)
        dur = time.time() - t0

        res_record = {
            "id": sample_id,
            "user_text": user_text,
            "tool_calls": pred_calls,
            "state_hash": hash_sys,
            "latency": dur
        }

        # Nếu tập dữ liệu có nhãn Gold (như val_benchmark_200.jsonl) -> Chấm điểm trực tiếp theo chuẩn BTC
        if is_gold_eval:
            gold_calls = item.get("label", {}).get("tool_calls", [])
            gold_env = ToolEnv(split, sample_id, root=data_root)
            for c in gold_calls:
                gold_env.call(c["name"], c.get("arguments") or {})
            hash_gold = gold_env.get_state_hash()
            score = evaluate_sample(pred_calls, gold_calls, hash_sys, hash_gold, fuzzy_map)
            res_record["hash_gold"] = hash_gold
            res_record["gold_calls"] = gold_calls
            res_record["score"] = score
            res_record["matched"] = (hash_sys == hash_gold)

        return res_record

    def log_and_save_result(res, out_f):
        completed_counter[0] += 1
        idx = completed_counter[0]
        sample_id = res["id"]
        dur = res["latency"]
        calls = res["tool_calls"]
        calls_str = ", ".join(f"{c['name']}({json.dumps(c.get('arguments', {}), ensure_ascii=False)})" for c in calls) if calls else "[] (Out-of-Scope)"

        print(f"\n[{idx}/{len(items)}] ID: {sample_id} | Thời gian: {dur:.2f}s", flush=True)
        print(f"   🎤 Lời thoại : \"{res['user_text']}\"", flush=True)
        print(f"   ⚡ Lệnh gọi  : {calls_str}", flush=True)
        print(f"   🔑 State Hash: {res['state_hash'][:24]}...", flush=True)

        if "score" in res:
            gold_scores.append(res["score"])
            if res["matched"]:
                nonlocal state_matches
                state_matches += 1
            status_icon = "🎯 MATCH (100% Khớp State)" if res["matched"] else "❌ MISMATCH (0 Điểm)"
            print(f"   🏆 ĐIỂM SỐ BTC: {res['score']*100:.2f}/100 | Trạng thái: {status_icon}", flush=True)
        print("-" * 75, flush=True)

        # Ghi ngay vào file submission.jsonl
        out_f.write(json.dumps({
            "id": res["id"],
            "tool_calls": res["tool_calls"],
            "state_hash": res["state_hash"]
        }, ensure_ascii=False) + "\n")
        out_f.flush()

        # Ghi vết chi tiết vào trace file
        with open(trace_path, "a", encoding="utf-8") as tf:
            tf.write(json.dumps(res, ensure_ascii=False) + "\n")

    with open(output_path, "w", encoding="utf-8") as out_f:
        if args.concurrency > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {executor.submit(process_sample, it): it["id"] for it in items}
                for fut in as_completed(futures):
                    res = fut.result()
                    results_map[res["id"]] = res
                    with write_lock:
                        log_and_save_result(res, out_f)
        else:
            for it in items:
                res = process_sample(it)
                results_map[res["id"]] = res
                log_and_save_result(res, out_f)

    total_time = time.time() - t_start
    avg_latency = total_time / len(items) if items else 0.0

    # 5. Đóng gói ZIP chuẩn BTC
    zip_path = output_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(output_path, arcname=output_path.name)

    print(f"\n✅ Đã hoàn thành suy luận trong {total_time:.1f}s (Trung bình: {avg_latency:.2f}s/mẫu)")
    print(f"📦 Đã đóng gói file nộp bài: {zip_path.name}")

    # 6. Chấm kiểm tra Schema và tính hợp lệ tự động
    print(f"\n🔍 Đang chạy kiểm định Schema theo chuẩn BTC cho file: {output_path.name}...")
    report = validate_submission_file(output_path, public_dir, catalog_path)
    report["avg_latency"] = avg_latency
    report["gold_scores"] = gold_scores
    report["state_matches"] = state_matches
    print_report(report, output_path)


def print_report(stats: Dict[str, Any], file_path: Path):
    """In bảng báo cáo kết quả kiểm định chi tiết."""
    total = stats.get("total_lines", 0)
    schema_errs = stats.get("schema_errors", 0)
    valid_lines = stats.get("valid_lines", 0)
    exec_calls = stats.get("execute_calls", 0)
    ask_calls = stats.get("ask_missing_calls", 0)
    oos_calls = stats.get("out_of_scope_calls", 0)
    valid_hash = stats.get("valid_hash_count", 0)
    unique_tools = len(stats.get("unique_tools_used", set()))
    avg_lat = stats.get("avg_latency", 0.0)

    valid_rate = (valid_lines / total * 100) if total > 0 else 0.0
    hash_rate = (valid_hash / total * 100) if total > 0 else 0.0

    print("\n" + "=" * 75)
    print("       BÁO CÁO KIỂM ĐỊNH KẾT QUẢ PUBLIC TEST (SUBMISSION HEALTH)")
    print("=" * 75)
    print(f" 📄 File kiểm tra:                {file_path.name}")
    print(f" 🔢 Tổng số mẫu:                 {total}")
    print(f"  Tỷ lệ hợp lệ Schema BTC:      {valid_rate:.2f}% ({valid_lines}/{total})")
    print(f" ❌ Số lỗi Schema (U bị từ chối): {schema_errs}")
    print(f" 🔑 Mã băm State Hash hợp lệ:     {hash_rate:.2f}% ({valid_hash}/{total})")
    if avg_lat > 0:
        print(f" ⏱️ Tốc độ xử lý trung bình:      {avg_lat:.2f} giây/mẫu (Quy chế: <= 30.0s)")

    print("-" * 75)
    print(" 📊 PHÂN BỔ HÀNH VI CỦA AGENT TRÊN TẬP TEST:")
    print(f"    - Thực thi công cụ (Action):  {exec_calls:3d} mẫu ({(exec_calls/total*100) if total else 0:.1f}%)")
    print(f"    - Hỏi lại (ask_missing_info): {ask_calls:3d} mẫu ({(ask_calls/total*100) if total else 0:.1f}%)")
    print(f"    - Ngoài phạm vi / Hủy ([]):   {oos_calls:3d} mẫu ({(oos_calls/total*100) if total else 0:.1f}%)")
    print(f"    - Số loại công cụ được dùng:  {unique_tools} tools")
    print("-" * 75)

    if "gold_scores" in stats and stats["gold_scores"]:
        scores = stats["gold_scores"]
        final_sc = sum(scores) / len(scores) * 100.0
        st_rate = stats.get("state_matches", 0) / len(scores) * 100.0
        print("-" * 75)
        print(f" 🏆 TỔNG ĐIỂM CHUẨN BTC (OFFICIAL SCORE): {final_sc:.2f} / 100")
        print(f" 🎯 TỶ LỆ KHỚP STATE (GATEKEEPER):       {st_rate:.2f}% ({stats.get('state_matches', 0)}/{len(scores)})")
        print("-" * 75)

    if schema_errs == 0 and total > 0:
        print(" 🎉 KẾT LUẬN: ĐẠT CHUẨN 100% — HOÀN TOÀN SẴN SÀNG ĐỂ NỘP LÊN BTC!")
        print(f"    👉 Bạn có thể nộp ngay file '{file_path.with_suffix('.zip').name}' lên cổng thi.")
    else:
        print(f" ⚠️ CẢNH BÁO: Phát hiện {schema_errs} lỗi Schema. Chi tiết các lỗi:")
        for err in stats.get("error_details", [])[:10]:
            print(f"    - {err}")
        if len(stats.get("error_details", [])) > 10:
            print(f"    ... và {len(stats.get('error_details')) - 10} lỗi khác.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

