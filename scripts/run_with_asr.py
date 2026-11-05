"""
run_with_asr.py — Đánh giá và so sánh trực tiếp điểm số giữa:
1. Benchmark với DB chuẩn của BTC (Gold audio_text trong val_benchmark_200.jsonl)
2. Benchmark với DB thực tế đã tối ưu ASR của Khoa (MultiASR_Results: PhoWhisper-medium / Whisper-turbo / PhoWhisper-small)
3. Xuất file submission.jsonl trên 634 mẫu public_test (nếu bật --make_submission)
"""

import os
import sys
import json
import copy
import argparse
from pathlib import Path
from tqdm import tqdm

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
from src.agent.rag.structured_router import StructuredDomainRouter
from src.agent.react_loop import ReActAgent
from src.agent.llm_client import get_llm_client
from local_eval import load_fuzzy_map, evaluate_sample


def load_asr_map(asr_jsonl_path: Path) -> dict:
    asr_map = {}
    if not asr_jsonl_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file ASR: {asr_jsonl_path}")
    with open(asr_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["id"] in asr_map:
                raise ValueError(f"Trùng ID trong file ASR: {rec['id']}")
            asr_map[rec["id"]] = rec
    return asr_map


def inject_asr_into_item(item: dict, asr_rec: dict) -> tuple[dict, str]:
    """
    Thay thế câu khẩu lệnh chính và các lượt user trong history bằng kết quả ASR của Khoa.
    """
    new_item = copy.deepcopy(item)
    user_text = asr_rec.get("transcript", item.get("audio_text", ""))
    new_item["audio_text"] = user_text

    hist_Map = {}
    for h_tr in asr_rec.get("history_user_transcripts", []):
        if "history_index" in h_tr and "transcript" in h_tr:
            hist_Map[h_tr["history_index"]] = h_tr["transcript"]

    if "history" in new_item and isinstance(new_item["history"], list):
        for idx, turn in enumerate(new_item["history"]):
            if idx in hist_Map:
                turn["audio_text"] = hist_Map[idx]

    return new_item, user_text


def eval_dataset(agent, eval_items, root: Path, fuzzy_map: dict, asr_map: dict = None, desc: str = "Eval", trace_file: Path = None):
    from zaic_tools import ToolEnv

    scores = []
    state_matches = 0
    schema_penalties = 0
    total = len(eval_items)

    for idx, item in enumerate(tqdm(eval_items, desc=desc), 1):
        sample_id = item["id"]
        gold_calls = item.get("label", {}).get("tool_calls", [])

        if asr_map is not None:
            if sample_id not in asr_map or not asr_map[sample_id].get("transcript"):
                raise ValueError(f"Thiếu transcript ASR cho {sample_id}")
            run_item, user_text = inject_asr_into_item(item, asr_map[sample_id])
        else:
            run_item = item
            user_text = item.get("audio_text", "")

        gold_env = ToolEnv("train", sample_id, root=root)
        for c in gold_calls:
            gold_env.call(c["name"], c.get("arguments") or {})
        hash_gold = gold_env.get_state_hash()

        sys_env = ToolEnv("train", sample_id, root=root)
        pred_calls, hash_sys = agent.run(sys_env, run_item, user_text)

        has_rejected = any("error" in c.get("result", {}) for c in sys_env.log)
        if has_rejected:
            schema_penalties += 1

        sc = evaluate_sample(pred_calls, gold_calls, hash_sys, hash_gold, fuzzy_map)
        scores.append(sc)
        matched = (hash_sys == hash_gold)
        if matched:
            state_matches += 1

        if trace_file is not None:
            trace = {"source": "asr" if asr_map is not None else "gold", "id": sample_id,
                     "user_text": user_text, "gold_calls": gold_calls, "pred_calls": pred_calls,
                     "state_match": matched, "score": sc, "schema_rejected": has_rejected}
            with open(trace_file, "a", encoding="utf-8") as out:
                out.write(json.dumps(trace, ensure_ascii=False) + "\n")

        if total <= 20:
            status_icon = "✅" if matched and sc == 1.0 else ("⚠️" if matched else "❌")
            print(f"\n   {status_icon} [{idx:02d}/{total}] {sample_id} | Điểm mẫu: {sc:.2f} (Hash match={matched})", flush=True)
            if sc < 1.0:
                print(f"      - PRED: {json.dumps(pred_calls, ensure_ascii=False)}", flush=True)
                print(f"      - GOLD: {json.dumps(gold_calls, ensure_ascii=False)}", flush=True)
            else:
                print(f"      - CALLS: {[c['name'] for c in pred_calls]}", flush=True)

        if idx % 25 == 0 or idx == total:
            cur_score = (100.0 / idx) * sum(scores)
            cur_state = (state_matches / idx) * 100.0
            print(f"\n   -> [{desc}] Tiến độ {idx}/{total} | State Hash: {cur_state:.2f}% ({state_matches}/{idx}) | Phạt U: {schema_penalties} | Điểm: {cur_score:.4f}/100", flush=True)

    final_score = (100.0 / total) * sum(scores) if total > 0 else 0.0
    state_rate = (state_matches / total) * 100.0 if total > 0 else 0.0
    res = {
        "total": total,
        "final_score": final_score,
        "state_match_rate": state_rate,
        "state_matches": state_matches,
        "schema_penalties": schema_penalties,
    }
    print(f"\n✅ HOÀN TẤT [{desc}]: Khớp State Hash = {state_rate:.2f}% ({state_matches}/{total}) | Phạt U = {schema_penalties} | ĐIỂM CHUNG CUỘC = {final_score:.4f} / 100\n", flush=True)
    return res


def resolve_data_root(preferred_path: str = None) -> Path:
    candidates = [
        preferred_path,
        "D:/Download/train+public_test/train+public_test",
        "D:/TOOL_TH/ZAC_2026/train+public_test/train+public_test",
        "/content/train+public_test/train+public_test",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError("Không tìm thấy thư mục data_root! Hãy truyền --data_root hợp lệ.")


def resolve_asr_dir(preferred_path: str = None) -> Path:
    candidates = [
        preferred_path,
        str(PROJECT_ROOT / "data" / "MultiASR_Results"),
        "C:/Users/Admin/Downloads/MultiASR_Results",
        "D:/TOOL_TH/ZAC_2026/MultiASR_Results",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError("Không tìm thấy thư mục MultiASR_Results! Hãy truyền --asr_dir hợp lệ.")


def generate_submission(agent, root: Path, asr_dir: Path, output_file: Path, limit: int = None, asr_model: str = "phowhisper_medium", workers: int = 4):
    import zipfile
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from zaic_tools import ToolEnv
    from src.agent.prompt import _get_multi_asr_variants, _get_atomic_retriever

    split_dir = root / "public_test"
    test_jsonl = split_dir / "public_test.jsonl"
    asr_jsonl = asr_dir / f"public_test_{asr_model}_for_agent.jsonl"

    print(f"\n🚀 BẮT ĐẦU CHẠY INFERENCE VÀ XUẤT SUBMISSION TRÊN PUBLIC_TEST ({asr_model}, workers={workers})")
    asr_map = load_asr_map(asr_jsonl)
    _get_multi_asr_variants("public_0001", "")
    if getattr(agent, "exemplar_retriever", None):
        _get_atomic_retriever(agent.exemplar_retriever)

    items = []
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                items.append(json.loads(line))

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Đọc các mẫu đã xử lý từ trước nếu có để hỗ trợ resume
    processed_map = {}
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        processed_map[rec["id"]] = rec
                    except Exception:
                        pass
        if len(processed_map) >= len(items):
            print(f"♻️ File {output_file.name} đã có đủ {len(processed_map)} mẫu từ lần chạy cũ -> Xoá để chạy mới hoàn toàn!")
            processed_map.clear()
            output_file.unlink(missing_ok=True)
        elif processed_map:
            print(f"ℹ️ Tìm thấy {len(processed_map)} mẫu đã xử lý trước đó trong {output_file.name}. Tiếp tục chạy phần còn lại...")

    pending_items = [it for it in items if it["id"] not in processed_map]
    write_lock = threading.Lock()

    if pending_items:
        with open(output_file, "a", encoding="utf-8") as out_f:
            def _process_one(item):
                sample_id = item["id"]
                if sample_id not in asr_map or not asr_map[sample_id].get("transcript"):
                    raise ValueError(f"Thiếu transcript ASR cho {sample_id}")
                run_item, user_text = inject_asr_into_item(item, asr_map[sample_id])
                env = ToolEnv("public_test", sample_id, root=root)
                pred_calls, hash_sys = agent.run(env, run_item, user_text)
                rec = {
                    "id": sample_id,
                    "tool_calls": pred_calls,
                    "state_hash": hash_sys,
                }
                with write_lock:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    processed_map[sample_id] = rec
                return rec

            if workers <= 1:
                for it in tqdm(pending_items, desc="Generating Submission"):
                    _process_one(it)
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = [ex.submit(_process_one, it) for it in pending_items]
                    for fut in tqdm(as_completed(futs), total=len(futs), desc="Generating Submission"):
                        fut.result()

    ordered_recs = [processed_map[it["id"]] for it in items if it["id"] in processed_map]
    with open(output_file, "w", encoding="utf-8") as out_f:
        for rec in ordered_recs:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Đóng gói zip submission theo đúng chuẩn BTC
    zip_path = output_file.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(output_file, arcname=output_file.name)

    print(f"\n✅ ĐÃ XUẤT FILE SUBMISSION THÀNH CÔNG:")
    print(f"   - File JSONL: {output_file} ({len(ordered_recs)} mẫu)")
    print(f"   - File ZIP:   {zip_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Chạy so sánh Benchmark BTC vs DB ASR của Khoa")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--asr_dir", type=str, default=None)
    parser.add_argument("--eval_file", type=str, default="data/val_benchmark_200.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Số lượng mẫu tối đa (mặc định: toàn bộ)")
    parser.add_argument("--sample_ids", type=str, default=None, help="Danh sách ID cần chấm, phân tách bằng dấu phẩy")
    parser.add_argument("--mode", choices=["hf", "server"], default="server")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--asr_model", choices=["phowhisper_medium", "whisper_turbo", "phowhisper_small", "all"], default="phowhisper_medium")
    parser.add_argument("--no_exemplars", action="store_true", help="Thử agent không dùng ví dụ BM25 trong prompt")
    parser.add_argument("--structured-router", action="store_true", help="Thử domain classifier + workflow chunks trước BM25 exemplar")
    parser.add_argument(
        "--resolve-lock-ids",
        dest="resolve_lock_ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Đọc danh sách khóa và đối chiếu lock_id từ câu hiện tại/history (mặc định bật)",
    )
    parser.add_argument("--eval_source", choices=["both", "gold", "asr"], default="both", help="Chọn đầu vào để chấm benchmark")
    parser.add_argument("--trace_file", type=str, default=None, help="Ghi kết quả từng mẫu JSONL để phân tích lỗi")
    parser.add_argument("--make_submission", action="store_true", help="Chạy trên 634 mẫu public_test và xuất submission.jsonl + submission.zip")
    parser.add_argument("--output_file", type=str, default="submission.jsonl")
    args = parser.parse_args()
    requested_ids = {value.strip() for value in args.sample_ids.split(",")} if args.sample_ids else None
    trace_file = Path(args.trace_file) if args.trace_file else None
    if trace_file is not None:
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.write_text("", encoding="utf-8")

    root = resolve_data_root(args.data_root)
    asr_dir = resolve_asr_dir(args.asr_dir)

    split_dir = root / ("public_test" if args.make_submission else "train")
    catalog_file = split_dir / f"{split_dir.name}_tools_catalog.json"
    cache_path = PROJECT_ROOT / "data" / "bm25_train_index.pkl"

    sys.path.insert(0, str(split_dir))
    import zaic_tools  # noqa

    fuzzy_map = load_fuzzy_map(catalog_file)

    if args.mode == "hf":
        llm_client = get_llm_client(client_type="hf", model_name_or_path=args.model)
    else:
        llm_client = get_llm_client(client_type="openai", base_url=args.base_url, model=args.model)

    catalog_filter = CatalogFilter(str(catalog_file))
    eval_items = []
    excluded_ids = set()
    if not args.make_submission:
        eval_path = PROJECT_ROOT / args.eval_file if not Path(args.eval_file).is_absolute() else Path(args.eval_file)
        with open(eval_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                excluded_ids.add(item["id"])
                if requested_ids is not None and item["id"] not in requested_ids:
                    continue
                if args.limit is not None and len(eval_items) >= args.limit:
                    continue
                eval_items.append(item)

    retriever = BM25ExemplarRetriever()
    if args.make_submission:
        retriever.build_from_train_file(str(root / "train" / "train.jsonl"))
        retriever.save_cache(str(cache_path))
    else:
        retriever.build_from_train_file(str(root / "train" / "train.jsonl"), exclude_ids=excluded_ids)
        if any(rec["id"] in excluded_ids for rec in retriever.corpus):
            raise RuntimeError("Validation ID vẫn nằm trong chỉ mục BM25")
        print(f"[Clean holdout] Loại {len(excluded_ids)} validation ID khỏi BM25; còn {len(retriever.corpus)} ví dụ train.")


    structured_router = None
    if args.structured_router and not args.no_exemplars:
        print("[Structured Router] Đang huấn luyện domain classifier từ clean train chunks...")
        structured_router = StructuredDomainRouter(retriever).fit()

    agent = ReActAgent(
        llm_client=llm_client,
        catalog_filter=catalog_filter,
        exemplar_retriever=None if args.no_exemplars else retriever,
        history_resolver=HistoryResolver(),
        guardrail=SafeSchemaGuardrail(),
        max_steps=7,
        resolve_lock_ids=args.resolve_lock_ids,
        structured_router=structured_router
    )

    if args.make_submission:
        out_path = Path(args.output_file) if Path(args.output_file).is_absolute() else PROJECT_ROOT / args.output_file
        generate_submission(agent, root, asr_dir, out_path, limit=args.limit, asr_model=args.asr_model if args.asr_model != "all" else "phowhisper_medium")
        return

    print("\n" + "=" * 72)
    print(f" BẮT ĐẦU CHẠY ĐỐI CHIẾU TRÊN {len(eval_items)} MẪU BENCHMARK")
    print("=" * 72)

    # 1. Chạy trên DB chuẩn của BTC (Gold Text)
    res_gold = None
    if args.eval_source in ("both", "gold"):
        res_gold = eval_dataset(agent, eval_items, root, fuzzy_map, asr_map=None, desc="1. DB Chuẩn BTC (Gold Text)", trace_file=trace_file)

    # 2. Chạy trên DB ASR của Khoa
    models_to_run = (["phowhisper_medium", "whisper_turbo", "phowhisper_small"] if args.asr_model == "all" else [args.asr_model]) if args.eval_source in ("both", "asr") else []
    asr_results = {}
    for m_name in models_to_run:
        asr_file = asr_dir / f"train_{m_name}_for_agent.jsonl"
        asr_map = load_asr_map(asr_file)
        asr_results[m_name] = eval_dataset(agent, eval_items, root, fuzzy_map, asr_map=asr_map, desc=f"2. DB Khoa ({m_name})", trace_file=trace_file)

    print("\n" + "=" * 72)
    print(" BẢNG TỔNG HỢP ĐIỂM SỐ (TRÊN THANG 100.0 CỦA BTC)")
    print("=" * 72)
    print(f"{'Bộ dữ liệu (Dataset)':<35} | {'Khớp State Hash':<16} | {'Phạt U':<8} | {'ĐIỂM SỐ (/100)'}")
    print("-" * 72)
    if res_gold is not None:
        print(f"{'1. DB Ban Tổ Chức (Gold Text)':<35} | {res_gold['state_match_rate']:>6.2f}% ({res_gold['state_matches']}/{res_gold['total']}) | {res_gold['schema_penalties']:>6} | {res_gold['final_score']:>8.4f} / 100")
    for m_name, r in asr_results.items():
        label = f"2. DB Khoa ({m_name})"
        print(f"{label:<35} | {r['state_match_rate']:>6.2f}% ({r['state_matches']}/{r['total']}) | {r['schema_penalties']:>6} | {r['final_score']:>8.4f} / 100")
    print("=" * 72)


if __name__ == "__main__":
    main()

