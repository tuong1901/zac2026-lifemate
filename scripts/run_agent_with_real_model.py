"""
run_agent_with_real_model.py — Chạy Agent với mô hình ngôn ngữ lớn thật (Qwen3-4B-Instruct-2507).
Hỗ trợ 2 chế độ:
1. Mode 'hf': Nạp trực tiếp mô hình local qua HuggingFace (chạy trên GPU hoặc CPU).
2. Mode 'server': Kết nối tới local vLLM / Ollama server (hoặc OpenAI compatible endpoint).

Ví dụ chạy:
  # Chế độ HuggingFace local:
  python scripts/run_agent_with_real_model.py --mode hf --model Qwen/Qwen3-4B-Instruct-2507 --sample_id train_1364

  # Chế độ kết nối local server (vLLM / Ollama):
  python scripts/run_agent_with_real_model.py --mode server --base_url http://localhost:8000/v1 --sample_id train_1364
"""

import sys
import json
import argparse
from pathlib import Path

# Đảm bảo in tiếng Việt mượt mà trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("D:/Download/train+public_test/train+public_test")

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DATA_ROOT / "train"))

from zaic_tools import ToolEnv
from src.rules.catalog_filter import CatalogFilter
from src.rules.schema_guardrail import SafeSchemaGuardrail
from src.agent.rag.exemplar_retriever import BM25ExemplarRetriever
from src.agent.rag.history_resolver import HistoryResolver
from src.agent.react_loop import ReActAgent
from src.agent.llm_client import get_llm_client


def main():
    parser = argparse.ArgumentParser(description="Chạy Agent LifeMate với LLM thật")
    parser.add_argument("--mode", choices=["hf", "server"], default="hf", help="Chế độ LLM: 'hf' (HuggingFace) hoặc 'server' (vLLM/OpenAI compatible)")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Tên model hoặc đường dẫn checkpoint local")
    parser.add_argument("--base_url", default="https://paragraphs-preservation-triumph-month.trycloudflare.com/v1", help="Base URL nếu chạy server vLLM/Ollama")
    parser.add_argument("--api_key", default="EMPTY", help="API key nếu cần")
    parser.add_argument("--sample_id", default="train_1364", help="ID mẫu trong train.jsonl để chạy thử")
    parser.add_argument("--max_steps", type=int, default=5, help="Số bước ReAct tối đa")
    args = parser.parse_args()

    print("=" * 65)
    print("CHẠY THỬ AGENT LIFEMATE VỚI MÔ HÌNH THẬT")
    print("=" * 65)

    catalog_path = DATA_ROOT / "train" / "train_tools_catalog.json"
    cache_path = PROJECT_ROOT / "data" / "bm25_train_index.pkl"

    # 1. Khởi tạo LLM Client thật
    print(f"\n[1] Khởi tạo LLM Client (Mode: {args.mode}, Model: {args.model})...")
    if args.mode == "hf":
        llm_client = get_llm_client(client_type="hf", model_name_or_path=args.model)
    else:
        llm_client = get_llm_client(
            client_type="openai",
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model
        )

    # 2. Khởi tạo các module
    print("\n[2] Nạp các module RAG & Guardrail...")
    catalog_filter = CatalogFilter(str(catalog_path))
    
    retriever = BM25ExemplarRetriever()
    if cache_path.exists():
        retriever.load_cache(str(cache_path))
        print("-> Đã nạp BM25 Retriever từ cache.")
    else:
        retriever.build_from_train_file(str(DATA_ROOT / "train" / "train.jsonl"))
        print("-> Đã tạo chỉ mục BM25 Retriever.")

    history_resolver = HistoryResolver()
    guardrail = SafeSchemaGuardrail()

    agent = ReActAgent(
        llm_client=llm_client,
        catalog_filter=catalog_filter,
        exemplar_retriever=retriever,
        history_resolver=history_resolver,
        guardrail=guardrail,
        max_steps=args.max_steps
    )

    # 3. Tìm mẫu cần chạy
    train_file = DATA_ROOT / "train" / "train.jsonl"
    target_sample = None
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["id"] == args.sample_id:
                target_sample = item
                break

    if not target_sample:
        print(f"Lỗi: Không tìm thấy mẫu {args.sample_id} trong train.jsonl!")
        return

    print(f"\n[3] Bắt đầu suy luận cho mẫu: {target_sample['id']}")
    print(f"-> Khẩu lệnh người dùng: \"{target_sample['audio_text']}\"")

    env = ToolEnv("train", target_sample["id"], root=DATA_ROOT)

    # 4. Chạy Agent thật
    tool_calls, state_hash = agent.run(env, target_sample, target_sample["audio_text"])

    print("\n" + "=" * 65)
    print("KẾT QUẢ THỰC THI THỰC TẾ CỦA MÔ HÌNH:")
    print("=" * 65)
    print(f"ID: {target_sample['id']}")
    print(f"Số lượng Tool Calls đã gọi: {len(tool_calls)}")
    for tc in tool_calls:
        print(f"  - {tc['name']}({tc['arguments']})")
    print(f"State Hash của Agent: {state_hash}")

    # So sánh với Gold Label
    gold_calls = target_sample["label"]["tool_calls"]
    print("\n" + "-" * 60)
    print("ĐÁP ÁN CHUẨN (GOLD LABEL) CỦA BTC:")
    for gc in gold_calls:
        print(f"  - {gc['name']}({gc['arguments']})")

    # Kiểm tra xem có bị phạt U không
    has_rejected = any("error" in c.get("result", {}) for c in env.log)
    print(f"\n-> Kiểm tra lỗi schema (Phạt U): {'CÓ LỖI (BỊ PHẠT)' if has_rejected else 'HOÀN TOÀN KHÔNG BỊ PHẠT U (100% CLEAN)'}")


if __name__ == "__main__":
    main()
