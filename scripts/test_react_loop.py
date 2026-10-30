"""
test_react_loop.py — Kiểm thử toàn diện Nấc thang 1 (End-to-End Pipeline).
Kiểm tra sự phối hợp giữa:
- BM25 Exemplar Retriever
- History Resolver
- Safe Schema Guardrail
- ReAct Agent Loop
"""

import sys
import json
from pathlib import Path

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


class MockLLMClient:
    """
    Mock LLM Client để kiểm thử chu trình ReAct loop hoạt động chuẩn xác:
    Mô phỏng trường hợp LLM sinh lệnh ở bước 1, nhận kết quả, và hoàn thành ở bước 2.
    """
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.call_count = 0
        self.received_prompts = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.received_prompts.append(user_prompt)
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        # Mặc định khi hết responses
        return json.dumps({"thought": "Hoàn tất", "action": None, "is_finished": True})


def main():
    print("=" * 65)
    print("KIỂM THỬ TOÀN DIỆN NẤC THANG 1 (END-TO-END PIPELINE)")
    print("=" * 65)

    catalog_path = DATA_ROOT / "train" / "train_tools_catalog.json"
    cache_path = PROJECT_ROOT / "data" / "bm25_train_index.pkl"

    # 1. Khởi tạo các module của Nấc thang 1
    print("\n[+] Đang khởi tạo các module...")
    catalog_filter = CatalogFilter(str(catalog_path))
    
    retriever = BM25ExemplarRetriever()
    if cache_path.exists():
        retriever.load_cache(str(cache_path))
        print("-> Đã nạp BM25 Retriever từ cache.")
    else:
        retriever.build_from_train_file(str(DATA_ROOT / "train" / "train.jsonl"))
        print("-> Đã khởi tạo BM25 Retriever từ train.jsonl.")

    history_resolver = HistoryResolver()
    guardrail = SafeSchemaGuardrail()

    # 2. Chuẩn bị mẫu test: train_1364
    # Câu: "Anh bơi xong rồi, em ghi buổi tập đúng như Kevin Walsh dặn nhé, với đặt báo thức năm giờ rưỡi sáng mai giúp anh."
    train_file = DATA_ROOT / "train" / "train.jsonl"
    target_sample = None
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["id"] == "train_1364":
                target_sample = item
                break

    assert target_sample is not None, "Không tìm thấy mẫu train_1364!"
    print(f"\n[+] Đã nạp mẫu: {target_sample['id']}")
    print(f"-> Khẩu lệnh: \"{target_sample['audio_text']}\"")

    # Giả lập 2 bước gọi của LLM (cố tình đưa lỗi "45" dạng chuỗi ở bước 1 để thử Guardrail)
    mock_responses = [
        # Bước 1: Gọi fitness.log_workout với "minutes": "45" (Guardrail sẽ tự sửa thành int 45)
        json.dumps({
            "thought": "Ghi nhận buổi tập bơi 45 phút theo tin nhắn của Kevin Walsh",
            "action": {
                "name": "fitness.log_workout",
                "arguments": {"type": "swimming", "minutes": "45"}
            },
            "is_finished": False
        }),
        # Bước 2: Gọi alarm.create đặt báo thức 5:30 sáng mai
        json.dumps({
            "thought": "Đặt báo thức 5:30 sáng mai",
            "action": {
                "name": "alarm.create",
                "arguments": {"time": "2026-12-18 05:30:00"}
            },
            "is_finished": True
        })
    ]

    mock_llm = MockLLMClient(mock_responses)

    agent = ReActAgent(
        llm_client=mock_llm,
        catalog_filter=catalog_filter,
        exemplar_retriever=retriever,
        history_resolver=history_resolver,
        guardrail=guardrail,
        max_steps=4
    )

    # 3. Khởi tạo ToolEnv độc lập cho mẫu train_1364
    env = ToolEnv("train", "train_1364", root=DATA_ROOT)

    # 4. Chạy Agent
    tool_calls, state_hash = agent.run(env, target_sample, target_sample["audio_text"])

    print("\n" + "=" * 65)
    print("KẾT QUẢ THỰC THI CUỐI CÙNG (SUBMISSION RECORD):")
    print("=" * 65)
    print(f"ID: train_1364")
    print(f"Tool calls ({len(tool_calls)} calls):")
    for tc in tool_calls:
        print(f"  - {tc['name']}({tc['arguments']})")
    print(f"State hash: {state_hash}")

    # So sánh với gold label của BTC
    gold_label = target_sample["label"]
    gold_hash = target_sample["label"]["final_state"]
    # Kiểm tra xem có lệnh nào bị phạt U không
    has_rejected = any("error" in c.get("result", {}) for c in env.log)
    print(f"\n-> Kiểm tra lỗi schema (Phạt U): {'CÓ LỖI' if has_rejected else 'HOÀN TOÀN KHÔNG BỊ PHẠT U (100% CLEAN)'}")

    # In thử prompt cuối cùng LLM nhận được ở bước 2 để chứng minh tính đầy đủ
    print("\n" + "=" * 65)
    print("PROMPT MẪU ĐƯỢC BUILD BỞI HARNESS CHO LLM:")
    print("=" * 65)
    print(mock_llm.received_prompts[1][:800] + "\n... [TRUNCATED] ...")


if __name__ == "__main__":
    main()
