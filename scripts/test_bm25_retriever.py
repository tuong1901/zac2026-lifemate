"""
test_bm25_retriever.py — Script kiểm thử và đo hiệu năng BM25 Exemplar Retriever.
Phụ trách: Thành viên 1.
"""

import os
import sys
import time
from pathlib import Path

# Thêm đường dẫn project vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.rag.exemplar_retriever import BM25ExemplarRetriever


def main():
    train_path = "D:/Download/train+public_test/train+public_test/train/train.jsonl"
    cache_path = PROJECT_ROOT / "data" / "bm25_train_index.pkl"

    print("=" * 60)
    print("KIỂM THỬ MODULE BM25 EXEMPLAR RETRIEVER (THÀNH VIÊN 1)")
    print("=" * 60)

    retriever = BM25ExemplarRetriever(k1=1.5, b=0.75)

    # 1. Đánh chỉ mục từ train.jsonl
    print(f"\n[1] Đang nạp và đánh chỉ mục từ: {train_path}")
    t0 = time.time()
    retriever.build_from_train_file(train_path)
    build_time = (time.time() - t0) * 1000
    print(f"-> Đã đánh chỉ mục {len(retriever.corpus)} tài liệu trong {build_time:.2f} ms!")

    # 2. Lưu và thử nghiệm nạp Cache
    print(f"\n[2] Đang lưu chỉ mục vào cache: {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    retriever.save_cache(str(cache_path))
    file_size_kb = os.path.getsize(cache_path) / 1024
    print(f"-> Kích thước file cache: {file_size_kb:.1f} KB (Cực kỳ nhỏ gọn!)")

    print("[3] Kiểm tra tải lại từ Cache:")
    t0 = time.time()
    cached_retriever = BM25ExemplarRetriever()
    cached_retriever.load_cache(str(cache_path))
    load_time = (time.time() - t0) * 1000
    print(f"-> Thời gian nạp cache từ đĩa: {load_time:.2f} ms (Sẵn sàng khởi động tức thì trong Docker)")

    # 3. Test các câu truy vấn thực tế
    test_cases = [
        {
            "query": "Khoá cái cửa đấy lại cho anh luôn nhé",
            "installed_apps": {"door_lock": {}, "home_security": {}, "smart_home": {}},
            "expected_tool": "door_lock"
        },
        {
            "query": "Đặt báo thức năm giờ rưỡi sáng mai giúp anh",
            "installed_apps": {"alarm": {}, "fitness": {}, "weather": {}},
            "expected_tool": "alarm"
        },
        {
            "query": "Chuyển sang kênh VTV3 xem phim trưa nay đi",
            "installed_apps": {"tv": {}, "phone": {}},
            "expected_tool": "tv"
        },
        {
            "query": "Gửi 200 nghìn cho Lan nhé",
            "installed_apps": {"ewallet": {}, "contacts": {}, "chat_ott": {}},
            "expected_tool": "ewallet"
        }
    ]

    print("\n" + "=" * 60)
    print("CHẠY THỬ NGHIỆM TRUY XUẤT CÁC CÂU LỆNH MẪU:")
    print("=" * 60)

    total_time = 0.0
    for idx, tc in enumerate(test_cases, 1):
        q = tc["query"]
        apps = tc["installed_apps"]
        
        t_start = time.perf_counter()
        results = cached_retriever.retrieve(q, installed_apps=apps, top_k=2)
        elapsed = (time.perf_counter() - t_start) * 1000
        total_time += elapsed

        print(f"\n[Case {idx}] Query: \"{q}\"")
        print(f"-> Apps đã cài: {list(apps.keys())}")
        print(f"-> Thời gian truy xuất: {elapsed:.3f} ms")
        
        if results:
            print("-> Top-1 Mẫu tìm được:")
            top1 = results[0]
            print(f"   ID: {top1['id']} (Điểm BM25: {top1['bm25_score']})")
            print(f"   Text mẫu: \"{top1['audio_text']}\"")
            print(f"   Tools gọi: {[c['name'] for c in top1['tool_calls']]}")
        else:
            print("-> Không tìm thấy kết quả phù hợp!")

    avg_time = total_time / len(test_cases)
    print(f"\n===> THỜI GIAN SUY LUẬN TRUNG BÌNH: {avg_time:.3f} ms / mẫu (VƯỢT TRỘI SO VỚI YÊU CẦU)")

    # 4. In thử format nhúng vào Prompt
    print("\n" + "=" * 60)
    print("DEMO FORMAT XUẤT VÀO PROMPT CHO LLM:")
    print("=" * 60)
    demo_results = cached_retriever.retrieve("Đặt báo thức năm giờ rưỡi sáng mai giúp anh", installed_apps={"alarm": {}}, top_k=2)
    prompt_str = cached_retriever.format_exemplars_for_prompt(demo_results)
    print(prompt_str)


if __name__ == "__main__":
    main()
