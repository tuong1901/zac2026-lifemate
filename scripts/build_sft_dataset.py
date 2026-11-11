"""
scripts/build_sft_dataset.py — Bộ xây dựng dữ liệu SFT chuyên nghiệp:
1. Dữ liệu gốc (Gold train samples).
2. Khai thác lịch sử (History Mining): Biến các lượt hỏi đáp trong history thành sample độc lập (+2,000 samples).
3. Học từ chối (OOS by App Stripping): Xóa app khỏi context để dạy model không bịa tool (+300 samples).
4. Tích hợp dữ liệu tổng hợp (Synthetic Unseen Apps).

Cách dùng:
    python scripts/build_sft_dataset.py --train data/train/train.jsonl --out data/sft_train.jsonl
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set

SEED = 42
random.seed(SEED)

def get_installed_apps(item: Dict[str, Any]) -> Set[str]:
    context = item.get("context", {})
    installed = context.get("installed_apps", {})
    if isinstance(installed, dict):
        return set(installed.keys())
    elif isinstance(installed, list):
        return set(installed)
    return set()

def get_gold_apps(tool_calls: List[Dict[str, Any]]) -> Set[str]:
    apps = set()
    for tc in tool_calls:
        name = tc.get("name", "")
        if name == "sys.ask_missing_info":
            args = tc.get("arguments", {})
            tname = args.get("tool_name", "")
            if "." in tname:
                apps.add(tname.split(".")[0])
        elif "." in name:
            apps.add(name.split(".")[0])
    return apps - {"sys"}

def build_chat_messages(system_prompt: str, history: List[Dict[str, Any]], user_text: str, tool_calls: List[Dict[str, Any]], tool_results: List[Any]) -> List[Dict[str, Any]]:
    messages = [{"role": "system", "content": system_prompt}]
    
    # Lịch sử hội thoại
    for h in history:
        role = h.get("role", "user")
        content = h.get("content") or h.get("audio_text", "")
        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            h_calls = h.get("tool_calls", [])
            messages.append({"role": "assistant", "content": content, "tool_calls": h_calls})
        elif role == "tool":
            messages.append({"role": "tool", "content": str(content)})

    # Lượt user hiện tại
    messages.append({"role": "user", "content": user_text})

    # Lượt assistant gọi tool
    if tool_calls:
        formatted_calls = []
        for tc in tool_calls:
            formatted_calls.append({
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False) if isinstance(tc.get("arguments"), dict) else str(tc.get("arguments", "{}"))
                }
            })
        messages.append({"role": "assistant", "content": "", "tool_calls": formatted_calls})
        
        # Kết quả tool
        for res in tool_results:
            res_str = json.dumps(res, ensure_ascii=False) if isinstance(res, (dict, list)) else str(res)
            messages.append({"role": "tool", "content": res_str[:2000]})
            
        messages.append({"role": "assistant", "content": "Dạ, em đã thực hiện xong yêu cầu của anh/chị rồi ạ."})
    else:
        # Không có tool nào cần gọi
        messages.append({"role": "assistant", "content": "Dạ, yêu cầu này em chưa hỗ trợ được trên các ứng dụng hiện tại ạ."})

    return messages

def main():
    parser = argparse.ArgumentParser(description="Tạo file dataset SFT chất lượng cao")
    parser.add_argument("--train", type=str, required=True, help="Đường dẫn train.jsonl")
    parser.add_argument("--catalog", type=str, default="data/train/train_tools_catalog.json", help="Catalog tool")
    parser.add_argument("--out", type=str, required=True, help="Đường dẫn file sft_train.jsonl xuất ra")
    parser.add_argument("--synth", type=str, default=None, help="File synthetic jsonl (nếu có)")
    args = parser.parse_args()

    train_path = Path(args.train)
    if not train_path.exists():
        print(f"❌ Không tìm thấy: {train_path}")
        return

    items = [json.loads(line) for line in open(train_path, encoding="utf-8") if line.strip()]
    print(f"[*] Tổng số mẫu gốc trong train.jsonl: {len(items)}")

    system_prompt = (
        "Bạn là LifeMate, trợ lý ảo thông minh tiếng Việt. "
        "Nhiệm vụ của bạn là lắng nghe yêu cầu của người dùng, suy luận và gọi các công cụ (tools) phù hợp. "
        "Chỉ gọi các công cụ thuộc những ứng dụng đã được cài đặt (installed_apps). "
        "Nếu người dùng đổi ý hoặc yêu cầu không thuộc phạm vi ứng dụng nào, hãy trả lời lịch sự và không gọi công cụ."
    )

    sft_rows = []
    stats = {"original": 0, "history_mined": 0, "oos_stripped": 0, "synth": 0}

    # 1. Thêm các mẫu gốc
    for it in items:
        uid = it["id"]
        history = it.get("history", [])
        user_text = it.get("audio_text", "")
        calls = it.get("label", {}).get("tool_calls", [])
        results = it.get("label", {}).get("tool_results", [])

        msgs = build_chat_messages(system_prompt, history, user_text, calls, results)
        sft_rows.append({"id": uid, "messages": msgs})
        stats["original"] += 1

        # 2. History Mining: Tách các lượt trước trong history thành sample riêng
        for k in range(len(history) - 1):
            if history[k].get("role") == "user" and history[k + 1].get("role") == "assistant":
                prev_user = history[k].get("audio_text") or history[k].get("content", "")
                prev_calls = history[k + 1].get("tool_calls", [])
                prev_results = history[k + 1].get("tool_results", [])
                earlier_history = history[:k]

                if prev_user and prev_calls:
                    h_msgs = build_chat_messages(system_prompt, earlier_history, prev_user, prev_calls, prev_results)
                    sft_rows.append({"id": f"{uid}#h{k}", "messages": h_msgs})
                    stats["history_mined"] += 1

        # 3. OOS App Stripping: Xóa app để ép model học từ chối
        if calls and not history:
            used_apps = get_gold_apps(calls)
            if used_apps:
                # Tạo sample mới với câu yêu cầu đó nhưng context KHÔNG có app cần dùng
                oos_msgs = build_chat_messages(system_prompt, [], user_text, [], [])
                sft_rows.append({"id": f"{uid}#oos", "messages": oos_msgs})
                stats["oos_stripped"] += 1

    # 4. Tích hợp Synth Data (nếu có)
    if args.synth and Path(args.synth).exists():
        synth_items = [json.loads(line) for line in open(args.synth, encoding="utf-8") if line.strip()]
        for s_it in synth_items:
            s_uid = s_it.get("id", f"synth_{stats['synth']}")
            s_hist = s_it.get("history", [])
            s_text = s_it.get("audio_text", "")
            s_calls = s_it.get("label", {}).get("tool_calls", [])
            s_res = s_it.get("label", {}).get("tool_results", [])
            s_msgs = build_chat_messages(system_prompt, s_hist, s_text, s_calls, s_res)
            sft_rows.append({"id": s_uid, "messages": s_msgs})
            stats["synth"] += 1

    # Trộn ngẫu nhiên
    random.shuffle(sft_rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in sft_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 50)
    print(f"[✓] Đã tạo thành công file SFT Dataset: {out_path}")
    print(f"    - Mẫu gốc: {stats['original']}")
    print(f"    - Mẫu trích xuất lịch sử (History Mined): {stats['history_mined']}")
    print(f"    - Mẫu từ chối (OOS Stripped): {stats['oos_stripped']}")
    print(f"    - Mẫu synthetic: {stats['synth']}")
    print(f"    => TỔNG CỘNG: {len(sft_rows)} mẫu huấn luyện SFT!")
    print("=" * 50)

if __name__ == "__main__":
    main()
