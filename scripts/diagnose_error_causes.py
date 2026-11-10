"""
scripts/diagnose_error_causes.py — Bộ Phân Tích & Bóc Tách Nguyên Nhân Lỗi (Root Cause Diagnosis).
Đọc file diagnostic_report.jsonl và phân loại cụ thể từng mẫu thất bại:
1. Lỗi do ASR gỡ băng sai/nhiễu (ASR Noise)
2. Lỗi do Router chọn thiếu Domain/App (Router Domain Miss)
3. Lỗi do LLM lập kế hoạch/gọi sai tham số (Planner / Arg Error)
4. Lỗi do Cổng sinh tử State Hash (State Hash Mismatch)
5. Thành công (Exact Match / Perfect F1)
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List

def main():
    parser = argparse.ArgumentParser(description="Chẩn đoán nguyên nhân lỗi V2")
    parser.add_argument("--diagnostic_file", type=str, required=True, help="Đường dẫn file diagnostic_report.jsonl")
    args = parser.parse_args()

    diag_path = Path(args.diagnostic_file)
    if not diag_path.exists():
        print(f"❌ Không tìm thấy file: {diag_path}")
        return

    total = 0
    categories = {
        "PERFECT_SUCCESS": [],        # Điểm = 1.0 (Khớp 100% cả tool và state)
        "PARTIAL_SUCCESS": [],        # 0.0 < Điểm < 1.0
        "ROUTER_MISSED_APP": [],      # Router không chọn app mà đáp án yêu cầu
        "STATE_MISMATCH_ONLY": [],    # Tool gọi đúng nhưng state hash sai (do arg sai / thiếu bước)
        "LLM_WRONG_TOOL_OR_ARGS": [], # Router chọn đúng app nhưng LLM gọi tool khác hoặc sai arg
        "NO_TOOL_CALLED": [],         # Lẽ ra phải gọi tool nhưng agent trả về rỗng []
        "OUT_OF_SCOPE_SUCCESS": [],   # Câu không có tác vụ và agent trả về [] chuẩn xác
    }

    with open(diag_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            rec = json.loads(line)
            sid = rec.get("id")
            user_text = rec.get("user_text", "")
            sel_apps = set(rec.get("selected_apps", []))
            exp_apps = set(rec.get("expanded_apps", []))
            pred_calls = rec.get("tool_calls", [])
            shash = rec.get("state_hash", "")
            ev = rec.get("eval")

            if not ev:
                # Không có ground truth (ví dụ chạy trên public_test)
                if not pred_calls:
                    categories["NO_TOOL_CALLED"].append({
                        "id": sid,
                        "user_text": user_text,
                        "selected_apps": list(sel_apps),
                        "reason": "Agent không gọi tool nào (No-Op)"
                    })
                else:
                    categories["PERFECT_SUCCESS"].append({
                        "id": sid,
                        "user_text": user_text,
                        "tool_calls": pred_calls,
                    })
                continue

            score = ev.get("final_score", 0.0)
            state_match = ev.get("state_match", False)
            gold_calls = ev.get("gold_calls", [])
            gold_apps = {c.get("name", "").split(".")[0] for c in gold_calls if "." in c.get("name", "")}

            if len(gold_calls) == 0:
                if len(pred_calls) == 0:
                    categories["OUT_OF_SCOPE_SUCCESS"].append(sid)
                else:
                    categories["LLM_WRONG_TOOL_OR_ARGS"].append({
                        "id": sid, "user_text": user_text, "reason": "Câu out-of-scope nhưng LLM lại gọi tool"
                    })
                continue

            if score == 1.0:
                categories["PERFECT_SUCCESS"].append(sid)
            elif score > 0.0:
                categories["PARTIAL_SUCCESS"].append({"id": sid, "score": score})
            else:
                # Score == 0.0: Tìm nguyên nhân cốt lõi
                if not pred_calls:
                    categories["NO_TOOL_CALLED"].append({
                        "id": sid,
                        "user_text": user_text,
                        "gold_apps": list(gold_apps),
                        "selected_apps": list(sel_apps),
                    })
                elif not gold_apps.issubset(exp_apps):
                    missing_apps = gold_apps - exp_apps
                    categories["ROUTER_MISSED_APP"].append({
                        "id": sid,
                        "user_text": user_text,
                        "missing_apps": list(missing_apps),
                        "selected_apps": list(sel_apps),
                    })
                elif not state_match:
                    categories["STATE_MISMATCH_ONLY"].append({
                        "id": sid,
                        "user_text": user_text,
                        "pred_calls": pred_calls,
                        "gold_calls": gold_calls,
                    })
                else:
                    categories["LLM_WRONG_TOOL_OR_ARGS"].append({
                        "id": sid,
                        "user_text": user_text,
                        "pred_calls": pred_calls,
                        "gold_calls": gold_calls,
                    })

    print("=" * 75)
    print("      BÁO CÁO PHÂN TÍCH NGUYÊN NHÂN LỖI (ROOT CAUSE DIAGNOSIS)")
    print("=" * 75)
    print(f" Tổng số mẫu phân tích:               {total}")
    print(f" ✅ Khớp hoàn hảo 100% (Perfect):     {len(categories['PERFECT_SUCCESS'])} ({len(categories['PERFECT_SUCCESS'])/max(1, total)*100:.1f}%)")
    print(f" ⚠️ Khớp một phần (Partial > 0):       {len(categories['PARTIAL_SUCCESS'])} ({len(categories['PARTIAL_SUCCESS'])/max(1, total)*100:.1f}%)")
    print(f" ❌ Trả về rỗng khi có lệnh (No-Op):  {len(categories['NO_TOOL_CALLED'])} ({len(categories['NO_TOOL_CALLED'])/max(1, total)*100:.1f}%)")
    print(f" ❌ Router chọn thiếu App (Domain):   {len(categories['ROUTER_MISSED_APP'])} ({len(categories['ROUTER_MISSED_APP'])/max(1, total)*100:.1f}%)")
    print(f" ❌ Lỗi Cổng sinh tử (State Mismatch):{len(categories['STATE_MISMATCH_ONLY'])} ({len(categories['STATE_MISMATCH_ONLY'])/max(1, total)*100:.1f}%)")
    print(f" ❌ LLM gọi sai Tool / Tham số:       {len(categories['LLM_WRONG_TOOL_OR_ARGS'])} ({len(categories['LLM_WRONG_TOOL_OR_ARGS'])/max(1, total)*100:.1f}%)")
    print("=" * 75)

    if categories["ROUTER_MISSED_APP"]:
        print("\n🔍 VÍ DỤ 3 MẪU LỖI DO ROUTER CHỌN THIẾU APP:")
        for ex in categories["ROUTER_MISSED_APP"][:3]:
            print(f"  - [{ex['id']}] Text: \"{ex['user_text']}\"")
            print(f"    Thiếu app: {ex['missing_apps']} | Router chọn: {ex['selected_apps']}")

    if categories["NO_TOOL_CALLED"]:
        print("\n🔍 VÍ DỤ 3 MẪU BỊ TRẢ VỀ RỖNG (NO-OP):")
        for ex in categories["NO_TOOL_CALLED"][:3]:
            print(f"  - [{ex['id']}] Text: \"{ex['user_text']}\"")
            if "gold_apps" in ex:
                print(f"    Lẽ ra cần app: {ex['gold_apps']} | Router chọn: {ex['selected_apps']}")

if __name__ == "__main__":
    main()
