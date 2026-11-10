import re
import json
import zipfile
from pathlib import Path

JH_PATH = Path(r"C:\Users\opc\Downloads\jh.txt")
DATA_ROOT = Path("data/public_test")
TEST_JSONL = DATA_ROOT / "public_test.jsonl"
OUT_SUBMISSION = Path(r"C:\Users\opc\Downloads\submission.jsonl")
OUT_ZIP = Path(r"C:\Users\opc\Downloads\submission.zip")

def main():
    print("Đang đọc file jh.txt...")
    with open(JH_PATH, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Pattern robust matching for sample blocks
    pattern = re.compile(
        r'\[Sample:\s*(public_\w+)\].*?'
        r'-\s*Công cụ gọi:\s*(\[.*?\])\s*\n'
        r'.*?'
        r'-\s*Mã băm:\s*([a-fA-F0-9]{32,64})',
        re.DOTALL
    )

    matches = pattern.findall(content)
    print(f"Tổng số mẫu tìm thấy trong log: {len(matches)}")

    parsed = {}
    for sid, calls_str, shash in matches:
        calls = []
        try:
            calls = json.loads(calls_str)
        except Exception as e:
            # Fallback json extraction if there's trailing garbage
            calls_str_clean = calls_str.strip()
            # find matching bracket
            end_idx = calls_str_clean.rfind(']')
            if end_idx != -1:
                try:
                    calls = json.loads(calls_str_clean[:end_idx+1])
                except Exception as e2:
                    print(f"Lỗi parse calls cho {sid}: {e2}")
            else:
                print(f"Lỗi parse calls cho {sid}: {e}")

        parsed[sid] = {
            "tool_calls": calls,
            "state_hash": shash
        }

    print(f"Tổng số ID độc nhất: {len(parsed)}")

    # Đọc danh sách chuẩn 634 ID của public_test
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        ordered_ids = [json.loads(line)["id"] for line in f if line.strip()]

    print(f"Tổng số ID chuẩn trong public_test.jsonl: {len(ordered_ids)}")

    missing = [sid for sid in ordered_ids if sid not in parsed]
    if missing:
        print(f"CẢNH BÁO: Còn thiếu {len(missing)} ID: {missing[:10]}")
    else:
        print("✅ Đầy đủ 100% (634/634) mẫu!")

    # Tạo submission.jsonl đúng thứ tự
    total_calls_count = 0
    with open(OUT_SUBMISSION, "w", encoding="utf-8") as out_f:
        for sid in ordered_ids:
            entry = parsed.get(sid, {"tool_calls": [], "state_hash": "0" * 64})
            calls = entry["tool_calls"]
            shash = entry["state_hash"]
            total_calls_count += len(calls)
            line_obj = {
                "id": sid,
                "tool_calls": calls,
                "state_hash": shash
            }
            out_f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")

    print(f"💾 Đã ghi file submission: {OUT_SUBMISSION}")
    print(f"   - Tổng số dòng: {len(ordered_ids)}")
    print(f"   - Tổng số tool calls: {total_calls_count}")

    # Đóng gói ZIP
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUT_SUBMISSION, arcname="submission.jsonl")

    print(f"📦 Đã đóng gói ZIP: {OUT_ZIP} ({OUT_ZIP.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
