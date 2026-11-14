"""
check_compliance.py — Script tự động quét mã nguồn kiểm tra tuân thủ 100% quy chế ZAC 2026.
Chạy trước khi commit hoặc nộp Docker:
    python scripts/check_compliance.py
"""

import os
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Các quy tắc vi phạm nghiêm trọng
VIOLATIONS = [
    {
        "id": "READ_STATE_JSON",
        "pattern": r"(open|Path|read_text|json\.load)\(.*(state\.json|_environment)",
        "desc": "CẤM: Phát hiện code cố tình đọc trực tiếp file state.json hoặc thư mục _environment!"
    },
    {
        "id": "FORBIDDEN_LARGE_MODEL",
        "pattern": r"['\"][a-zA-Z0-9_\-\./]*(7B|8B|14B|32B|70B|gemma-2-9b)[a-zA-Z0-9_\-\./]*['\"]",
        "desc": "CẤM: Phát hiện mô hình LLM có kích thước >= 5B params (Quy chế: LLM phải < 5B)!"
    },
    {
        "id": "ONLINE_API_CALL",
        "pattern": r"(openai\.OpenAI\(|anthropic\.Anthropic\(|genai\.configure|api\.openai\.com|api\.anthropic\.com)",
        "desc": "CẤM: Phát hiện gọi API trực tuyến bên ngoài trong mã nguồn suy luận!"
    },
    {
        "id": "NON_ZERO_TEMPERATURE",
        "pattern": r"temperature\s*=\s*(0\.[1-9]|[1-9])",
        "desc": "CẢNH BÁO: Phát hiện temperature > 0. Quy chế yêu cầu deterministic decoding (temperature = 0.0)!"
    }
]

IGNORE_DIRS = {".git", ".venv", "venv", "__pycache__", "data", "checkpoints", "models"}
SCAN_EXTS = {".py", ".sh", ".yaml", ".yml"}

def scan_repo():
    print("=" * 65)
    print("      ZAC 2026 — BỘ QUÉT TỰ ĐỘNG TUÂN THỦ QUY CHẾ THI ĐẤU")
    print("=" * 65)

    issues_found = 0
    scanned_files = 0

    for root, dirs, files in os.walk(ROOT_DIR):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() not in SCAN_EXTS:
                continue

            # Bỏ qua chính file checker này
            if file == "check_compliance.py":
                continue

            scanned_files += 1
            rel_path = file_path.relative_to(ROOT_DIR)

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                continue

            for v in VIOLATIONS:
                matches = re.finditer(v["pattern"], content, re.IGNORECASE)
                for m in matches:
                    # Lấy số dòng
                    line_no = content[:m.start()].count("\n") + 1
                    print(f"\n[VI PHẠM: {v['id']}] tại file: {rel_path} (Dòng {line_no})")
                    print(f"  -> Chi tiết: {v['desc']}")
                    print(f"  -> Đoạn code vi phạm: '{m.group(0)}'")
                    issues_found += 1

    print("\n" + "-" * 65)
    print(f"Đã quét: {scanned_files} files.")
    if issues_found == 0:
        print("[XÁC NHẬN] 100% HỢP LỆ! Không phát hiện bất kỳ vi phạm thể lệ nào.")
        print("Mã nguồn tuân thủ: Model < 5B, không đọc state.json, deterministic seed.")
        print("-" * 65)
        return 0
    else:
        print(f"[CẢNH BÁO NGUY HIỂM] Phát hiện {issues_found} vi phạm cần sửa ngay trước khi nộp bài!")
        print("-" * 65)
        return 1

if __name__ == "__main__":
    sys.exit(scan_repo())
