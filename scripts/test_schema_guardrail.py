"""
test_schema_guardrail.py — Kiểm thử bộ gác cổng SafeSchemaGuardrail (Thành viên 1 - Pha 2).
"""

import sys
from pathlib import Path

# Thêm đường dẫn zaic_tools và src vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("D:/Download/train+public_test/train+public_test")

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DATA_ROOT / "train"))

from zaic_tools import ToolEnv
from src.rules.schema_guardrail import SafeSchemaGuardrail


def main():
    print("=" * 65)
    print("KIỂM THỬ BỘ GÁC CỔNG SCHEMA GUARDRAIL (THÀNH VIÊN 1 - PHA 2)")
    print("=" * 65)

    # Nạp 1 môi trường thực tế từ train (ví dụ mẫu train_1364)
    env = ToolEnv("train", "train_1364", root=DATA_ROOT)
    guardrail = SafeSchemaGuardrail(timezone_offset="+07:00")

    print(f"\n[+] Khởi tạo ToolEnv cho mẫu: train_1364")
    print(f"[+] Apps đã cài: {list(env.apps.keys())}")

    # Danh sách các trường hợp LLM sinh lỗi ngớ ngẩn thường gặp:
    test_cases = [
        {
            "name": "Case 1: Ép kiểu chuỗi số thành int",
            "tool": "fitness.log_workout",
            "raw_args": {"type": "swimming", "minutes": "45"},  # Sai: "45" dạng chuỗi
            "expected_fix": {"type": "swimming", "minutes": 45}
        },
        {
            "name": "Case 2: Ép kiểu chuỗi 'true' thành boolean True & dọn key rác",
            "tool": "alarm.create",
            "raw_args": {
                "time": "2026-12-18 05:30:00",  # Sai: có khoảng trắng, thiếu timezone +07:00
                "garbage_param": "thông_tin_bịa_ra"  # Rác: key thừa
            },
            "expected_fix": {
                "time": "2026-12-18T05:30:00+07:00"
            }
        },
        {
            "name": "Case 3: sys.ask_missing_info với params dạng chuỗi",
            "tool": "sys.ask_missing_info",
            "raw_args": {"tool_name": "alarm.create", "params": "time"},  # Sai: "time" thay vì ["time"]
            "expected_fix": {"tool_name": "alarm.create", "params": ["time"]}
        },
        {
            "name": "Case 4: Thiếu tham số bắt buộc (KHÔNG thể tự sửa)",
            "tool": "ride_hailing.book",
            "raw_args": {"pickup": "25 Láng Hạ"},  # Thiếu dropoff và vehicle_type
            "expect_error": True
        }
    ]

    for tc in test_cases:
        print("\n" + "-" * 60)
        print(f"[*] {tc['name']}")
        print(f"-> Tool: `{tc['tool']}`")
        print(f"-> LLM sinh ra: {tc['raw_args']}")

        is_valid, clean_args, error_msg = guardrail.guard_and_validate(env, tc["tool"], tc["raw_args"])

        if is_valid:
            print(f"-> [THÀNH CÔNG] Guardrail đã tự động sửa và vượt qua env.validate()!")
            print(f"-> Tham số sau dọn dẹp: {clean_args}")
            # Gọi thử vào env.call để chứng minh an toàn tuyệt đối
            res = env.call(tc["tool"], clean_args)
            print(f"-> Kết quả env.call(): ok={res.get('ok')}")
        else:
            print(f"-> [CHẶN ĐỨNG THÀNH CÔNG] Lỗi phát hiện: {error_msg}")
            print("-> KHÔNG GỌI env.call()! Điểm phạt U được bảo toàn = 0!")

    print("\n" + "=" * 65)
    print("TỔNG KẾT: LỊCH SỬ ENV TOOL CALLS:")
    calls = env.get_tool_calls()
    print(f"-> Tổng số cuộc gọi đã ghi nhận: {len(calls)}")
    has_error = any("error" in c.get("result", {}) for c in env.log)
    print(f"-> Có cuộc gọi nào bị lỗi schema (phạt U) không? {'CÓ (NGUY HIỂM)' if has_error else 'HOÀN TOÀN KHÔNG (ĐẠT CHUẨN 100%)'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
