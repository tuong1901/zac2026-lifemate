"""
schema_guardrail.py — Bộ gác cổng kiểm tra và tự động sửa lỗi Schema (Pha 2 - Thành viên 1).
Bảo vệ tuyệt đối Agent không bị phạt điểm U (Penalized Calls).
"""

import re
from typing import Dict, Any, Tuple, Optional, List


class SafeSchemaGuardrail:
    """
    Lá chắn an toàn đứng trước ToolEnv:
    1. Tự động sửa lỗi kỹ thuật định dạng (Type Casting, Datetime ISO-8601, dọn dẹp key rác).
    2. Chạy thử vô hại bằng `env.validate()` (KHÔNG bị ghi vào bài nộp, KHÔNG bị phạt U).
    3. Chặn đứng 100% các call lỗi không cho lọt vào `env.call()`.
    """

    def __init__(self, timezone_offset: str = "+07:00"):
        self.timezone_offset = timezone_offset

    def sanitize_arguments(
        self,
        tool_name: str,
        arguments: Any,
        tool_spec: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Tự động sửa lỗi tham số theo đặc tả JSON Schema của tool.
        """
        if not isinstance(arguments, dict):
            return {}

        clean_args = dict(arguments)

        # Trường hợp đặc biệt: sys.ask_missing_info
        if tool_name == "sys.ask_missing_info":
            # params bắt buộc phải là list[str]
            if "params" in clean_args:
                if isinstance(clean_args["params"], str):
                    clean_args["params"] = [clean_args["params"]]
                elif not isinstance(clean_args["params"], list):
                    clean_args["params"] = []
            return clean_args

        if not tool_spec:
            return clean_args

        spec_params = tool_spec.get("parameters", {})
        properties = spec_params.get("properties", {})
        required = set(spec_params.get("required", []))

        sanitized: Dict[str, Any] = {}

        # 1. Dọn dẹp key rác (chỉ giữ lại các trường có trong properties của schema)
        for key, val in clean_args.items():
            if key not in properties:
                # Bỏ qua key lạ không được định nghĩa trong schema
                continue

            prop_def = properties[key]
            prop_type = prop_def.get("type")
            prop_format = prop_def.get("format")

            # Xử lý giá trị rỗng cho trường không bắt buộc
            if val in (None, "", [], {}) and key not in required:
                continue

            fixed_val = self._repair_value(key, val, prop_type, prop_format)
            sanitized[key] = fixed_val

        return sanitized

    def _repair_value(self, key: str, val: Any, prop_type: Optional[str], prop_format: Optional[str]) -> Any:
        """
        Tự động ép kiểu và chuẩn hóa giá trị cụ thể.
        """
        # Ép kiểu integer
        if prop_type == "integer":
            if isinstance(val, bool):
                return 1 if val else 0
            if isinstance(val, (int, float)):
                return int(val)
            if isinstance(val, str):
                s = val.strip()
                if re.match(r"^-?\d+$", s):
                    return int(s)

        # Ép kiểu number (float/int)
        elif prop_type == "number":
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val)
            if isinstance(val, str):
                try:
                    return float(val.strip())
                except ValueError:
                    pass

        # Ép kiểu boolean
        elif prop_type == "boolean":
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                s = val.strip().lower()
                if s in ("true", "1", "yes", "có", "bật", "khoá", "mở", "on"):
                    return True
                if s in ("false", "0", "no", "không", "tắt", "off"):
                    return False
            if isinstance(val, (int, float)):
                return bool(val)

        # Ép kiểu array
        elif prop_type == "array":
            if not isinstance(val, list):
                return [val] if val is not None else []

        # Chuẩn hóa format ngày giờ
        elif prop_type == "string":
            val_str = str(val).strip()
            is_datetime_key = key in ("time", "start", "end", "due", "since", "valid_until")
            is_date_key = key in ("date",)

            # Format date: YYYY-MM-DD
            if prop_format == "date" or (is_date_key and not prop_format):
                # Nếu LLM truyền dạng 2026-11-16T... -> cắt lấy 10 ký tự đầu
                if len(val_str) >= 10 and re.match(r"^\d{4}-\d{2}-\d{2}", val_str):
                    return val_str[:10]

            # Format date-time: ISO-8601 kèm offset
            elif prop_format == "date-time" or (is_datetime_key and re.match(r"^\d{4}-\d{2}-\d{2}", val_str)):
                # Nếu có dấu cách thay vì T: 2026-11-16 17:00:00 -> 2026-11-16T17:00:00
                if " " in val_str and "T" not in val_str:
                    val_str = val_str.replace(" ", "T")
                
                # Nếu chưa có timezone offset (+07:00 hoặc Z) -> tự thêm timezone
                if not re.search(r"(Z|[+-]\d{2}:?\d{2})$", val_str):
                    # Nếu thiếu giây: 2026-11-16T17:00 -> thêm :00
                    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$", val_str):
                        val_str = f"{val_str}:00"
                    val_str = f"{val_str}{self.timezone_offset}"
                return val_str

            # Format time: HH:MM
            elif prop_format == "time":
                # Nếu truyền dạng "5:30" -> đệm thành "05:30"
                if re.match(r"^\d:\d{2}", val_str):
                    val_str = f"0{val_str}"
                return val_str

        return val

    def guard_and_validate(
        self,
        env: Any,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Kiểm tra an toàn:
        1. Tự động sửa lỗi tham số (Auto-sanitize).
        2. Chạy thử nghiệm bằng env.validate() (KHÔNG gây phạt U).
        
        Returns:
            (is_valid, clean_args, error_message)
        """
        # Lấy spec của tool từ env (nếu có)
        tool_spec = None
        if hasattr(env, "tools") and isinstance(env.tools, dict):
            tool_spec = env.tools.get(tool_name)

        # 1. Tự động dọn rác và ép kiểu
        clean_args = self.sanitize_arguments(tool_name, arguments, tool_spec)

        # 2. Kiểm tra schema không mất phí bằng env.validate()
        check_result = env.validate(tool_name, clean_args)

        if check_result.get("ok", False):
            # Lệnh hợp lệ 100%, an toàn tuyệt đối để gọi env.call()
            return True, clean_args, None
        else:
            # Lệnh sai schema -> Trả về lỗi để Agent ReAct xử lý, TUYỆT ĐỐI KHÔNG GỌI env.call()
            err_msg = check_result.get("error", "Lỗi schema không xác định")
            return False, clean_args, err_msg
