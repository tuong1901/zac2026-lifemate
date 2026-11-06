"""
schema_guardrail.py — Bộ gác cổng kiểm tra và tự động sửa lỗi Schema (Pha 2 - Thành viên 1).
Bảo vệ tuyệt đối Agent không bị phạt điểm U (Penalized Calls).
"""

import re
from typing import Dict, Any, Tuple, Optional


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
            if "params" in clean_args:
                if isinstance(clean_args["params"], str):
                    clean_args["params"] = [clean_args["params"]]
                elif not isinstance(clean_args["params"], list):
                    clean_args["params"] = []
            return clean_args

        # Optional schema fields can encode explicit user intent (speed, repeat,
        # color, title, etc.). Preserve them; only undeclared keys are removed below.

        if tool_name in ("sms.send", "phone.call"):
            if str(clean_args.get("contact_id", "")).startswith("ct_") and "phone_number" in clean_args:
                clean_args.pop("phone_number", None)
        elif tool_name == "ewallet.topup_mobile":
            clean_args.pop("contact_id", None)

        if not tool_spec:
            return clean_args

        spec_params = tool_spec.get("parameters", {})
        properties = spec_params.get("properties", {})

        sanitized: Dict[str, Any] = {}

        # 1. Dọn dẹp key rác (chỉ giữ lại các trường có trong properties của schema)
        for key, val in clean_args.items():
            if key not in properties:
                continue

            prop_def = properties[key]
            prop_type = prop_def.get("type")
            prop_format = prop_def.get("format")

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
                return val  # Let schema validation reject bool-as-integer.
            if isinstance(val, int):
                return val
            if isinstance(val, float) and val.is_integer():
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
            is_datetime_key = key in ("time", "start", "end", "due", "since", "valid_until", "schedule_time")
            is_date_key = key in ("date", "date_from", "date_to")

            # Format date: YYYY-MM-DD
            if prop_format == "date" or (is_date_key and not prop_format):
                if len(val_str) >= 10 and re.match(r"^\d{4}-\d{2}-\d{2}", val_str):
                    return val_str[:10]

            # Format date-time: ISO-8601 kèm offset
            elif prop_format == "date-time" or (is_datetime_key and re.match(r"^\d{4}-\d{2}-\d{2}", val_str)):
                if " " in val_str and "T" not in val_str:
                    val_str = val_str.replace(" ", "T")
                if not re.search(r"(Z|[+-]\d{2}:?\d{2})$", val_str):
                    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$", val_str):
                        val_str = f"{val_str}:00"
                    val_str = f"{val_str}{self.timezone_offset}"
                return val_str

            # Format time: HH:MM
            elif prop_format == "time":
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
        """
        tool_spec = None
        if hasattr(env, "tools") and isinstance(env.tools, dict):
            tool_spec = env.tools.get(tool_name)

        clean_args = self.sanitize_arguments(tool_name, arguments, tool_spec)
        check_result = env.validate(tool_name, clean_args)

        if check_result.get("ok", False):
            return True, clean_args, None
        else:
            err_msg = check_result.get("error", "Lỗi schema không xác định")
            return False, clean_args, err_msg
