"""
schema_guardrail.py — Bộ gác cổng kiểm tra và tự động sửa lỗi Schema (Pha 2 - Thành viên 1).
Kiểm tra schema trước khi thực thi, giảm lời gọi bị từ chối.
"""

import re
import math
import copy
from typing import Dict, Any, Tuple, Optional


class SafeSchemaGuardrail:
    """
    Lá chắn an toàn đứng trước ToolEnv:
    1. Tự động sửa lỗi kỹ thuật định dạng (Type Casting, Datetime ISO-8601, dọn dẹp key rác).
    2. Chạy thử vô hại bằng `env.validate()` (KHÔNG bị ghi vào bài nộp, KHÔNG bị phạt U).
    3. Chặn lời gọi không vượt qua validate; lỗi thực thi vẫn được ghi nhận.
    """

    def __init__(self, timezone_offset: str = "+07:00"):
        self.timezone_offset = timezone_offset
        self._tool_specs = {}

    def bind(self, tools):
        """Bind the public catalog per sample without inspecting ToolEnv internals."""
        bound = copy.copy(self)
        bound._tool_specs = {tool['name']: copy.deepcopy(tool) for tool in tools}
        return bound

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
        # color, title, etc.). Only a closed schema permits removing extra keys.

        if not tool_spec:
            return clean_args

        return self._repair_schema(clean_args, tool_spec.get("parameters", {}))

    def _repair_schema(self, value, schema, key=""):
        """Repair nested declared fields without manufacturing missing values."""
        if not isinstance(schema, dict):
            return value
        # Branch selection and references belong to the authoritative validator.
        # Guessing a branch can erase fields which another branch requires.
        if any(k in schema for k in ('$ref', 'allOf', 'anyOf', 'oneOf', 'if', 'patternProperties')):
            return value
        if value is None:
            return None
        kind = schema.get("type")
        if kind == "object" or "properties" in schema:
            if not isinstance(value, dict):
                return value
            properties = schema.get("properties", {})
            repaired = {}
            for name, child in value.items():
                if name in properties:
                    repaired[name] = self._repair_schema(child, properties[name], name)
                elif schema.get("additionalProperties", True) is True:
                    repaired[name] = child
                elif isinstance(schema.get("additionalProperties"), dict):
                    repaired[name] = self._repair_schema(child, schema["additionalProperties"], name)
            return repaired
        if kind == "array":
            items = schema.get("items", {})
            # Do not turn a scalar into a list: that changes argument structure.
            if not isinstance(value, list):
                return value
            if isinstance(items, dict):
                return [self._repair_schema(child, items, key) for child in value]
            return value
        return self._repair_value(key, value, kind, schema.get("format"))

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
                return val
            if isinstance(val, str):
                try:
                    number = float(val.strip())
                    return number if math.isfinite(number) else val
                except ValueError:
                    pass

        # Ép kiểu boolean
        elif prop_type == "boolean":
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                s = val.strip().lower()
                if s in ("true", "1", "yes", "có", "bật", "on"):
                    return True
                if s in ("false", "0", "no", "không", "tắt", "off"):
                    return False
            if isinstance(val, (int, float)) and val in (0, 1):
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
        if not isinstance(arguments, dict):
            return False, {}, "Arguments must be an object"
        # A valid argument object must survive unchanged: even a harmless-looking
        # normalization can change an exact-match field or the final state hash.
        original = copy.deepcopy(arguments)
        check_result = env.validate(tool_name, original)
        if check_result.get("ok", False):
            return True, original, None
        tool_spec = self._tool_specs.get(tool_name)
        clean_args = self.sanitize_arguments(tool_name, arguments, tool_spec)
        check_result = env.validate(tool_name, clean_args)

        if check_result.get("ok", False):
            return True, clean_args, None
        else:
            err_msg = check_result.get("error", "Lỗi schema không xác định")
            return False, clean_args, err_msg
