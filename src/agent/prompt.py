"""
prompt.py — Quản lý System Prompt và Prompt Builder cho Agent ZAC 2026 LifeMate.
Tối ưu hóa đặc biệt cho Qwen 2.5 3B: CoT ngắn gọn, JSON nghiêm ngặt, tích hợp RAG & History Memory.
"""

from typing import Dict, List, Any, Optional

SYSTEM_PROMPT = """Bạn là trợ lý ảo thông minh tiếng Việt LifeMate (ZAC 2026).
Nhiệm vụ của bạn là nhận khẩu lệnh của người dùng, phân tích ngữ cảnh và các ví dụ tham khảo, sau đó quyết định gọi các công cụ (tools) một cách chuẩn xác.

QUY TẮC CỐT LÕI (BẮT BUỘC TUÂN THỦ):
1. Bạn CHỈ ĐƯỢC PHÉP gọi các tool có trong danh sách [DANH SÁCH CÔNG CỤ ĐƯỢC PHÉP]. Tuyệt đối không tự bịa tool.
2. Nếu lệnh của người dùng THIẾU THAM SỐ BẮT BUỘC và không thể tìm thấy trong ngữ cảnh/lịch sử/tool đọc:
   -> Gọi công cụ hệ thống `sys.ask_missing_info` với cú pháp:
      {"tool_name": "<tên_tool_bị_thiếu>", "params": ["<danh_sách_param_còn_thiếu>"]}
3. Nếu yêu cầu của người dùng NẰM NGOÀI PHẠM VI các app đã cài hoặc không có tool nào phù hợp:
   -> Đặt "action": null và "is_finished": true (không gọi tool nào, trả về []).
4. Nếu yêu cầu cần nhiều bước phụ thuộc:
   - Nếu cần ID hoặc thông tin chưa rõ (ví dụ: tìm số liên lạc trước khi gọi điện/nhắn tin, xem danh sách khoá trước khi khoá cửa):
     -> Hãy gọi công cụ tìm kiếm / liệt kê trước.

ĐỊNH DẠNG ĐẦU RA:
Bạn phải trả về DUY NHẤT một khối JSON hợp lệ (không kèm lời dẫn bên ngoài):
{
  "thought": "Suy luận ngắn gọn lý do chọn hành động này",
  "action": {
    "name": "<tên_tool>",
    "arguments": { ... }
  },
  "is_finished": false
}

Khi đã hoàn thành tất cả các bước hoặc khi yêu cầu ngoài phạm vi, trả về:
{
  "thought": "Đã hoàn thành toàn bộ yêu cầu",
  "action": null,
  "is_finished": true
}
"""


def compact_observation(result: Any, max_length: int = 400) -> str:
    """
    Rút gọn kết quả trả về của tool để tránh làm phình to context tokens.
    """
    if result is None:
        return "null"
    import json
    if isinstance(result, (dict, list)):
        compact_str = json.dumps(result, ensure_ascii=False)
    else:
        compact_str = str(result)
        
    if len(compact_str) > max_length:
        return compact_str[:max_length] + "... (đã rút gọn)"
    return compact_str


def build_user_prompt(
    user_utterance: str,
    context: Dict[str, Any],
    history_entities: str,
    tools_description: str,
    few_shot_examples: str,
    previous_steps: List[Dict[str, Any]]
) -> str:
    """
    Ghép nối toàn bộ ngữ cảnh thành User Prompt hoàn chỉnh cho LLM.
    """
    current_time = context.get("current_time", "N/A")
    user_profile = context.get("user_profile", {})

    prompt_parts = [
        f"[NGỮ CẢNH HIỆN TẠI]",
        f"- Thời điểm hiện tại: {current_time}",
        f"- Thông tin người dùng: {user_profile}",
        "",
        f"[THÔNG TIN TRÍCH XUẤT TỪ LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ]",
        history_entities if history_entities else "Không có lịch sử trước đó.",
        "",
        f"[VÍ DỤ THỰC TẾ TƯƠNG TỰ ĐỂ THAM KHẢO]",
        few_shot_examples if few_shot_examples else "Không có ví dụ mẫu.",
        "",
        f"[DANH SÁCH CÔNG CỤ ĐƯỢC PHÉP DÙNG]",
        tools_description,
        ""
    ]

    # Trạng thái thực thi các bước trước đó trong lượt này
    if previous_steps:
        prompt_parts.append("[CÁC BƯỚC ĐÃ THỰC HIỆN TRONG LƯỢT NÀY]")
        for step in previous_steps:
            s_idx = step.get("step", 1)
            t_name = step.get("tool_called", "")
            t_args = step.get("arguments", {})
            obs = step.get("observation", step.get("result", ""))
            err = step.get("error")
            
            if err:
                prompt_parts.append(f"- Bước {s_idx}: Gọi `{t_name}({t_args})` -> [LỖI SCHEMA]: {err} (Hãy sửa tham số hoặc gọi tool tìm kiếm)")
            else:
                prompt_parts.append(f"- Bước {s_idx}: Gọi `{t_name}({t_args})` -> [KẾT QUẢ]: {obs}")
        prompt_parts.append("")

    prompt_parts.append(f"[YÊU CẦU CỦA NGƯỜI DÙNG]")
    prompt_parts.append(f"\"{user_utterance}\"")
    prompt_parts.append("")
    prompt_parts.append("Hãy suy luận bước tiếp theo và trả về DUY NHẤT một khối JSON hợp lệ:")

    return "\n".join(prompt_parts)
