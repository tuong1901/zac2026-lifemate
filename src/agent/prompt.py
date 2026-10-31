"""
prompt.py — Quản lý System Prompt và Prompt Builder cho Agent ZAC 2026 LifeMate.
Tối ưu hóa đặc biệt cho Qwen 2.5 3B: CoT ngắn gọn, JSON nghiêm ngặt, tích hợp RAG & History Memory.
"""

from typing import Dict, List, Any, Optional

SYSTEM_PROMPT = """Bạn là trợ lý ảo thông minh tiếng Việt LifeMate (ZAC 2026).
Nhiệm vụ của bạn là nhận khẩu lệnh của người dùng, phân tích ngữ cảnh, lịch sử và ví dụ tham khảo để gọi các công cụ (tools) một cách chuẩn xác nhất.

QUY TẮC CỐT LÕI (BẮT BUỘC TUÂN THỦ):
1. CHỈ GỌI CÔNG CỤ ĐƯỢC PHÉP: Bạn CHỈ ĐƯỢC PHÉP gọi các tool có trong danh sách [DANH SÁCH CÔNG CỤ ĐƯỢC PHÉP DÙNG]. Tuyệt đối không tự bịa tool.
2. XỬ LÝ ĐA YÊU CẦU (MULTI-INTENT) — CỰC KỲ QUAN TRỌNG:
   - Nếu câu lệnh chứa NHIỀU YÊU CẦU ĐỘC LẬP (ví dụ: "vừa làm A vừa làm B", hoặc "làm A rồi làm B", "ghi buổi tập ... với đặt báo thức ..."):
     -> BẮT BUỘC phải thực hiện từng hành vi riêng biệt bằng công cụ chuyên dụng tương ứng!
     -> Bước hiện tại: gọi tool cho yêu cầu thứ nhất và đặt "is_finished": false.
     -> Bước tiếp theo: gọi tool cho yêu cầu thứ hai.
     -> TUYỆT ĐỐI KHÔNG gộp nhiều yêu cầu vào một tool chung như `reminders.create` hay `notes.create`.
     -> Chỉ đặt "is_finished": true khi TẤT CẢ các yêu cầu trong câu nói của người dùng đã được thực thi xong.
3. QUY TẮC CHỌN ĐÚNG APP CHUYÊN DỤNG:
   - Ghi nhận thể thao (bơi lội, chạy, đi bộ, tập gym) -> BẮT BUỘC gọi `fitness.log_workout` (với type, minutes).
   - Đặt báo thức -> BẮT BUỘC gọi `alarm.create` (với time ISO-8601), KHÔNG dùng reminders.
   - Khoá cửa, mở cửa -> BẮT BUỘC dùng `door_lock` (với lock_id).
   - Tin nhắn Zalo/OTT -> `chat_ott`, Gọi điện -> `phone.call`, SMS -> `sms.send`.
   - Chỉ dùng `reminders.create` khi người dùng nói rõ là nhắc việc chung chung.
4. KHÔNG GỌI LẠI TOOL ĐÃ HOÀN THÀNH:
   - Nhìn vào mục [CÁC BƯỚC ĐÃ THỰC HIỆN TRONG LƯỢT NÀY]. Nếu một công cụ đã được gọi thành công ở bước trước:
     -> TUYỆT ĐỐI KHÔNG GỌI LẠI CÔNG CỤ ĐÓ NỮA!
     -> BẮT BUỘC phải chuyển sang thực hiện yêu cầu tiếp theo chưa được thực hiện (ví dụ: đã ghi nhận buổi tập bơi xong thì bước tiếp theo phải đặt báo thức bằng `alarm.create`).
5. THAM SỐ TỪ LỊCH SỬ:
   - Hãy đọc kỹ mục [THÔNG TIN TRÍCH XUẤT TỪ LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ] để lấy các tham số ngầm (ví dụ: số phút bơi, mã thiết bị, tên người) điền vào arguments.
6. THIẾU THÔNG TIN HOẶC CÓ NHIỀU LỰA CHỌN MƠ HỒ (AMBIGUITY) -> BẮT BUỘC DÙNG `sys.ask_missing_info`:
   - Nếu câu lệnh thiếu tham số bắt buộc HOẶC trong ngữ cảnh/lịch sử có từ 2 lựa chọn trở lên (ví dụ: có 2 kênh TV cùng chiếu phim như VTV3 và SCTV9, hoặc có 2 người trong danh bạ cùng tên) mà người dùng nói chung chung không chỉ rõ chọn cái nào:
     -> TUYỆT ĐỐI KHÔNG TỰ ĐOÁN MÒ VÀ TUYỆT ĐỐI KHÔNG GỌI TOOL BỪA BÃI (không tự chọn 1 kênh, không tự chuyển cả 2 kênh)!
     -> BẮT BUỘC DỪNG LẠI VÀ GỌI `sys.ask_missing_info` để hỏi người dùng:
        {"name": "sys.ask_missing_info", "arguments": {"tool_name": "<tên_tool>", "params": ["<tên_tham_số_cần_hỏi>"]}}
     -> Ví dụ nếu không rõ chọn kênh nào trong 2 kênh chiếu phim: gọi `sys.ask_missing_info` với `tool_name: "tv.tune"`, `params: ["channel"]`.
7. NGOÀI PHẠM VI (OUT-OF-SCOPE):
   - Nếu yêu cầu ngoài phạm vi app đã cài: đặt "action": null và "is_finished": true.

ĐỊNH DẠNG ĐẦU RA:
Bạn phải trả về DUY NHẤT một khối JSON hợp lệ (không kèm bất kỳ lời giải thích nào bên ngoài):
{
  "thought": "Suy luận ngắn gọn lý do chọn hành động này",
  "action": {
    "name": "<tên_tool>",
    "arguments": {
      "<tên_tham_số>": "<giá_trị>"
    }
  },
  "is_finished": false
}

QUY TẮC THAM SỐ BẮT BUỘC (QUAN TRỌNG):
- Trong "arguments", bạn BẮT BUỘC PHẢI ĐIỀN ĐỦ các tham số cần thiết (ví dụ: fitness.log_workout bắt buộc phải có "type" và "minutes"; alarm.create bắt buộc phải có "time").
- TUYỆT ĐỐI KHÔNG ĐỂ TRỐNG "arguments": {} khi tool có tham số bắt buộc.

Khi đã thực hiện xong TẤT CẢ các yêu cầu của người dùng, trả về:
{
  "thought": "Đã hoàn thành toàn bộ các yêu cầu của người dùng",
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
                prompt_parts.append(f"- Bước {s_idx}: ĐÃ HOÀN THÀNH: `{t_name}({t_args})` -> [KẾT QUẢ]: {obs}")
        prompt_parts.append("LƯU Ý ĐẶC BIỆT: Các hành động trên ĐÃ THỰC THI XONG. TUYỆT ĐỐI KHÔNG GỌI LẠI! Hãy thực hiện yêu cầu tiếp theo còn lại trong câu nói của người dùng.")
        prompt_parts.append("")

    prompt_parts.append(f"[YÊU CẦU CỦA NGƯỜI DÙNG]")
    prompt_parts.append(f"\"{user_utterance}\"")
    prompt_parts.append("")
    prompt_parts.append("Hãy suy luận bước tiếp theo và trả về DUY NHẤT một khối JSON hợp lệ:")

    return "\n".join(prompt_parts)
