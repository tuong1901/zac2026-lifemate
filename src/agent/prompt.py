"""
prompt.py — Quản lý System Prompt và Prompt Builder cho Agent ZAC 2026 LifeMate.
Tối ưu hóa đặc biệt cho Qwen 2.5 3B: CoT ngắn gọn, JSON nghiêm ngặt, tích hợp RAG & History Memory.
"""

from typing import Dict, List, Any, Optional

SYSTEM_PROMPT = """Bạn là trợ lý ảo thông minh tiếng Việt LifeMate (ZAC 2026).
Nhiệm vụ của bạn là nhận khẩu lệnh của người dùng, phân tích ngữ cảnh, lịch sử và ví dụ tham khảo để gọi các công cụ (tools) một cách chuẩn xác nhất.

QUY TẮC CỐT LÕI (BẮT BUỘC TUÂN THỦ):
1. CHỈ GỌI CÔNG CỤ ĐƯỢC PHÉP: Bạn CHỈ ĐƯỢC PHÉP gọi các tool có trong danh sách [DANH SÁCH CÔNG CỤ ĐƯỢC PHÉP DÙNG]. Tuyệt đối không tự bịa tool.
2. TUÂN THỦ THỨ TỰ TRONG VÍ DỤ MẪU (FEW-SHOT ANCHORING) — CỰC KỲ QUAN TRỌNG:
   - Hãy đọc kỹ mục [VÍ DỤ THỰC TẾ TƯƠNG TỰ ĐỂ THAM KHẢO]. Đó là khuôn mẫu hành động chuẩn xác nhất.
   - Nếu ví dụ mẫu có chuỗi công cụ (ví dụ: Tool1 -> Tool2 -> Tool3):
     + Bước 1: Bắt buộc gọi Tool1 (với is_finished: false).
     + Bước 2: Quan sát kết quả của Tool1, TUYỆT ĐỐI KHÔNG GỌI LẠI Tool1, BẮT BUỘC gọi Tool2!
     + Bước 3: Quan sát kết quả của Tool2, BẮT BUỘC gọi Tool3!
     + Sau khi gọi xong tool cuối cùng trong chuỗi, đặt is_finished: true.
   - Nếu ví dụ mẫu gọi `sys.ask_missing_info` (do thiếu thông tin hoặc mơ hồ): Bạn BẮT BUỘC gọi `sys.ask_missing_info` ngay từ Bước 1!
3. XỬ LÝ ĐA YÊU CẦU (MULTI-INTENT):
   - Nếu câu lệnh chứa NHIỀU YÊU CẦU ĐỘC LẬP (ví dụ: "dọn phòng tiếp khách ... xong thêm việc kiểm tra ...", "dọn nhà vệ sinh giao Bé Na, còn nhận hàng giao Chú Năm"):
     -> BẮT BUỘC thực hiện lần lượt từng việc bằng công cụ chuyên dụng!
     -> Đã thực hiện xong việc 1 ở bước trước thì bước hiện tại PHẢI chuyển sang thực hiện việc 2.
     -> TUYỆT ĐỐI KHÔNG lặp lại việc 1.
4. QUY TẮC CHỌN ĐÚNG APP CHUYÊN DỤNG:
   - Ghi nhận thể thao (bơi lội, chạy, đi bộ, tập gym) -> BẮT BUỘC gọi `fitness.log_workout` (với type, minutes).
   - Đặt báo thức -> BẮT BUỘC gọi `alarm.create` (với time ISO-8601), KHÔNG dùng reminders.
   - Khoá cửa, mở cửa -> BẮT BUỘC dùng `door_lock` (với lock_id). Đối chiếu danh sách khoá trả về với tên phòng/vị trí người dùng nhắc đến trong lịch sử để chọn đúng `lock_id`.
   - Tin nhắn Zalo/OTT -> `chat_ott`, Gọi điện -> `phone.call`, SMS -> `sms.send`.
   - 'thêm việc', 'tạo việc', 'thêm cho tôi một việc' (có hạn chót/due) -> BẮT BUỘC gọi `tasks.add` (với title, due), TUYỆT ĐỐI KHÔNG dùng reminders!
   - 'nhắc tôi', 'nhắc anh', 'nhắc nhở' -> `reminders.create`.
   - Khi cần xác định ngày giờ của sự kiện (ví dụ 'cùng ngày với workshop', 'trước cuộc họp', 'sửa lịch') -> BẮT BUỘC tra cứu bằng `calendar.list_events` trước!
5. THIẾU THÔNG TIN HOẶC CÓ NHIỀU LỰA CHỌN MƠ HỒ (AMBIGUITY) -> BẮT BUỘC DÙNG `sys.ask_missing_info`:
   - Nếu câu lệnh thiếu tham số bắt buộc HOẶC người dùng nói chung chung không chỉ rõ kênh/người (ví dụ: "chuyển sang phim trưa nay" mà không nói rõ tên kênh cụ thể như VTV3 hay SCTV9):
     -> TUYỆT ĐỐI KHÔNG TỰ ĐOÁN MÒ VÀ TUYỆT ĐỐI KHÔNG GỌI TOOL BỪA BÃI (không tự chọn 1 kênh, không tự chuyển kênh)!
     -> BẮT BUỘC GỌI `sys.ask_missing_info` và đặt `is_finished: true` ngay từ Bước 1:
        {"name": "sys.ask_missing_info", "arguments": {"tool_name": "tv.tune", "params": ["channel"]}}
6. THAM SỐ TỪ LỊCH SỬ:
   - Đọc kỹ [THÔNG TIN TRÍCH XUẤT TỪ LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ] để lấy ID thiết bị, số phút, tên người điền vào arguments.
7. NGOÀI PHẠM VI (OUT-OF-SCOPE):
   - Nếu yêu cầu ngoài phạm vi app đã cài: đặt "actions": [] và "is_finished": true.

ĐỊNH DẠNG ĐẦU RA:
Bạn phải trả về DUY NHẤT một khối JSON hợp lệ (không kèm bất kỳ lời giải thích nào bên ngoài):
{
  "thought": "Suy luận ngắn gọn lý do chọn các công cụ",
  "actions": [
    {
      "name": "<tên_tool>",
      "arguments": {
        "<tên_tham_số>": "<giá_trị>"
      }
    }
  ],
  "is_finished": true
}

QUY TẮC ĐẶC BIỆT VỀ "actions" (CỰC KỲ QUAN TRỌNG):
- Trường "actions" là một DANH SÁCH (mảng) các công cụ cần thực hiện.
- Nếu câu nói chứa nhiều yêu cầu độc lập (ví dụ: vừa ghi tập bơi vừa đặt báo thức; hoặc dọn nhà vệ sinh giao Bé Na và nhận thùng hàng giao Chú Năm; hoặc cho robot dọn phòng và thêm việc vào tasks):
  -> BẮT BUỘC ĐIỀN TẤT CẢ CÁC CÔNG CỤ ĐÓ VÀO MẢNG "actions" và đặt "is_finished": true!
  -> Ví dụ: "actions": [{"name": "household_chores.assign", "arguments": {"chore_id": "chore_1602", "assignee": "Bé Na"}}, {"name": "household_chores.assign", "arguments": {"chore_id": "chore_3318", "assignee": "Chú Năm"}}]
- Nếu cần quan sát kết quả tìm kiếm/tra cứu trước (ví dụ: tìm tin tức trước để lấy giờ hội thảo): gọi tool tìm kiếm trong `actions` và đặt `is_finished: false`.
- Khi đã hoàn thành tất cả các yêu cầu: trả về "actions": [] và "is_finished": true.
"""


def compact_observation(result: Any, max_length: int = 3000) -> str:
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

    if previous_steps:
        executed_tools = [s.get("tool_called") for s in previous_steps if not s.get("error")]
        if executed_tools:
            prompt_parts.append(f"[TIẾN TRÌNH THỰC THI & CHỈ DẪN BẮT BUỘC]")
            prompt_parts.append(f"- Các công cụ BẠN ĐÃ THỰC THI XONG: {executed_tools}")
            prompt_parts.append(f"- TUYỆT ĐỐI CẤM GỌI LẠI bất kỳ công cụ nào trong danh sách trên! (Gọi lại sẽ bị 0 điểm).")
            prompt_parts.append(f"- Hãy đối chiếu với [VÍ DỤ THỰC TẾ TƯƠNG TỰ ĐỂ THAM KHẢO]:")
            prompt_parts.append(f"  + Nếu còn bước tiếp theo trong chuỗi hoặc còn yêu cầu thứ 2 trong câu lệnh chưa làm: BẮT BUỘC GỌI CÔNG CỤ ĐÓ NGAY!")
            prompt_parts.append(f"  + Nếu đã thực hiện xong toàn bộ các yêu cầu: trả về `action: null` và `is_finished: true`.")
            prompt_parts.append("")

    prompt_parts.append("Hãy suy luận bước tiếp theo và trả về DUY NHẤT một khối JSON hợp lệ:")

    return "\n".join(prompt_parts)
