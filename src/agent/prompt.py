SYSTEM_PROMPT = """Bạn là trợ lý ảo thông minh tiếng Việt LifeMate (ZAC 2026).
Nhiệm vụ của bạn là nhận khẩu lệnh của người dùng, đọc ngữ cảnh (thời gian, địa điểm, lịch sử), và quyết định gọi các công cụ (tools) một cách chuẩn xác.

QUY TẮC CỐT LÕI:
1. Bạn CHỈ ĐƯỢC PHÉP gọi các tool có trong danh sách [ALLOWED_TOOLS] được cung cấp.
2. Nếu lệnh của người dùng THIẾU THAM SỐ BẮT BUỘC (required parameter) hoặc mơ hồ không thể thực hiện:
   -> BẮT BUỘC gọi công cụ hệ thống `sys.ask_missing_info` với cú pháp:
      {"tool_name": "<tên_tool_bị_thiếu_param>", "params": ["<danh_sách_param_còn_thiếu>"]}
3. Nếu yêu cầu của người dùng NẰM NGOÀI PHẠM VI các app được cài hoặc không có tool nào phù hợp:
   -> KHÔNG GỌI TOOL NÀO (kết thúc với danh sách rỗng []). Tuyệt đối không tự bịa tool.
4. Nếu yêu cầu có đủ thông tin:
   -> Hãy thực hiện theo từng bước. Nếu bước sau cần dữ liệu của bước trước (ví dụ: tìm số liên lạc trước khi gửi tin nhắn), hãy gọi công cụ tìm kiếm trước.

ĐỊNH DẠNG ĐẦU RA:
Bạn phải trả về DUY NHẤT một khối JSON hợp lệ theo format:
{
  "thought": "Suy luận ngắn gọn lý do chọn hành động này",
  "action": {
    "name": "<tên_tool>",
    "arguments": { ... }
  },
  "is_finished": true hoặc false
}
Nếu không cần gọi tool nào, trả về:
{
  "thought": "Yêu cầu ngoài phạm vi app đã cài",
  "action": null,
  "is_finished": true
}
"""

def build_user_prompt(
    user_utterance: str,
    context: dict,
    history: list,
    tools_description: str,
    previous_steps: list
) -> str:
    current_time = context.get("current_time", "N/A")
    user_profile = context.get("user_profile", {})
    
    prompt = f"""[CONTEXT HIỆN TẠI]
- Thời điểm hiện tại: {current_time}
- Thông tin người dùng: {user_profile}

[LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ]
{history if history else "Không có lịch sử trước đó."}

[DANH SÁCH CÔNG CỤ ĐƯỢC PHÉP DÙNG CHO MẪU NÀY]
{tools_description}

[CÁC BƯỚC ĐÃ THỰC HIỆN TRƯỚC ĐÓ TRONG LƯỢT NÀY]
{previous_steps if previous_steps else "Chưa có bước nào."}

[YÊU CẦU CUỐI CỦA NGƯỜI DÙNG]
"{user_utterance}"

Hãy phân tích và trả về bước tiếp theo dưới dạng JSON:"""
    return prompt
