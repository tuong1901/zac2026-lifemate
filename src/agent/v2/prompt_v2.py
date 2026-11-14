"""
src/agent/v2/prompt_v2.py — Module Quản lý Prompt cho Kiến trúc Hai Giai Đoạn (V2).

Kiến trúc 2-Stage:
- Giai đoạn 1 (Stage 1 Prompt): Đọc toàn bộ audio transcript (lịch sử + câu hiện tại) cùng danh sách
  ứng dụng đã cài đặt (installed_apps chỉ gồm tên miền & mô tả ngắn). Trích xuất ý định, chọn lọc
  các ứng dụng cần thiết (ưu tiên Recall 100%, chấp nhận chọn dư nhưng không được thiếu) và cô đọng lịch sử.
- Giai đoạn 2 (Stage 2 Prompt): Nhận tập công cụ đã được thu hẹp (chỉ thuộc các ứng dụng được chọn ở Stage 1)
  cùng bản tóm tắt lịch sử cô đọng, thực hiện suy luận ReAct để gọi tool chính xác và siêu tốc.
"""

import json
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime

# ==============================================================================
# 1. TỪ ĐIỂN TỔNG QUAN 46 MIỀN ỨNG DỤNG (APP DOMAINS)
# ==============================================================================
APP_DOMAIN_CATALOG: Dict[str, str] = {
    "smart_home": "Điều khiển đèn, quạt, điều hoà, rèm, ổ cắm theo phòng hoặc tầng",
    "home_security": "Camera an ninh, cảm biến cửa, chuông báo động",
    "door_lock": "Khoá/mở cửa thông minh, cấp mã số khách tạm thời",
    "robot_vacuum": "Điều khiển robot hút bụi/lau nhà chạy hoặc dừng theo phòng/khu vực",
    "appliances": "Điều khiển thiết bị gia dụng: máy giặt, lò vi sóng, nồi cơm, máy lọc khí",
    "home_energy": "Tra cứu lượng điện tiêu thụ trong nhà, thiết lập ngưỡng cảnh báo",
    "alarm": "Đặt, sửa, tắt, bật, xoá báo thức hàng ngày",
    "reminders": "Tạo, xem, huỷ lời nhắc việc theo thời điểm hoặc vị trí",
    "calendar": "Xem, thêm, sửa, huỷ sự kiện lịch hẹn, lịch họp, kiểm tra lịch rảnh",
    "notes": "Tạo, đọc, tìm kiếm, sửa, xoá ghi chú văn bản",
    "tasks": "Quản lý danh sách việc cần làm (to-do list), đánh dấu hoàn thành, hạn chót",
    "contacts": "Tìm kiếm liên hệ trong danh bạ, số điện thoại, biệt danh, mối quan hệ",
    "phone": "Thực hiện cuộc gọi thoại trực tiếp qua số điện thoại hoặc liên hệ",
    "sms": "Gửi tin nhắn SMS tới số điện thoại hoặc danh bạ",
    "chat_ott": "Nhắn tin qua ứng dụng chat (Zalo, OTT), gửi tin nhắn nhóm, tìm tin nhắn",
    "email": "Đọc, tìm kiếm hòm thư, soạn thảo và gửi email",
    "video_meeting": "Tạo phòng họp trực tuyến, kiểm tra lịch họp video",
    "banking": "Tài khoản ngân hàng, kiểm tra số dư, sao kê giao dịch, chuyển khoản, sổ tiết kiệm",
    "ewallet": "Ví điện tử (ZaloPay, MoMo): nạp tiền, rút tiền, chuyển ví, nạp tiền điện thoại",
    "bill_payment": "Tra cứu và thanh toán hoá đơn sinh hoạt: tiền điện, tiền nước, internet, truyền hình",
    "loyalty": "Kiểm tra điểm tích luỹ thành viên, đổi voucher ưu đãi",
    "weather": "Tra cứu thông tin thời tiết hiện tại và dự báo các ngày tới theo địa điểm",
    "gold_price": "Tra cứu biến động giá vàng SJC, vàng nhẫn, giá vàng thế giới",
    "fuel_price": "Tra cứu giá xăng dầu hiện hành (RON 95, E5, Dầu DO)",
    "lottery": "Xem kết quả xổ số kiến thiết các đài, dò vé số theo ngày",
    "exchange_rate": "Tra cứu tỷ giá ngoại tệ (USD, EUR, JPY...) và quy đổi tiền tệ",
    "news": "Đọc và tìm kiếm tin tức thời sự, kinh tế, công nghệ theo chủ đề",
    "calculator": "Tính toán biểu thức số học, quy đổi đơn vị đo lường",
    "maps": "Tìm kiếm địa điểm, chỉ đường đi, tra cứu khoảng cách và thời gian di chuyển",
    "food_delivery": "Tìm kiếm quán ăn, xem menu món, đặt đồ ăn giao tận nơi",
    "ride_hailing": "Đặt xe công nghệ di chuyển (xe máy, ô tô 4 chỗ, 7 chỗ), kiểm tra cước phí",
    "flights": "Tìm kiếm vé máy bay, đặt chỗ, huỷ vé, làm thủ tục check-in trực tuyến",
    "shopping_list": "Quản lý danh sách đồ cần mua khi đi chợ/siêu thị",
    "package_tracking": "Tra cứu hành trình vận đơn hàng bưu kiện, tình trạng giao hàng",
    "music": "Phát nhạc, dừng nhạc, chuyển bài hát, phát danh sách bài hát",
    "tv": "Xem kênh truyền hình, lịch phát sóng chương trình TV",
    "podcast_radio": "Nghe kênh radio, chương trình podcast, sách nói",
    "medication": "Theo dõi đơn thuốc, nhắc nhở lịch uống thuốc hàng ngày",
    "fitness": "Theo dõi chỉ số tập luyện: bước chân, calo tiêu thụ, quãng đường",
    "learning": "Khóa học trực tuyến, từ vựng, bài tập học tập",
    "bike_rental": "Tìm trạm và thuê/trả xe đạp công cộng",
    "home_automation": "Thiết lập kịch bản tự động hoá nhà thông minh theo thời gian",
    "household_chores": "Phân công và theo dõi danh sách việc nhà giữa các thành viên",
    "library": "Tìm kiếm đầu sách trong thư viện, đặt mượn/trả sách",
    "subscriptions": "Quản lý và tra cứu các gói đăng ký định kỳ (xem phim, nghe nhạc...)",
    "sys": "Công cụ hệ thống (hỏi bổ sung thông tin khi yêu cầu còn mơ hồ)",
}

# ==============================================================================
# 2. BỘ LIÊN KẾT ỨNG DỤNG ĐỒNG HÀNH (COMPANION APPS FOR SAFE RECALL)
# ==============================================================================
# "Có thể lấy thừa nhưng phải đủ" — Nếu Stage 1 chọn 1 app, tự động bổ sung app đồng hành liên quan mật thiết
COMPANION_APPS_MAPPING: Dict[str, List[str]] = {
    # Thanh toán hoá đơn thường đi kèm tài khoản ngân hàng hoặc ví điện tử
    "bill_payment": ["ewallet", "banking", "bill_payment"],
    "banking": ["banking", "ewallet", "contacts"],
    "ewallet": ["ewallet", "banking", "contacts"],
    # Đặt xe thường liên quan đến bản đồ và danh bạ đón người khác
    "ride_hailing": ["ride_hailing", "maps", "contacts"],
    # Đặt món thường liên quan đến thanh toán
    "food_delivery": ["food_delivery", "ewallet", "banking"],
    # Các app liên lạc thường đi đôi với danh bạ
    "phone": ["phone", "contacts"],
    "sms": ["sms", "contacts"],
    "chat_ott": ["chat_ott", "contacts"],
    "email": ["email", "contacts"],
    # Lời nhắc và công việc, lịch trình
    "reminders": ["reminders", "tasks", "calendar"],
    "tasks": ["tasks", "reminders", "calendar"],
    "calendar": ["calendar", "reminders", "contacts"],
    # Nhà thông minh & gia dụng
    "smart_home": ["smart_home", "home_automation", "appliances"],
    "appliances": ["appliances", "smart_home"],
    "home_security": ["home_security", "door_lock"],
    "door_lock": ["door_lock", "home_security"],
    # Đi chợ & ghi chú
    "shopping_list": ["shopping_list", "notes", "food_delivery"],
    # Chuyến bay & lịch trình
    "flights": ["flights", "calendar"],
}


def expand_selected_apps_with_safety_net(
    selected_apps: List[str],
    installed_apps: Dict[str, Any]
) -> List[str]:
    """
    Mở rộng danh sách ứng dụng đã chọn bằng mạng lưới ứng dụng đồng hành (Companion Network),
    đồng thời lọc lại chỉ giữ các ứng dụng thực sự đã được cài đặt (installed_apps),
    luôn luôn giữ 'sys' cho mọi tình huống.
    """
    installed_set = set(installed_apps.keys()) if isinstance(installed_apps, dict) else set(installed_apps)
    expanded: Set[str] = set()

    for app in selected_apps:
        app_clean = app.strip().lower()
        if not app_clean:
            continue
        expanded.add(app_clean)
        if app_clean in COMPANION_APPS_MAPPING:
            for companion in COMPANION_APPS_MAPPING[app_clean]:
                expanded.add(companion)

    # sys luôn được giữ lại
    final_apps: Set[str] = {"sys"}
    for app in expanded:
        if app == "sys" or app in installed_set:
            final_apps.add(app)

    return sorted(list(final_apps))


# ==============================================================================
# 3. GIAI ĐOẠN 1: ROUTER & HISTORY CONDENSER PROMPT (STAGE 1)
# ==============================================================================
STAGE1_SYSTEM_PROMPT = """Bạn là Bộ Định Tuyến Ý Định & Rút Gọn Ngữ Cảnh (Intent Router & Context Condenser) cho Trợ lý Ảo Thông Minh ZAC 2026.
Nhiệm vụ của bạn là đọc toàn bộ lịch sử trò chuyện cùng câu nói hiện tại của người dùng, kết hợp với danh sách ứng dụng đang cài đặt (installed_apps), sau đó:
1. Phân loại xem yêu cầu cần gọi tới những ứng dụng nào (selected_apps).
   - NGUYÊN TẮC AN TOÀN: Có thể lấy thừa nhưng tuyệt đối KHÔNG ĐƯỢC THIẾU ứng dụng cần thiết để xử lý trọn vẹn yêu cầu.
   - CÂU LỆNH CÓ ĐIỀU KIỆN (ví dụ: 'nếu mai mưa thì dời chạy bộ và nhắn tin, không mưa thì thôi'): BẮT BUỘC chọn TẤT CẢ các app liên quan: `weather` (kiểm tra thời tiết), `calendar` (lịch trình), `sms` hoặc `chat_ott` (nhắn tin/liên lạc).
   - Nếu câu nói chứa 'gọi điện', 'nhắn tin', 'chuyển tiền cho [Tên người]': luôn chọn thêm `contacts`.
   - Nếu người dùng chỉ chào hỏi, cảm ơn, khen ngợi, hoặc nói huỷ lệnh ("thôi không cần nữa", "huỷ đi"), trả về danh sách rỗng: "selected_apps": [].
2. Trích xuất bản tóm tắt lịch sử cô đọng (condensed_history):
   - Ghi lại các mã định danh (ID thiết bị, ID hoá đơn, ID vé, số tiền, tên người, trạng thái, thời gian) đã xuất hiện ở các lượt trước mà lượt hiện tại cần dùng tới.
   - Viết ngắn gọn, súc tích (1-3 câu). Nếu không có lịch sử trước đó hoặc không có thông tin liên quan, ghi "Không có".

Định dạng đầu ra BẮT BUỘC là JSON duy nhất:
```json
{
  "thought": "Giải thích ngắn gọn ý định của người dùng và các thông tin cần lấy từ lịch sử",
  "selected_apps": ["tên_app_1", "tên_app_2"],
  "condensed_history": "Tóm tắt ngắn gọn các thực thể, ID, số tiền, trạng thái từ lịch sử"
}
```
"""


def build_stage1_router_prompt(
    user_text: str,
    history: List[Dict[str, Any]],
    installed_apps: Dict[str, Any],
    current_time_str: str = ""
) -> str:
    """
    Xây dựng prompt gọn nhẹ cho Giai đoạn 1.
    Chỉ truyền tên miền ứng dụng và mô tả 1 dòng, không truyền bất kỳ JSON schema nào.
    Prompt này thường chỉ ~250 - 400 tokens, LLM sinh ~50 tokens -> Siêu nhanh (< 0.8s).
    """
    installed_keys = list(installed_apps.keys()) if isinstance(installed_apps, dict) else list(installed_apps)
    
    # Định dạng danh sách ứng dụng cài sẵn
    apps_desc_lines = []
    for app_id in sorted(installed_keys):
        desc = APP_DOMAIN_CATALOG.get(app_id, "Ứng dụng hỗ trợ tiện ích người dùng")
        apps_desc_lines.append(f"- {app_id}: {desc}")
    installed_apps_str = "\n".join(apps_desc_lines) if apps_desc_lines else "(Không có app nào ngoài sys)"

    # Định dạng lịch sử hội thoại (lấy tối đa 5 lượt gần nhất để giữ context sạch)
    history_lines = []
    if history:
        for idx, turn in enumerate(history[-5:], 1):
            u_txt = turn.get("user", "") or turn.get("user_utterance", "") or turn.get("transcript", "")
            t_calls = turn.get("tool_calls", [])
            calls_str = ", ".join([f"{c.get('name')}({json.dumps(c.get('arguments', {}), ensure_ascii=False)})" for c in t_calls if isinstance(c, dict)])
            obs_str = ""
            if "observation" in turn:
                obs_val = str(turn["observation"])
                if len(obs_val) > 200:
                    obs_val = obs_val[:200] + "..."
                obs_str = f" | Kết quả: {obs_val}"
            
            line_txt = f"Lượt #{idx} - Người dùng: \"{u_txt}\""
            if calls_str:
                line_txt += f" | Công cụ đã gọi: [{calls_str}]"
            if obs_str:
                line_txt += obs_str
            history_lines.append(line_txt)

    history_str = "\n".join(history_lines) if history_lines else "Không có lịch sử trước đó."

    time_ctx = f"Thời điểm hiện tại: {current_time_str}\n" if current_time_str else ""

    return f"""{time_ctx}DANH SÁCH ỨNG DỤNG ĐÃ CÀI ĐẶT TRÊN THIẾT BỊ:
{installed_apps_str}

LỊCH SỬ TRÒ CHUYỆN TRƯỚC ĐÓ:
{history_str}

CÂU NÓI HIỆN TẠI CỦA NGƯỜI DÙNG:
"{user_text}"

Hãy phân tích và trả về kết quả JSON theo đúng định dạng yêu cầu."""


# ==============================================================================
# 4. GIAI ĐOẠN 2: DETAILED REACT PLANNER PROMPT (STAGE 2)
# ==============================================================================
STAGE2_SYSTEM_PROMPT = """Bạn là Trợ lý Ảo ZAC 2026 LifeMate điều khiển thiết bị thông minh thông qua việc gọi các công cụ (Tool Calls).
Bạn tuân thủ nghiêm ngặt quy trình ReAct (Thought -> Action).

NGUYÊN TẮC THỰC THI:
1. Đọc kỹ "Tóm tắt ngữ cảnh & lịch sử" và câu nói của người dùng để nắm rõ mục tiêu.
2. Tra cứu trước khi hành động: Nếu cần ID, số dư, danh bạ, trạng thái mà chưa biết rõ, hãy gọi công cụ đọc (read) trước.
3. Không bịa đặt tham số (No Hallucination): Chỉ dùng các ID, giá trị thực tế lấy từ dữ liệu hệ thống hoặc lịch sử. Nếu người dùng không cung cấp đủ thông tin cốt lõi bắt buộc, hãy gọi `sys.ask_missing_info`.
4. KHI ĐÃ HOÀN THÀNH HÀNH ĐỘNG CHÍNH (ví dụ: đã đặt xe thành công, đã tạo/sửa báo thức thành công, đã gửi tin nhắn thành công, đã chuyển tiền xong): BẠN ĐÃ ĐẠT ĐƯỢC MỤC TIÊU! Trả về `\"action\": null` để kết thúc ngay. TUYỆT ĐỐI KHÔNG GỌI `sys.ask_missing_info` hay gọi lại công cụ sau khi hành động đã thành công.
5. KHÔNG LẶP LẠI CÔNG CỤ ĐỌC: Nếu bạn đã gọi công cụ tra cứu (`get_balance`, `search`, `list`) ở bước trước và đã có kết quả trong lịch sử, hãy dùng kết quả đó để hành động, TUYỆT ĐỐI KHÔNG gọi lại chính công cụ đó lần thứ 2.
6. THAM SỐ CHUẨN XÁC: Tên tham số trong `arguments` chỉ là tên thuần túy (ví dụ: `\"tool_name\": \"alarm.create\", \"params\": [\"label\"]`), KHÔNG ĐƯỢC chứa dấu sao `*` hay kiểu dữ liệu như `: string`.
7. Nếu yêu cầu đã hoàn thành trọn vẹn hoặc không cần gọi thêm công cụ nào, hãy trả về `action: null` hoặc không gọi thêm.

Định dạng phản hồi BẮT BUỘC là JSON duy nhất:
```json
{
"thought": "Suy luận logic từng bước ngắn gọn",
"action": {
    "name": "tên_công_cụ",
    "arguments": { "tham_số_1": "giá_trị_1" }
}
}
```
Nếu không cần gọi công cụ nào (đã hoàn thành hoặc không hợp lệ):
```json
{
"thought": "Đã hoàn thành toàn bộ yêu cầu của người dùng / Không cần gọi thêm công cụ",
"action": null
}
```
"""


def build_stage2_planner_prompt(
    user_text: str,
    condensed_history: str,
    allowed_tools_desc: str,
    current_time_str: str = "",
    observations_history: str = "",
    step_directive: str = ""
) -> str:
    """
    Xây dựng prompt cho Giai đoạn 2.
    Context ở đây cực kỳ tinh gọn vì:
    - allowed_tools_desc chỉ gồm các công cụ thuộc selected_apps (giảm từ ~140 tools xuống còn ~4-10 tools).
    - condensed_history chỉ giữ thông tin cốt lõi từ Stage 1.
    """
    time_ctx = f"Thời điểm hiện tại: {current_time_str}\n" if current_time_str else ""
    obs_ctx = f"\nKẾT QUẢ CÁC BƯỚC ĐÃ THỰC HIỆN TRƯỚC ĐÓ TRONG LƯỢT NÀY:\n{observations_history}\n" if observations_history else ""
    dir_ctx = f"\nCHỈ DẪN BƯỚC HIỆN TẠI: {step_directive}\n" if step_directive else ""

    return f"""{time_ctx}CÔNG CỤ KHẢ DỤNG CHO YÊU CẦU NÀY:
{allowed_tools_desc}

TÓM TẮT NGỮ CẢNH & LỊCH SỬ TỪ GIAI ĐOẠN 1:
{condensed_history}

YÊU CẦU HIỆN TẠI CỦA NGƯỜI DÙNG:
"{user_text}"
{obs_ctx}{dir_ctx}
Hãy đưa ra suy luận và hành động tiếp theo bằng định dạng JSON:"""

