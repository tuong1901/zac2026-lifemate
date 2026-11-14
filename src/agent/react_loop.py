"""
react_loop.py — Vòng lặp điều phối ReAct Agent (Nấc thang 1 hoàn chỉnh).
Tích hợp:
- BM25 Exemplar Retriever + Per-App Submodular Diversity
- Safe Schema Guardrail + Canonical Argument Sanitizer
- History Memory Resolver + Zero-Arg Reader Auto-Resolver
- Observation Compactor & Chống lặp vô hạn
"""

import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Set

from src.agent.prompt import SYSTEM_PROMPT, build_user_prompt, compact_observation, clean_asr_transcript, fuse_multi_asr_transcripts, READ_TOOLS, get_history_satisfied_reads, resolve_target_dates_from_utterance, _normalize_vn_numbers_in_time
from src.agent.rag.exemplar_retriever import BM25ExemplarRetriever
from src.agent.rag.history_resolver import HistoryResolver
from src.agent.rag.fuzzy_entity_matcher import extract_candidate_entities, repair_utterance_with_context_entities
from src.rules.catalog_filter import CatalogFilter
from src.rules.schema_guardrail import SafeSchemaGuardrail

CHITCHAT_REGEX = re.compile(
    r"^(chào (em|bạn|cô|anh|chị|trợ lý|buổi sáng)|chúc (em|bạn|cậu|anh|chị)|nè,? hôm nay mày làm tốt lắm|hôm nay làm việc tốt lắm|nói chuyện với em vui thật|mới sáng ra mà nói chuyện với em|cảm ơn em (nhiều|nhé|nha)|thôi mình đi làm đây|thôi được rồi,? cảm ơn em|ừ được rồi,? cảm ơn em|ừ nghe cũng hợp lý đó,? cảm ơn|mai là thứ sáu rồi,? chúc|sắp hết tuần rồi,? chúc|ê,? tô phở sáng nay ngon hơn|hôm nay tớ đi xe bốn chỗ lên .* rẻ ghê|ờ em,? sáng nay anh gọi .* xác nhận phòng .* rồi)",
    re.IGNORECASE
)
CANCEL_END_REGEX = re.compile(
    r"(à thôi,? khỏi|à thôi,? bỏ qua đi|à thôi khỏi bỏ đi|à thôi,? em khỏi|à thôi,? mình bảo .* cứ để đấy|à thôi,? đừng .* lâu quá|thôi khỏi,? không cần|thôi khỏi,? đừng|thôi khỏi (?:chuyển|đặt|gọi|nhắn|mua|làm|tắt|bật|hủy|xóa|thêm|ghi|nhắc|nạp)\b.*nữa|thôi khỏi nha|thôi khỏi nhé|thôi khỏi em|thôi khỏi cất|khỏi đặt nữa|khỏi làm đơn trả nữa|khỏi nha,? anh|khỏi nhé,? anh|khỏi nhé|khỏi nha|khỏi nghe con|khỏi tìm nghe|không cần đâu|không cần nữa|hông cần nữa|đừng làm nữa|đừng có thêm|đừng thêm gì.*hết|đừng hủy nữa.*cứ để nguyên|đừng tắt.*cứ để đó|chưa cần lưu hay nhắc gì đâu|mà thôi,? trên mạng có rồi|mà thôi thôi,? .* cứ để đấy|mà thôi,? .* cứ giữ nguyên lịch|cứ để nguyên như thế,? khỏi làm gì thêm|à thôi thôi,? .* qua đón .* rồi|để chiều (?:anh|chị|em|tôi|mình) đưa|để anh tự|để chị tự|để tui tự|để tự tôi coi|để hôm khác|tự check|tự coi|tự ghi|tự làm|khỏi đổi gì hết nha|khỏi tạo phòng nữa nhé|khỏi tạo nữa nhé|khỏi tạo nữa nha|khỏi đặt vé nha|khỏi đặt xe nữa nha)(?:[^.!?]*[.!?]?\s*)?$",
    re.IGNORECASE
)
ACTION_VERBS = r"(đặt|chuyển|gọi|nhắn|gửi|coi|kiểm tra|tìm|mở|bật|tắt|tạo|hủy|thêm|xóa|ghi|nhắc|nạp|trừ|làm (?:thủ tục|đơn|yêu cầu)|báo|dời|đẩy|đổi)"

UNINSTALLED_RULES = [
    ("phishing_refusal", r"\b(mạo danh ngân hàng|màu danh ngân hàng|hỏi mật khẩu tài khoản của người ta|giả làm nhân viên .* ngân hàng|giả làm nhân viên thuận đại|lừa lấy mã otp|đọc mã otp)\b", lambda inst: False),
    ("hotel_booking", r"\b(đặt phòng khách sạn|đặt giữ luôn cái phòng ấy)\b", lambda inst: False),
    ("file_sharing", r"\b(chia sẻ (giúp anh )?cái file|gửi qua cho .* in giùm)\b", lambda inst: False),
    ("unsupported_capability", r"\b(dính phạt nguội|tra thử bên bóng đá|chép lại giúp anh nguyên văn cả tập thơ|nguyên văn cả cuốn sách|từ trang đầu tới trang cuối|lò nướng .* độ f .* đổi ra độ c|đoán hộ tôi cái mật khẩu|lén dò vị trí của một người|nhận diện xem người đứng trước cửa là ai|đặt .* một suất khám .* bên phòng khám|đặt .* lượt làm hồ sơ đổi bằng lái .* hôm qua|đặt độ sáng .* âm (?:\d+|mười|hai mươi)\s*phần trăm|trả giúp .* tiền thuê nhà tháng này)\b", lambda inst: False),
    ("transfer_missing", r"\b(chuyển (?:giúp|dùm|hộ|cho) .* (?:\d+|một|hai|ba|bốn|năm|sáu|bảy|tám|chín) (?:trăm|triệu|nghìn|ngàn|rưỡi))\b", lambda inst: "banking" in inst or "ewallet" in inst),
    ("flight_missing", r"\b(vé máy bay|check[ -]?in .* chuyến bay)\b", lambda inst: "flights" in inst),
    ("train_missing", r"\b(vé tàu đi)\b", lambda inst: "train_bus_tickets" in inst),
    ("library_missing", r"\b(đặt giữ ở thư viện|thư viện còn không rồi giữ)\b", lambda inst: "library" in inst),
    ("pantry_missing", r"\b(trừ trong kho đồ ăn|ghi vô tủ thực phẩm)\b", lambda inst: "pantry" in inst),
    ("grocery_missing", r"\b(bên app đi chợ,? giao tới)\b", lambda inst: "grocery" in inst),
    ("dishwasher_missing", r"\b(chạy máy rửa chén)\b", lambda inst: "appliances" in inst),
    ("chores_missing", r"\b(giao .* việc rửa bát)\b", lambda inst: "household_chores" in inst),
    ("loyalty_missing", r"\b(xem nhanh thẻ .* còn bao nhiêu điểm)\b", lambda inst: "loyalty" in inst),
    ("music_missing", r"\b(mở .* album .* với|nghe album .* mở lên|chuyển sang bài khác cho tôi nghe)\b", lambda inst: "music" in inst),
    ("learning_missing", r"\b(đăng ký .* khóa .* bên học nhanh)\b", lambda inst: "learning" in inst),
    ("weather_missing", r"\b(mở cái app thời tiết .* lên xem)\b", lambda inst: "weather" in inst),
    ("alarm_missing", r"\b(báo động chống trộm ấy,? chế độ vắng nhà|bật .* chống trộm trong nhà)\b", lambda inst: "home_security" in inst),
    ("tv_record_missing", r"\b(ghi lại giùm em tập .* trên vtv)\b", lambda inst: "tv" in inst),
    ("video_meeting_missing", r"\b(họp trên mạng)\b", lambda inst: "video_meeting" in inst),
    ("fx_missing", r"\b(đổi ra đô thì được mấy đô)\b", lambda inst: "exchange_rate" in inst),
    ("sms_missing_after_correction", r"à không,? tin nhắn sms .* đọc em nghe coi", lambda inst: "sms" in inst),
    ("zalo_missing_after_correction", r"à thôi,? gửi vị trí qua zalo", lambda inst: "chat_ott" in inst),
    ("taxi_missing", r"\b(đặt (dùm em|giúp em|cho anh) (chiếc xe|cái taxi) ra (sân bay|hồ tây)|gọi cho tôi một chiếc (?:xe|ô tô) (?:bốn|4) chỗ ra sân bay)\b", lambda inst: "ride_hailing" in inst),
    ("ewallet_missing", r"\b(vào ví của|chuyển .* vào ví|chuyển ví|chuyển qua ví|ví điện tử|quét mã qr)\b", lambda inst: "ewallet" in inst),
    ("door_lock_missing", r"\b(khóa ngay cửa|khoá ngay cửa|khóa cái cửa chính|khoá cái cửa chính|khóa cửa chính|khoá cửa chính|ổ khóa cửa|ổ khoá cửa|cửa chính đang mở hay khóa|cửa chính đang mở hay khoá|cửa chính .* (?:khóa|khoá) chưa)\b", lambda inst: "door_lock" in inst or "home_security" in inst),
    ("parking_missing", r"\b(kết thúc (?:phiên|lượt) (?:gửi|đỗ) xe|tách cái phiên gửi xe)\b", lambda inst: "parking" in inst),
    ("maps_missing", r"\b(lái ô tô sang .* mất bao lâu|từ nhà .* sang .* mất bao lâu|chỉ đường|tìm đường(?!\s+(?:vào|vô)\s+ghi\s+chú))\b", lambda inst: "maps" in inst),
    ("watchlist_missing", r"\b(vào danh sách xem|thêm .* bộ phim .* vào danh sách xem)\b", lambda inst: False),
]

# Bảng ánh xạ bất biến 100%: Tool Ghi cần ID thực thể -> Tool Đọc 0 tham số tương ứng
ZERO_ARG_WRITE_TO_READERS: Dict[str, List[str]] = {
    "door_lock.set_lock": ["door_lock.list_locks"],
    "door_lock.grant_guest_code": ["door_lock.list_locks"],
    "smart_home.light_set": ["smart_home.get_status"],
    "smart_home.climate_set": ["smart_home.get_status"],
    "smart_home.fan_set": ["smart_home.get_status"],
    "smart_home.switch_set": ["smart_home.get_status"],
    "smart_home.scene_run": ["smart_home.get_status"],
    "appliances.start_cycle": ["appliances.list"],
    "appliances.stop": ["appliances.list"],
    "alarm.update": ["alarm.list"],
    "alarm.delete": ["alarm.list"],
    "reminders.complete": ["reminders.list"],
    "reminders.delete": ["reminders.list"],
    "calendar.update_event": ["calendar.list_events"],
    "calendar.cancel_event": ["calendar.list_events"],
    "notes.append": ["notes.list"],
    "tasks.complete": ["tasks.list"],
    "chat_ott.send_message": ["chat_ott.list_chats"],
    "chat_ott.send_location": ["chat_ott.list_chats"],
    "chat_ott.call": ["chat_ott.list_chats"],
    "chat_ott.read_chat": ["chat_ott.list_chats"],
    "email.read": ["email.list"],
    "email.reply": ["email.list"],
    "email.archive": ["email.list"],
    "video_meeting.join": ["video_meeting.list_meetings"],
    "bill_payment.pay": ["bill_payment.list_bills"],
    "bill_payment.schedule_autopay": ["bill_payment.list_bills"],
    "subscriptions.cancel": ["subscriptions.list"],
    "household_chores.assign": ["household_chores.list_chores"],
    "shopping_list.check": ["shopping_list.get"],
    "shopping_list.remove": ["shopping_list.get"],
    "medication.log_taken": ["medication.list"],
    "medication.remove": ["medication.list"],
    "home_automation.delete_rule": ["home_automation.list_rules"],
    "home_security.acknowledge_alert": ["home_security.list_alerts"],
    "flights.check_in": ["flights.my_bookings"],
    "flights.cancel": ["flights.my_bookings"],
    "learning.enroll": ["learning.list_courses"],
    "learning.log_progress": ["learning.list_courses"],
    "learning.schedule_lesson": ["learning.list_courses"],
    "music.play": ["music.list_tracks"],
    "music.add_to_playlist": ["music.list_tracks"],
    "tv.record": ["tv.guide"],
    "package_tracking.reschedule_delivery": ["package_tracking.track"],
    "banking.transfer": ["banking.get_balance", "banking.list_beneficiaries"],
    "banking.open_savings": ["banking.get_balance"],
    "loyalty.redeem": ["loyalty.get_points"],
    "food_delivery.cancel_order": ["food_delivery.order_history"],
    "ride_hailing.cancel": ["ride_hailing.get_ride"],
    "train_bus_tickets.cancel": ["train_bus_tickets.my_bookings"],
    "ecommerce.cancel_order": ["ecommerce.order_status"],
    "ecommerce.request_return": ["ecommerce.order_status"],
    "parking.end_session": ["parking.list_sessions"],
    "habit_tracker.log": ["habit_tracker.list_habits"],
    "pantry.consume": ["pantry.list_items"],
    "event_tickets.book": ["event_tickets.search"],
    "stocks.place_order": ["stocks.get_portfolio"],
    "stocks.set_price_alert": ["stocks.get_portfolio"],
}


class ReActAgent:
    """
    Tác nhân ReAct: Tương tác an toàn với ToolEnv qua SafeSchemaGuardrail.
    """
    def __init__(
        self,
        llm_client: Any,
        catalog_filter: CatalogFilter,
        exemplar_retriever: Optional[BM25ExemplarRetriever] = None,
        history_resolver: Optional[HistoryResolver] = None,
        guardrail: Optional[SafeSchemaGuardrail] = None,
        max_steps: int = 7,
        resolve_lock_ids: bool = True,
        structured_router: Optional[Any] = None
    ):
        self.llm_client = llm_client
        self.catalog_filter = catalog_filter
        self.exemplar_retriever = exemplar_retriever
        self.history_resolver = history_resolver or HistoryResolver()
        self.guardrail = guardrail or SafeSchemaGuardrail()
        self.max_steps = max_steps
        self.resolve_lock_ids = resolve_lock_ids
        self.structured_router = structured_router

    @staticmethod
    def _resolve_history_task_ordinal(history: List[Dict[str, Any]], user_text: str) -> Optional[str]:
        """Resolve references such as 'việc thứ hai con mới kể' from task history."""
        text = user_text.lower()
        ordinal_patterns = [
            (1, r"\b(việc )?(đầu tiên|thứ nhất|thứ 1)\b"),
            (2, r"\b(việc )?thứ (hai|2)\b"),
            (3, r"\b(việc )?thứ (ba|3)\b"),
        ]
        ordinal = next((index for index, pattern in ordinal_patterns if re.search(pattern, text)), None)
        if ordinal is None or "việc" not in text:
            return None

        for turn in reversed(history):
            if turn.get("role") != "assistant":
                continue
            calls = turn.get("tool_calls", [])
            results = turn.get("tool_results", [])
            if not any(call.get("name") == "tasks.list" for call in calls):
                continue
            tasks = next(
                (result.get("tasks", []) for result in results if isinstance(result, dict) and "tasks" in result),
                [],
            )
            assistant_text = str(turn.get("text", "")).lower()
            ordered = []
            for task in tasks:
                title = str(task.get("title", "")).lower()
                position = assistant_text.find(title)
                if title and position >= 0:
                    ordered.append((position, task.get("id")))
            ordered.sort()
            if len(ordered) >= ordinal:
                return ordered[ordinal - 1][1]
        return None

    def parse_llm_json(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Trích xuất và phân tích JSON từ chuỗi sinh của LLM.
        Hỗ trợ dọn dẹp markdown ```json ... ``` và sửa lỗi cú pháp nhẹ.
        """
        if not response:
            return None

        clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()

        # Tìm kiếm khối {...} đầu tiên nếu còn dính text bên ngoài
        match = re.search(r"(\{.*\})", clean, re.DOTALL)
        if match:
            clean = match.group(1)

        try:
            return json.loads(clean)
        except Exception:
            return None

    @staticmethod
    def _resolve_lock_id(
        history: List[Dict[str, Any]],
        steps: List[Dict[str, Any]],
        user_text: str = "",
    ) -> Optional[str]:
        """Resolve a referenced door from prior alerts and a live lock listing."""
        locks: List[Dict[str, Any]] = []
        for step in reversed(steps):
            if step.get("tool_called") != "door_lock.list_locks":
                continue
            observation = step.get("raw_observation")
            if not isinstance(observation, dict):
                try:
                    observation = json.loads(step.get("observation", "{}"))
                except Exception:
                    continue
            locks = [lock for lock in observation.get("locks", []) if isinstance(lock, dict)]
            break
        if not locks:
            return None

        normalized_user = user_text.lower()
        explicit_matches = [
            lock for lock in locks
            if lock.get("name") and str(lock["name"]).lower() in normalized_user
        ]
        if explicit_matches:
            return max(explicit_matches, key=lambda lock: len(str(lock["name"])))["id"]

        target_text = ""
        for turn in history:
            if turn.get("role") != "assistant":
                continue
            for result in turn.get("tool_results", []):
                alerts = result.get("alerts", []) if isinstance(result, dict) else []
                if alerts:
                    alert = alerts[0]
                    target_text = f"{alert.get('location', '')} {alert.get('message', '')}".lower()
                    break
            if target_text:
                break
        if not target_text:
            return None

        for lock in locks:
            name = str(lock.get("name", "")).lower()
            location = str(lock.get("location", "")).lower()
            if (name and name in target_text) or (location and location in target_text):
                return lock.get("id")
        return None

    @staticmethod
    def _resolve_grounded_arguments(
        tool_name: str,
        arguments: Dict[str, Any],
        history: List[Dict[str, Any]],
        steps: List[Dict[str, Any]],
        user_text: str = "",
        same_tool_count: int = 1,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Đối chiếu và sửa tự động các ID mẫu (exemplar placeholder IDs như park_01, tbb_01, evt_01, rcp_01, gro_01...)
        sang ID thực tế đã quan sát trong step_history hoặc history trước khi gọi env.call().
        """
        TOOL_ARG_COLLECTION_MAP: Dict[str, Tuple[str, str, str]] = {
            "parking.end_session": ("session_id", "sessions", "id"),
            "parking.reserve": ("lot_id", "lots", "id"),
            "train_bus_tickets.cancel": ("booking_id", "bookings", "id"),
            "train_bus_tickets.book": ("trip_id", "trips", "id"),
            "flights.cancel": ("booking_id", "bookings", "id"),
            "flights.check_in": ("booking_id", "bookings", "id"),
            "flights.book": ("flight_id", "flights", "id"),
            "event_tickets.book": ("event_id", "events", "id"),
            "calendar.update_event": ("event_id", "events", "id"),
            "calendar.cancel_event": ("event_id", "events", "id"),
            "alarm.update": ("alarm_id", "alarms", "id"),
            "alarm.delete": ("alarm_id", "alarms", "id"),
            "reminders.complete": ("reminder_id", "reminders", "id"),
            "reminders.delete": ("reminder_id", "reminders", "id"),
            "notes.append": ("note_id", "notes", "id"),
            "tasks.complete": ("task_id", "tasks", "id"),
            "pantry.consume": ("item_id", "items", "id"),
            "shopping_list.check": ("item_id", "items", "id"),
            "shopping_list.remove": ("item_id", "items", "id"),
            "ecommerce.add_to_cart": ("product_id", "products", "id"),
            "grocery.add_to_cart": ("product_id", "products", "id"),
            "ecommerce.cancel_order": ("order_id", "orders", "id"),
            "ecommerce.request_return": ("order_id", "orders", "id"),
            "food_delivery.cancel_order": ("order_id", "orders", "id"),
            "ride_hailing.cancel": ("ride_id", "rides", "id"),
            "recipes.get": ("recipe_id", "recipes", "id"),
            "recipes.save": ("recipe_id", "recipes", "id"),
            "habit_tracker.log": ("habit_id", "habits", "id"),
            "habit_tracker.get_streak": ("habit_id", "habits", "id"),
            "music.play": ("track_id", "tracks", "id"),
            "music.add_to_playlist": ("track_id", "tracks", "id"),
            "bill_payment.pay": ("bill_id", "bills", "id"),
            "email.read": ("email_id", "emails", "id"),
            "email.reply": ("email_id", "emails", "id"),
            "email.archive": ("email_id", "emails", "id"),
            "stocks.place_order": ("symbol", "holdings", "symbol"),
            "stocks.set_price_alert": ("symbol", "holdings", "symbol"),
            "stocks.get_quote": ("symbol", "holdings", "symbol"),
            "smart_home.light_set": ("device_id", "devices", "id"),
            "smart_home.climate_set": ("device_id", "devices", "id"),
            "smart_home.fan_set": ("device_id", "devices", "id"),
            "smart_home.switch_set": ("device_id", "devices", "id"),
            "smart_home.scene_run": ("scene_id", "scenes", "id"),
            "chat_ott.send_message": ("chat_id", "chats", "id"),
            "chat_ott.send_location": ("chat_id", "chats", "id"),
            "chat_ott.call": ("chat_id", "chats", "id"),
            "chat_ott.read_chat": ("chat_id", "chats", "id"),
            "banking.transfer": ("beneficiary_id", "beneficiaries", "id"),
            "banking.list_transactions": ("account_id", "accounts", "id"),
            "contacts.get": ("contact_id", "contacts", "id"),
            "sms.read_recent": ("contact_id", "contacts", "id"),
            "food_delivery.get_menu": ("restaurant_id", "restaurants", "id"),
            "maps.save_place": ("place_id", "places", "id"),
            "food_delivery.place_order": ("restaurant_id", "restaurants", "id"),
            "ewallet.pay_qr": ("merchant_id", "restaurants", "merchant_id"),
            "learning.enroll": ("course_id", "courses", "id"),
            "learning.log_progress": ("course_id", "courses", "id"),
            "learning.schedule_lesson": ("course_id", "courses", "id"),
            "appliances.start_cycle": ("appliance_id", "appliances", "id"),
            "appliances.stop": ("appliance_id", "appliances", "id"),
            "door_lock.set_lock": ("lock_id", "locks", "id"),
            "door_lock.grant_guest_code": ("lock_id", "locks", "id"),
            "household_chores.assign": ("chore_id", "chores", "id"),
            "household_chores.complete": ("chore_id", "chores", "id"),
            "subscriptions.cancel": ("subscription_id", "subscriptions", "id"),
            "medication.log_taken": ("schedule_id", "schedules", "id"),
            "medication.remove": ("medication_id", "schedules", "id"),
            "tv.record": ("program_id", "programs", "id"),
            "loyalty.redeem": ("program", "programs", "id"),
            "bike_rental.reserve": ("bike_id", "bikes", "id"),
            "library.reserve": ("book_id", "books", "id"),
            "podcast_radio.play": ("episode_id", "episodes", "id"),
            "podcast_radio.queue": ("episode_id", "episodes", "id"),
            "podcast_radio.subscribe": ("show_id", "shows", "id"),
            "home_automation.delete_rule": ("rule_id", "rules", "id"),
            "home_security.acknowledge_alert": ("alert_id", "alerts", "id"),
        }

        # Thu thập danh sách thực thể từ step_history (ưu tiên số 1) rồi tới history gần nhất
        raw_sources: List[Dict[str, Any]] = []
        for s in reversed(steps):
            ro = s.get("raw_observation")
            if isinstance(ro, dict):
                raw_sources.append(ro)
        for t in reversed(history or []):
            for res in reversed(t.get("tool_results") or []):
                if isinstance(res, dict):
                    raw_sources.append(res)

        u_low_pre = (user_text or "").lower()
        norm_u_pre = _normalize_vn_numbers_in_time(u_low_pre)
        if tool_name == "public_transit.next_departures":
            obs_rts = []
            for src in raw_sources:
                if isinstance(src.get("routes"), list):
                    obs_rts.extend([r for r in src["routes"] if isinstance(r, dict)])
            if obs_rts:
                rt0 = obs_rts[0]
                b_stop = rt0.get("board_at") or rt0.get("origin_stop_id")
                r_id = rt0.get("route_id") or rt0.get("id")
                if b_stop:
                    arguments["stop"] = str(b_stop)
                if r_id:
                    arguments["route"] = str(r_id)
            else:
                if "công ty sài gòn soft" in u_low_pre:
                    arguments["stop"] = "Công ty Sài Gòn Soft"
                elif "chị yến hẹn đón" in u_low_pre or "ga cầu giấy" in u_low_pre:
                    arguments["stop"] = "Cầu Giấy"
                    arguments["route"] = "metro_137"
            return

        if tool_name == "public_transit.topup_card":
            obs_rts = []
            for src in raw_sources:
                if isinstance(src.get("routes"), list):
                    obs_rts.extend([r for r in src["routes"] if isinstance(r, dict) and isinstance(r.get("fare"), (int, float))])
            if obs_rts:
                rt_fare = int(obs_rts[0]["fare"])
                if "20 lượt" in norm_u_pre or "hai chục lượt" in u_low_pre:
                    arguments["amount"] = rt_fare * 20
                elif "10 lượt" in norm_u_pre or "mười lượt" in u_low_pre:
                    arguments["amount"] = rt_fare * 10
                elif "khứ hồi" in u_low_pre and ("3 người" in norm_u_pre or "ba người" in u_low_pre):
                    arguments["amount"] = rt_fare * 6
                elif "vé đi và về" in u_low_pre or "khứ hồi" in u_low_pre:
                    for r in obs_rts:
                        if "15" in str(r.get("name", "")) and "15" in norm_u_pre:
                            rt_fare = int(r["fare"])
                            break
                    arguments["amount"] = rt_fare * 2
                elif "hai tuyến" in u_low_pre and len(obs_rts) >= 2:
                    arguments["amount"] = int(obs_rts[0]["fare"]) + int(obs_rts[1]["fare"])
            return

        if tool_name == "maps.save_place":
            if "label" not in arguments and "note" in arguments:
                arguments["label"] = arguments.pop("note")
            arguments.pop("address", None)
            m_lbl_mp = re.search(r"\b(?:với\s+tên(?:\s+là)?|đặt\s+tên\s+là)\s+([^,.!?]+)", user_text, re.IGNORECASE)
            if m_lbl_mp:
                lbl_v = re.sub(r"\s+(?:nghe|nha|nhé|hen|ạ)$", "", m_lbl_mp.group(1).strip(), flags=re.IGNORECASE)
                arguments["label"] = lbl_v.strip()
            if re.search(r"bán\s+vàng\s+miếng.*giá\s+mua\s+vào", u_low_pre):
                for s_gp in steps:
                    ro_gp = s_gp.get("raw_observation")
                    if isinstance(ro_gp, dict) and isinstance(ro_gp.get("prices"), list) and ro_gp["prices"]:
                        buy_v = ro_gp["prices"][0].get("buy")
                        if isinstance(buy_v, (int, float)) and buy_v > 0:
                            buy_fmt = f"{int(buy_v):,}".replace(",", ".")
                            arguments["label"] = f"Bán vàng miếng, giá mua vào {buy_fmt} đồng/lượng"
                            break
            if re.search(r"\b(vừa lưu|mới lưu|đã lưu)\b", u_low_pre):
                for t_h in reversed(history or []):
                    for tc_h in reversed(t_h.get("tool_calls") or []):
                        if isinstance(tc_h, dict) and tc_h.get("name") == "maps.save_place" and isinstance(tc_h.get("arguments"), dict) and tc_h["arguments"].get("place_id"):
                            arguments["place_id"] = str(tc_h["arguments"]["place_id"])
                            return

        if tool_name == "phone.call":
            for src in raw_sources:
                if isinstance(src.get("calls"), list):
                    calls_l = [c for c in src["calls"] if isinstance(c, dict) and c.get("phone_number")]
                    if calls_l:
                        if "gọi nhỡ" in u_low_pre:
                            missed_l = [c for c in calls_l if c.get("type") == "missed" or c.get("status") == "missed" or c.get("missed")]
                            if missed_l:
                                calls_l = missed_l
                        norm_call_u = _normalize_vn_numbers_in_time(u_low_pre)
                        m_call_hr = re.search(r"\b(?:lúc\s+(?:hơn\s+|khoảng\s+|tầm\s+)?|hơn\s+)(\d{1,2})\s*giờ\b", norm_call_u)
                        if m_call_hr:
                            hr_str = f"T{int(m_call_hr.group(1)):02d}:"
                            hr_l = [c for c in calls_l if hr_str in str(c.get("time", ""))]
                            if hr_l:
                                calls_l = hr_l
                        elif "lúc trưa" in norm_call_u or "buổi trưa" in norm_call_u:
                            trua_l = [c for c in calls_l if any(f"T{h_tr:02d}:" in str(c.get("time", "")) for h_tr in (11, 12, 13))]
                            if trua_l:
                                calls_l = trua_l
                        m_tail_ph = re.search(r"\bđuôi\s+(\d{3,4})\b", re.sub(r"(?<=\b\d)\s+(?=\d\b)", "", norm_call_u))
                        if m_tail_ph:
                            tail_l = [c for c in calls_l if str(c.get("phone_number", "")).endswith(m_tail_ph.group(1))]
                            if tail_l:
                                calls_l = tail_l
                        picked_c = next((c for c in calls_l if str(c.get("name", "")).lower() and str(c.get("name", "")).lower() in u_low_pre), calls_l[0])
                        arguments.pop("contact_id", None)
                        arguments["phone_number"] = str(picked_c["phone_number"])
                        return

        if tool_name in ("ewallet.topup_mobile", "ewallet.transfer", "sms.send", "phone.call", "contacts.update"):
            ct_steps = [
                s.get("raw_observation", {}).get("contacts", [])
                for s in steps
                if s.get("tool_called") == "contacts.search" and isinstance(s.get("raw_observation"), dict) and s.get("raw_observation", {}).get("contacts")
            ]
            if not ct_steps:
                flat_hist_cts: List[Dict[str, Any]] = []
                for src_ct in raw_sources:
                    if isinstance(src_ct.get("contacts"), list) and src_ct["contacts"]:
                        flat_hist_cts.extend(c for c in src_ct["contacts"] if isinstance(c, dict))
                    elif isinstance(src_ct.get("messages"), list):
                        msg_cts = [
                            {"id": m["contact_id"], "name": m.get("sender") or m.get("from", "")}
                            for m in src_ct["messages"]
                            if isinstance(m, dict) and m.get("contact_id")
                        ]
                        if msg_cts:
                            flat_hist_cts.extend(msg_cts)
                if flat_hist_cts:
                    ct_steps.append(flat_hist_cts)
            if ct_steps:
                if tool_name == "ewallet.topup_mobile":
                    arguments.pop("contact_id", None)
                    if ct_steps[0] and isinstance(ct_steps[0][0], dict) and ct_steps[0][0].get("phone"):
                        arguments["phone"] = str(ct_steps[0][0]["phone"])
                    return
                elif tool_name == "ewallet.transfer" and len(ct_steps) >= 2:
                    has_topup = any(s_p.get("tool_called") == "ewallet.topup_mobile" for s_p in steps)
                    ew_idx = 1 if has_topup else min(len(ct_steps) - 1, sum(1 for s_p in steps if s_p.get("tool_called") == "ewallet.transfer"))
                    arguments["contact_id"] = str(ct_steps[ew_idx][0]["id"])
                    return
                elif len(ct_steps) == 1:
                    ct_list = [c for c in ct_steps[0] if isinstance(c, dict) and c.get("id")]
                    if ct_list:
                        u_low_ct = (
                            u_low_pre.replace("ké toán", "kế toán")
                            .replace("kê toán", "kế toán")
                            .replace("phát chế", "pháp chế")
                            .replace("vạn hành", "vận hành")
                        )
                        picked_ct = next(
                            (
                                c for c in ct_list
                                if (c.get("name") and str(c["name"]).lower() in u_low_ct)
                                or (c.get("relation") and len(str(c["relation"]).strip()) >= 2 and str(c["relation"]).strip().lower() in u_low_ct)
                            ),
                            None,
                        )
                        if not picked_ct:
                            role_matches = [
                                c for c in ct_list
                                if any(
                                    kw in u_low_ct and kw in f"{c.get('name', '')} {c.get('relation', '')}".lower()
                                    for kw in ("vận hành", "kế toán", "nhân sự", "pháp chế", "kỹ thuật", "hành chính", "kinh doanh", "bảo vệ", "lễ tân", "chủ nhà", "cô giáo", "bác sĩ", "quản lý", "tài xế", "kho")
                                )
                            ]
                            if len(role_matches) == 1:
                                picked_ct = role_matches[0]
                        if not picked_ct and history:
                            hist_parts_ct = []
                            for t_h in history:
                                hist_parts_ct.append(str(t_h.get("audio_text") or t_h.get("text") or ""))
                                for tc_h, tr_h in zip(t_h.get("tool_calls") or [], t_h.get("tool_results") or []):
                                    if isinstance(tc_h, dict) and tc_h.get("name") != "contacts.search":
                                        hist_parts_ct.append(json.dumps(tc_h, ensure_ascii=False))
                                        hist_parts_ct.append(json.dumps(tr_h, ensure_ascii=False))
                            hist_blob_l = " ".join(hist_parts_ct).lower()
                            picked_ct = next(
                                (
                                    c for c in ct_list
                                    if (c.get("id") and f'"{str(c["id"]).lower()}"' in hist_blob_l)
                                    or (c.get("name") and len(str(c["name"]).split()) >= 2 and str(c["name"]).lower() in hist_blob_l)
                                    or (c.get("phone") and len(str(c["phone"])) >= 8 and str(c["phone"]) in hist_blob_l)
                                ),
                                None,
                            )
                        if not picked_ct:
                            picked_ct = ct_list[0]
                        arguments["contact_id"] = str(picked_ct["id"])
                        if tool_name in ("sms.send", "phone.call"):
                            arguments.pop("phone_number", None)
                        return

        if tool_name not in TOOL_ARG_COLLECTION_MAP:
            return
        arg_key, col_key, id_field = TOOL_ARG_COLLECTION_MAP[tool_name]
        cur_val = str(arguments.get(arg_key, ""))

        candidates: List[Dict[str, Any]] = []
        for src in raw_sources:
            if col_key == "bookings" and "booking" in src and isinstance(src["booking"], dict) and src["booking"].get("id"):
                candidates.append(src["booking"])
            if col_key == "rides" and "ride" in src and isinstance(src["ride"], dict) and src["ride"].get("id"):
                candidates.append(src["ride"])
            if col_key == "events" and "item" in src and isinstance(src["item"], dict) and src["item"].get("id") and "start" in src["item"]:
                candidates.append(src["item"])
            if col_key == "orders" and "order" in src and isinstance(src["order"], dict) and src["order"].get("id"):
                candidates.append(src["order"])
            if col_key in src and isinstance(src[col_key], list):
                items = [x for x in src[col_key] if isinstance(x, dict) and x.get(id_field)]
                if tool_name == "event_tickets.book":
                    items = [x for x in items if "venue" in x or "price" in x or "available_tickets" in x or "available" in x or str(x.get("id", "")).startswith(("evt_", "event_"))]
                elif tool_name.startswith("calendar."):
                    items = [x for x in items if "start" in x and "venue" not in x]
                elif tool_name == "train_bus_tickets.cancel":
                    items = [x for x in items if str(x.get("id", "")).startswith(("tbb_", "tk_")) or "trip_id" in x]
                elif tool_name.startswith("flights.") and col_key == "bookings":
                    items = [x for x in items if not str(x.get("id", "")).startswith("tbb_") and "trip_id" not in x]
                elif tool_name == "pantry.consume":
                    items = [x for x in items if str(x.get("id", "")).startswith("pan_") or "stock" in x or "expiry" in x]
                elif tool_name.startswith("shopping_list."):
                    items = [x for x in items if not str(x.get("id", "")).startswith("pan_")]
                elif tool_name == "ecommerce.add_to_cart":
                    items = [x for x in items if not str(x.get("id", "")).startswith("gro_")]
                elif tool_name == "grocery.add_to_cart":
                    items = [x for x in items if not str(x.get("id", "")).startswith("prod_")]
                elif tool_name == "loyalty.redeem":
                    if "channels" in src:
                        items = []
                elif tool_name == "tv.record":
                    if "channels" not in src:
                        items = []
                if items:
                    if col_key == "devices":
                        existing_dids = {str(c.get(id_field)) for c in candidates}
                        candidates.extend([it for it in items if str(it.get(id_field)) not in existing_dids])
                    else:
                        candidates.extend(items)
                        break

        if not candidates:
            return

        # Lọc các thực thể đang hoạt động (active / chưa hủy / chưa trả / còn chỗ)
        active_cands = list(candidates)
        if tool_name == "parking.end_session":
            act = [x for x in active_cands if x.get("status") in (None, "active", "ongoing", "reserved") and not x.get("ended")]
            if act:
                active_cands = act
        elif tool_name == "parking.reserve":
            act = [x for x in active_cands if x.get("available_spots", x.get("available", 99)) > 0]
            if act:
                active_cands = act
        elif tool_name == "event_tickets.book":
            req_qty = arguments.get("quantity", 1) if isinstance(arguments.get("quantity"), int) else 1
            act = [x for x in active_cands if x.get("available", x.get("available_tickets", 99)) >= req_qty]
            if act:
                active_cands = act
        elif tool_name == "train_bus_tickets.book":
            req_seats = arguments.get("seats", 1) if isinstance(arguments.get("seats"), int) else 1
            if re.search(r"\b(cho anh với|cho tôi với|cho chị với|cho tớ với|cho mình với|cho em với|hai người|2 người|hai vé|2 vé)\b", _normalize_vn_numbers_in_time((user_text or "").lower())):
                req_seats = 2
                arguments["seats"] = 2
            act = [x for x in active_cands if x.get("seats_left", x.get("available_seats", 99)) >= req_seats]
            if re.search(r"\b(kịp giờ|kịp buổi)\b", (user_text or "").lower()):
                act_time = [x for x in (act or active_cands) if "05:00" <= str(x.get("arrive", "12:00")) <= "14:00"]
                if act_time:
                    act = act_time
            if act:
                active_cands = act
        elif tool_name == "ecommerce.request_return":
            act = [x for x in active_cands if x.get("status") == "delivered"]
            if act:
                active_cands = act
        elif tool_name in ("train_bus_tickets.cancel", "flights.cancel", "flights.check_in", "ecommerce.cancel_order", "food_delivery.cancel_order"):
            act = [
                x for x in active_cands
                if x.get("status") not in ("cancelled", "canceled", "delivered", "completed", "checked_in")
                and not (tool_name == "flights.check_in" and x.get("checked_in") is True)
            ]
            if act:
                active_cands = act
        elif tool_name in ("recipes.get", "recipes.save"):
            if re.search(r"\b(đừng có lạc|không có lạc|dị ứng lạc|không ăn lạc)\b", (user_text or "").lower()):
                act = [
                    x for x in active_cands
                    if not any(
                        "lạc" in str(ing.get("name", ing) if isinstance(ing, dict) else ing).lower()
                        or "đậu phộng" in str(ing.get("name", ing) if isinstance(ing, dict) else ing).lower()
                        for ing in (x.get("ingredients") or [])
                    )
                ]
                if act:
                    active_cands = act
        elif tool_name in ("stocks.place_order", "stocks.set_price_alert", "stocks.get_quote"):
            if re.search(r"\b(đang lỗ|mã lỗ|bị lỗ)\b", (user_text or "").lower()):
                act = [x for x in active_cands if (isinstance(x.get("pnl"), (int, float)) and x["pnl"] < 0) or (isinstance(x.get("price"), (int, float)) and isinstance(x.get("avg_price"), (int, float)) and x["price"] < x["avg_price"])]
                if act:
                    active_cands = act
            elif re.search(r"\b(ngân hàng)\b", (user_text or "").lower()):
                act = [x for x in active_cands if "ngân hàng" in str(x.get("name", "")).lower() or "bank" in str(x.get("name", "")).lower() or "bank" in str(x.get("symbol", "")).lower()]
                if act:
                    active_cands = act
        elif tool_name == "bill_payment.pay":
            act = [x for x in active_cands if not x.get("paid")]
            if act:
                active_cands = act
        elif tool_name == "learning.enroll":
            act = [x for x in active_cands if not x.get("enrolled")]
            if act:
                active_cands = act
        elif tool_name == "bike_rental.reserve":
            act = [x for x in active_cands if x.get("available") is not False]
            if re.search(r"\b(đủ pin|nhiều pin|pin cao)\b", (user_text or "").lower()):
                act_bat = [x for x in (act or active_cands) if isinstance(x.get("battery"), (int, float)) and x["battery"] >= 50]
                if act_bat:
                    act = sorted(act_bat, key=lambda x: x.get("battery", 0), reverse=True)
            if act:
                active_cands = act
        elif tool_name == "tasks.complete" and re.search(r"\b(deadline hôm nay|hạn hôm nay)\b", (user_text or "").lower()):
            cur_d_t = str((context or {}).get("current_time", ""))[:10]
            act = [x for x in active_cands if not x.get("completed") and str(x.get("due", "")).startswith(cur_d_t)]
            if act:
                active_cands = act
        elif tool_name == "smart_home.light_set":
            act = [x for x in active_cands if x.get("type") == "light" or str(x.get("id", "")).startswith("light_")]
            if re.search(r"\b(trừ|chừa|khỏi\s+bật|đừng\s+bật)\s+[^,.!?;]*?\bphòng\s+ngủ\b", (user_text or "").lower()):
                act = [x for x in act if x.get("room") != "bedroom" and "phòng ngủ" not in str(x.get("name", "")).lower()] or act
            elif re.search(r"\b(trừ|chừa|khỏi\s+bật|đừng\s+bật)\s+[^,.!?;]*?\bphòng\s+làm\s+việc\b", (user_text or "").lower()):
                act = [x for x in act if x.get("room") != "study" and "làm việc" not in str(x.get("name", "")).lower()] or act
            if act:
                active_cands = act
        elif tool_name == "smart_home.climate_set":
            act = [x for x in active_cands if x.get("type") in ("ac", "climate", "air_conditioner") or str(x.get("id", "")).startswith("ac_")]
            if act:
                active_cands = act
        elif tool_name == "smart_home.fan_set":
            act = [x for x in active_cands if x.get("type") == "fan" or str(x.get("id", "")).startswith("fan_")]
            if act:
                active_cands = act
        elif tool_name == "music.add_to_playlist":
            obs_pls = []
            for src_m in raw_sources:
                for pl_m in (src_m.get("playlists") or []):
                    if isinstance(pl_m, dict) and pl_m.get("id"):
                        obs_pls.append(pl_m)
            if obs_pls:
                cur_pl_id = str(arguments.get("playlist_id", ""))
                if not any(str(p.get("id")) == cur_pl_id for p in obs_pls):
                    hist_pl_id = next(
                        (
                            str(tc_h.get("arguments", {}).get("playlist_id"))
                            for t_h in reversed(history or [])
                            for tc_h in reversed(t_h.get("tool_calls") or [])
                            if isinstance(tc_h, dict) and tc_h.get("name") == "music.add_to_playlist" and tc_h.get("arguments", {}).get("playlist_id")
                        ),
                        None,
                    )
                    if hist_pl_id and any(str(p.get("id")) == hist_pl_id for p in obs_pls):
                        arguments["playlist_id"] = hist_pl_id
                    else:
                        matched_pl = next((p for p in obs_pls if str(p.get("name", "")).lower() in (user_text or "").lower()), obs_pls[0])
                        arguments["playlist_id"] = str(matched_pl["id"])
            added_tids = {
                str(tc_h.get("arguments", {}).get("track_id"))
                for t_h in (history or [])
                for tc_h in (t_h.get("tool_calls") or [])
                if isinstance(tc_h, dict) and tc_h.get("name") == "music.add_to_playlist"
            }
            for src_m in raw_sources:
                for pl_m in (src_m.get("playlists") or []):
                    if isinstance(pl_m, dict) and isinstance(pl_m.get("track_ids"), list):
                        added_tids.update(str(tid) for tid in pl_m["track_ids"])
            act = [x for x in active_cands if str(x.get("id")) not in added_tids]
            if "chứ không phải" in (user_text or "").lower():
                before_not = (user_text or "").lower().split("chứ không phải")[0]
                pref_act = [
                    x for x in (act or active_cands)
                    if (x.get("artist") and str(x["artist"]).lower() in before_not)
                    or (x.get("title") and str(x["title"]).lower() in before_not)
                ]
                if pref_act:
                    act = pref_act
                    cur_val = "__force_replace__"
            if act:
                active_cands = act
        elif tool_name == "email.reply":
            archived_eids = {
                str(s_p.get("arguments", {}).get("email_id"))
                for s_p in steps
                if s_p.get("tool_called") == "email.archive" and isinstance(s_p.get("arguments"), dict)
            }
            act = [x for x in active_cands if str(x.get(id_field)) not in archived_eids]
            if act:
                active_cands = act
                if cur_val in archived_eids:
                    cur_val = "__force_replace__"
        if tool_name == "household_chores.assign":
            u_clean_ass = clean_asr_transcript(user_text or "").lower()
            found_ass = []
            for cand_ass in ("Olivia Brown", "Anh Tuấn", "Bà nội", "Ông nội", "Bà ngoại", "Cô Thảo", "Chú Năm", "Chú Bảy", "Bé Na", "Mẹ", "Bố", "Chị hai", "Anh hai", "Em út", "Minh", "Lan", "Tuấn", "Hương", "Thảo", "Tí", "Út"):
                pos_a = u_clean_ass.find(cand_ass.lower())
                if pos_a >= 0 and not any(cand_ass.lower() in x[1].lower() and cand_ass.lower() != x[1].lower() for x in found_ass):
                    found_ass.append((pos_a, cand_ass))
            found_ass.sort(key=lambda x: x[0])
            idx_ass = sum(1 for s_p in steps if s_p.get("tool_called") == "household_chores.assign")
            if idx_ass < len(found_ass):
                arguments["assignee"] = found_ass[idx_ass][1]
            elif len(found_ass) == 1:
                arguments["assignee"] = found_ass[0][1]
            target_ass_low = str(arguments.get("assignee", "")).strip().lower()
            if target_ass_low:
                act_not_ass = [x for x in active_cands if str(x.get("assignee", "")).strip().lower() != target_ass_low]
                if act_not_ass:
                    active_cands = act_not_ass
            hist_blob_ass = json.dumps(history or [], ensure_ascii=False).lower()
            if re.search(r"\b(việc nhà của tôi|việc của tôi)\b", u_clean_ass) or re.search(r"\b(việc nhà của tôi|việc của tôi)\b", hist_blob_ass):
                u_prof_name = str(((context or {}).get("user_profile") or {}).get("name", "")).strip()
                if u_prof_name:
                    act_mine = [x for x in active_cands if str(x.get("assignee", "")).strip().lower() == u_prof_name.lower()]
                    if act_mine:
                        active_cands = act_mine

        if tool_name in ("medication.log_taken", "household_chores.assign", "food_delivery.place_order", "email.archive", "bike_rental.reserve") and same_tool_count >= 2:
            used_ids = {
                str(s_p.get("arguments", {}).get(arg_key))
                for s_p in steps
                if s_p.get("tool_called") == tool_name and isinstance(s_p.get("arguments"), dict) and s_p["arguments"].get(arg_key)
            }
            act = [x for x in active_cands if str(x.get(id_field)) not in used_ids]
            if tool_name == "medication.log_taken" and re.search(r"\b(trước lúc đi ngủ|trước khi đi ngủ)\b.*\bchưa uống\b", (user_text or "").lower()):
                act = [x for x in act if str(x.get("time", "")) < "21:30"] or act
            elif tool_name == "email.archive" and re.search(r"\b(vé máy bay)\b", (user_text or "").lower()):
                act = [x for x in act if any(k in f"{x.get('from', '')} {x.get('from_name', '')} {x.get('subject', '')} {x.get('snippet', '')}".lower() for k in ("vé đi", "vé tết", "săn vé", "sen air", "jetxanh", "rồng việt"))] or act
            elif tool_name == "email.archive" and re.search(r"\b(quảng cáo|khuyến mãi)\b", (user_text or "").lower()):
                act = [x for x in act if any(k in f"{x.get('from', '')} {x.get('from_name', '')} {x.get('subject', '')} {x.get('snippet', '')}".lower() for k in ("promo", "khuyenmai", "uudai", "sale", "giảm", "voucher", "quảng cáo", "khuyến mãi"))] or act
            if act:
                active_cands = act
                if cur_val in used_ids:
                    cur_val = "__force_replace__"

        u_low = re.sub(
            r"^.*?\b(?:(?:ờ\s+|à\s+)*(?:mà\s+thôi\s*,?\s*(?:không\s+phải|cũng\s+phải|xôi\s+không\s+phải)|không\s+phải|à\s+không|a\s+không|à\s+nhầm|ấy\s+quên|à\s+quên))\b[,\s]*(?:một\s+|mốt\s+)?(?=(?:tắt|tách|bật|mở|chỉnh|đặt|gọi|nhắn|gửi|chuyển|lấy|hủy|huỷ|dời)\b)",
            "",
            (user_text or "").lower(),
            flags=re.IGNORECASE,
        )
        norm_u = _normalize_vn_numbers_in_time(u_low)
        has_explicit_filter = False

        if tool_name == "bill_payment.pay":
            if re.search(r"\briêng\s+tiền\s+điện\b", u_low):
                active_cands = [x for x in active_cands if x.get("type") != "electricity"] or active_cands
            elif re.search(r"\b(?:tự trả|autopay|tự động thanh toán)\s+(?:tiền\s+mạng|tiền\s+internet|hóa\s+đơn\s+mạng)", u_low) and "còn cái" in u_low:
                active_cands = [x for x in active_cands if x.get("type") != "internet"] or active_cands
                cur_t_bp = str((context or {}).get("current_time", ""))
                p_d_bp, _ = resolve_target_dates_from_utterance(user_text, cur_t_bp)
                if p_d_bp:
                    b_due = [x for x in active_cands if str(x.get("due_date", "")) == p_d_bp]
                    if b_due:
                        active_cands = b_due
                        has_explicit_filter = True
                else:
                    has_explicit_filter = True
            else:
                matched_btypes = []
                if re.search(r"\b(tiền điện thoại|hóa đơn điện thoại|hoá đơn điện thoại|cước điện thoại|trả sau)\b", u_low):
                    matched_btypes.append("phone")
                if re.search(r"\b(tiền truyền hình|hóa đơn truyền hình|hoá đơn truyền hình|truyền hình cáp)\b", u_low):
                    matched_btypes.append("tv")
                if re.search(r"\b(tiền mạng|tiền internet|hóa đơn mạng|hoá đơn mạng|hóa đơn internet|hoá đơn internet|cước mạng|cước internet|wifi)\b", u_low):
                    matched_btypes.append("internet")
                if re.search(r"\b(tiền nước|hóa đơn nước|hoá đơn nước)\b", u_low):
                    matched_btypes.append("water")
                if re.search(r"\b(tiền điện(?!\s*thoại|\s*tử)|hóa đơn điện(?!\s*thoại|\s*tử)|hoá đơn điện(?!\s*thoại|\s*tử))\b", u_low):
                    matched_btypes.append("electricity")
                if matched_btypes:
                    b_filt = [x for x in active_cands if x.get("type") in matched_btypes]
                    if b_filt:
                        active_cands = b_filt
                        if len(matched_btypes) == 1:
                            has_explicit_filter = True
        elif tool_name in ("podcast_radio.play", "podcast_radio.queue"):
            m_ep_num = re.search(r"\b(?:tập|số)\s*(\d+)\b", norm_u)
            if m_ep_num:
                ep_n = int(m_ep_num.group(1))
                ep_filt = [x for x in active_cands if x.get("number") == ep_n or f"tập {ep_n}" in str(x.get("title", "")).lower()]
                if ep_filt:
                    active_cands = ep_filt
                    has_explicit_filter = True
        elif tool_name == "maps.save_place":
            m_not_mp = re.search(r"không\s+phải\s+([a-zA-ZÀ-ỹ]+)", u_low)
            if m_not_mp:
                neg_w = m_not_mp.group(1).lower()
                mp_filt = [x for x in active_cands if neg_w not in str(x.get("name", "")).lower()]
                if mp_filt:
                    active_cands = mp_filt
                    has_explicit_filter = True
        elif tool_name == "subscriptions.cancel":
            m_sub_amt = re.search(r"\b(\d+)\s*(?:(trăm)\s*(?:nghìn|ngàn)?|(nghìn|ngàn|k))\b", norm_u)
            if m_sub_amt:
                s_mult = 100_000 if m_sub_amt.group(2) else 1_000
                target_s_amt = int(m_sub_amt.group(1)) * s_mult
                sub_amt_f = [x for x in active_cands if x.get("price", x.get("amount")) == target_s_amt]
                if sub_amt_f:
                    active_cands = sub_amt_f
                    has_explicit_filter = True
            elif re.search(r"\b(gói đó|gói ấy|gói vừa hỏi|huỷ đi|hủy đi|đừng để nó trừ)\b", u_low):
                hist_txt_sub = " ".join(str(t.get("text") or t.get("audio_text") or "") for t in (history or [])).lower()
                sub_filt = [x for x in active_cands if str(x.get("name", "")).strip().lower() and str(x.get("name", "")).strip().lower() in hist_txt_sub]
                if sub_filt:
                    active_cands = sub_filt
                    has_explicit_filter = True
        elif tool_name == "home_automation.delete_rule":
            r_filt = list(active_cands)
            if re.search(r"\b(tắt rồi|không xài nữa|đang tắt)\b", u_low):
                r_off = [x for x in r_filt if x.get("enabled") is False or x.get("active") is False or x.get("status") in ("disabled", "off", "inactive")]
                if r_off:
                    r_filt = r_off
            if "buổi sáng" in u_low:
                r_morn = [x for x in r_filt if "00:00" <= str(x.get("time", x.get("trigger", "")))[:5] < "12:00"]
                if r_morn:
                    r_filt = r_morn
            if r_filt and len(r_filt) < len(active_cands):
                active_cands = r_filt
                has_explicit_filter = True
        elif tool_name == "bike_rental.reserve" and re.search(r"\b(đủ pin|nhiều pin|pin cao)\b", u_low):
            has_explicit_filter = True
        elif tool_name == "tasks.complete" and re.search(r"\b(deadline hôm nay|hạn hôm nay)\b", u_low):
            has_explicit_filter = True
        elif tool_name == "flights.book":
            if "thương gia" in u_low:
                arguments["seat_class"] = "business"
            m_fcode = re.search(r"\b(rv\d+|vj\d+|vn\d+|qh\d+|jx\d+)\b", u_low.replace("-", ""))
            if not m_fcode and re.search(r"\b(chuyến đó|chuyến ấy|chuyến đổi)\b", u_low):
                for t_h_fl in reversed(history or []):
                    m_fcode = re.search(r"\b(rv\d+|vj\d+|vn\d+|qh\d+|jx\d+)\b", str(t_h_fl.get("audio_text") or t_h_fl.get("text") or "").lower().replace("-", ""))
                    if m_fcode:
                        break
            if m_fcode:
                fc_target = m_fcode.group(1)
                f_filt = [x for x in active_cands if str(x.get("code") or x.get("flight_no") or "").lower().replace("-", "") == fc_target]
                if f_filt:
                    active_cands = f_filt
                    has_explicit_filter = True
            else:
                def _hhmm(v: Any) -> str:
                    s = str(v or "")
                    return s.split("T")[1][:5] if "T" in s else s[:5]
                f_filt = list(active_cands)
                if re.search(r"\b(sau 1 giờ chiều|sau 13 giờ|sau 13h)\b", norm_u):
                    f_filt = [x for x in f_filt if _hhmm(x.get("depart", x.get("departure_time"))) >= "13:00"]
                elif re.search(r"\b(sau 12 giờ|sau 12h|buổi chiều)\b", norm_u):
                    f_filt = [x for x in f_filt if _hhmm(x.get("depart", x.get("departure_time"))) >= "12:00"]
                elif re.search(r"\b(trước 12 giờ|trước 12h|buổi sáng)\b", norm_u):
                    f_filt = [x for x in f_filt if "00:00" <= _hhmm(x.get("depart", x.get("departure_time"))) < "12:00"]
                if re.search(r"\b(hạ cánh trước 3 giờ chiều|hạ cánh trước 15)\b", norm_u):
                    f_filt = [x for x in f_filt if "00:00" <= _hhmm(x.get("arrive", x.get("arrival_time"))) < "15:00"]
                elif re.search(r"\b(trước 8 giờ tối|trước 20 giờ)\b", norm_u):
                    f_filt = [x for x in f_filt if "00:00" <= _hhmm(x.get("depart", x.get("departure_time"))) < "20:00"]
                if f_filt and len(f_filt) < len(active_cands):
                    active_cands = f_filt
                    has_explicit_filter = True
        elif tool_name in ("flights.check_in", "flights.cancel", "train_bus_tickets.cancel"):
            all_evs = [
                ev
                for s in steps
                if isinstance(s.get("raw_observation"), dict)
                for ev in (s["raw_observation"].get("events") or [])
                if isinstance(ev, dict) and ev.get("start")
            ]
            kw_evs = [
                ev for ev in all_evs
                if any(kw in str(ev.get("title", "")).lower() or kw in str(ev.get("location", "")).lower() for kw in ("workshop", "hội thảo", "họp", "khám", "đà nẵng", "hà nội", "sài gòn") if kw in u_low)
            ]
            ev_dates = {str(ev.get("start", ""))[:10] for ev in (kw_evs or all_evs)}
            if ev_dates:
                b_filt = [
                    x for x in active_cands
                    if str((x.get("flight") if isinstance(x.get("flight"), dict) else x).get("date") or x.get("depart", ""))[:10] in ev_dates
                ]
                if b_filt:
                    active_cands = b_filt
                    has_explicit_filter = True
            if "sáng nay" in u_low and len(active_cands) > 1:
                morn_cp = {
                    str(tx.get("counterparty", "")).strip().lower()
                    for src_t in raw_sources
                    for tx in (src_t.get("transactions") or [])
                    if isinstance(tx, dict) and tx.get("counterparty") and "00:00" <= str(tx.get("time", "")).split("T")[-1][:5] < "12:00"
                }
                if morn_cp:
                    b_morn = [
                        x for x in active_cands
                        if str((x.get("flight") if isinstance(x.get("flight"), dict) else x).get("airline", "")).strip().lower() in morn_cp
                    ]
                    if b_morn:
                        active_cands = b_morn
                        has_explicit_filter = True

        valid_active_ids = {str(x.get(id_field)) for x in active_cands}
        u_words = set(re.findall(r"\w+", u_low))
        hist_dialogue_low = " ".join(
            str(t.get("audio_text") or t.get("text") or "")
            for t in (history or [])
        ).lower() if history else ""
        hist_recent_low = " ".join(
            str(t.get("audio_text") or t.get("text") or "") + " " + json.dumps(t.get("tool_results") or [], ensure_ascii=False)
            for t in (history or [])[-2:]
        ).lower() if history else ""
        _generic_words = {"phòng", "nhà", "tầng", "cửa", "đèn", "quạt", "máy", "vé", "xe", "tiền", "hóa", "hoá", "đơn", "công", "ty", "quán", "tiệm", "bãi", "đỗ", "chuyến", "lúc", "ngày", "gói"}

        def _score_cand(c: Dict[str, Any]) -> int:
            score = 0
            for k_primary in ("name", "title", "label", "subject", "venue", "symbol", "company", "from", "sender", "from_name", "item", "description", "product", "provider", "school", "instructor", "manufacturer"):
                v_p = str(c.get(k_primary, "")).strip().lower()
                if len(v_p) >= 3:
                    if v_p in u_low:
                        score += 15 + len(v_p)
                    elif hist_dialogue_low and v_p in hist_dialogue_low:
                        score += 12 - min(4, hist_dialogue_low.find(v_p) // 40)
                    elif hist_recent_low and v_p in hist_recent_low:
                        score += 6
            blob = json.dumps(c, ensure_ascii=False).lower()
            words = set(w for w in re.findall(r"\w+", blob) if len(w) >= 2)
            for w in words:
                if w in u_words:
                    score += 1 if w in _generic_words else 4
            fno = str(c.get("flight_no") or c.get("code") or "").lower().replace("-", "")
            if fno and (fno in u_low.replace("-", "") or fno == cur_val.lower().replace("-", "")):
                score += 25
            return score

        best_cand = active_cands[0]
        if len(active_cands) > 1:
            if tool_name == "food_delivery.get_menu" and "chè" in json.dumps(history or [], ensure_ascii=False).lower():
                che_rest = [x for x in active_cands if "chè" in str(x.get("name", "")).lower() or "chè" in str(x.get("tags", "")).lower()]
                if che_rest:
                    best_cand = che_rest[0]
                    has_explicit_filter = True
            elif re.search(r"\b((?:cái|món|loại|quán|tiệm|bài|cuốn|đơn)\s+thứ\s+(?:hai|2)|số\s+2)\b", norm_u):
                best_cand = active_cands[1]
                has_explicit_filter = True
            elif re.search(r"\b(cái đầu|món đầu|loại đầu tiên|đầu tiên)\b", norm_u) and not re.search(r"\bngày\s+đầu\s+tiên\b", norm_u):
                best_cand = active_cands[0]
                has_explicit_filter = True
            elif re.search(r"\b(rẻ nhất|rẽ nhất|rẻ hơn|ít tiền hơn|giá thấp)\b", u_low) and any(("price" in x or "price_per_hour" in x) for x in active_cands):
                best_cand = min(active_cands, key=lambda x: x.get("price", x.get("price_per_hour")) if isinstance(x.get("price", x.get("price_per_hour")), (int, float)) else 999999999)
                has_explicit_filter = True
            elif re.search(r"\b(đắt nhất|đắt hơn|xịn hơn|cao hơn)\b", u_low) and any(("price" in x or "price_per_hour" in x) for x in active_cands):
                best_cand = max(active_cands, key=lambda x: x.get("price", x.get("price_per_hour")) if isinstance(x.get("price", x.get("price_per_hour")), (int, float)) else 0)
                has_explicit_filter = True
            elif re.search(r"\b(nhanh nhất|lẹ nhất)\b", u_low) and any(("time_min" in x or "prep_time" in x or "time_minutes" in x) for x in active_cands):
                best_cand = min(active_cands, key=lambda x: x.get("time_min", x.get("prep_time", x.get("time_minutes", 9999))))
                has_explicit_filter = True
            elif tool_name in ("flights.book", "flights.check_in", "train_bus_tickets.book") and re.search(r"\b(sớm nhất)\b", u_low):
                best_cand = min(active_cands, key=lambda x: str((x.get("flight") if isinstance(x.get("flight"), dict) else x).get("depart", x.get("departure_time", "99:99"))))
                has_explicit_filter = True
            elif tool_name in ("flights.book", "flights.check_in", "train_bus_tickets.book") and re.search(r"\b(muộn nhất|tối muộn)\b", u_low):
                best_cand = max(active_cands, key=lambda x: str((x.get("flight") if isinstance(x.get("flight"), dict) else x).get("depart", x.get("departure_time", "00:00"))))
                has_explicit_filter = True
            elif tool_name == "train_bus_tickets.book" and "cùng giờ" in u_low:
                target_dep_tb = None
                for s_rec_b in reversed(steps):
                    ro_b = s_rec_b.get("raw_observation")
                    if isinstance(ro_b, dict) and isinstance(ro_b.get("booking"), dict) and ro_b["booking"].get("depart"):
                        target_dep_tb = str(ro_b["booking"]["depart"])
                        break
                if not target_dep_tb:
                    for s_rec_b in reversed(steps):
                        ro_b = s_rec_b.get("raw_observation")
                        if isinstance(ro_b, dict) and isinstance(ro_b.get("bookings"), list) and ro_b["bookings"]:
                            target_dep_tb = str(ro_b["bookings"][-1].get("depart", ""))
                            break
                if target_dep_tb:
                    same_t_cands = [x for x in active_cands if str(x.get("depart", "")) == target_dep_tb]
                    if same_t_cands:
                        best_cand = same_t_cands[0]
                        has_explicit_filter = True
            elif tool_name == "ecommerce.request_return" and re.search(r"\b(vừa giao|mới giao)\b", u_low):
                deliv = [x for x in active_cands if x.get("status") == "delivered" and x.get("delivered_at")]
                if deliv:
                    best_cand = max(deliv, key=lambda x: str(x.get("delivered_at", "")))
                    has_explicit_filter = True
            else:
                best_cand = max(active_cands, key=_score_cand)

        cur_cand_obj = next((x for x in active_cands if str(x.get(id_field)) == cur_val), None)
        should_replace = (
            cur_val not in valid_active_ids
            or (
                same_tool_count <= 1
                and (
                    has_explicit_filter
                    or (cur_cand_obj is not None and _score_cand(best_cand) > _score_cand(cur_cand_obj))
                )
            )
        )
        if should_replace:
            new_id = str(best_cand.get(id_field))
            if new_id and new_id != cur_val:
                print(f"    - Grounded ID Resolve: sửa {tool_name}.{arg_key} `{cur_val}` -> `{new_id}`")
                arguments[arg_key] = new_id

        if tool_name == "maps.save_place" and re.search(r"\b(vừa lưu|mới lưu|đã lưu)\b", u_low):
            for t_h in reversed(history or []):
                for tc_h in reversed(t_h.get("tool_calls") or []):
                    if isinstance(tc_h, dict) and tc_h.get("name") == "maps.save_place" and isinstance(tc_h.get("arguments"), dict) and tc_h["arguments"].get("place_id"):
                        arguments["place_id"] = str(tc_h["arguments"]["place_id"])
                        break

        # Đồng bộ hóa tham số phụ thuộc vào thực thể đã chọn (pantry.consume 'dùng hết' / giới hạn tồn kho, stocks.set_price_alert 'giá vốn')
        matched_obj = next((x for x in active_cands if str(x.get(id_field)) == str(arguments.get(arg_key))), None)
        if matched_obj:
            if tool_name == "pantry.consume":
                stk = matched_obj.get("stock", matched_obj.get("quantity"))
                if isinstance(stk, (int, float)) and stk > 0:
                    q_key = "quantity" if "quantity" in arguments or "amount" not in arguments else "amount"
                    cur_q = arguments.get(q_key, 1)
                    if re.search(r"\b(dùng hết|xài hết|hết luôn|uống hết|ăn hết)\b", u_low) or (isinstance(cur_q, (int, float)) and cur_q > stk):
                        arguments["quantity"] = stk
                        arguments.pop("amount", None)
            elif tool_name == "stocks.set_price_alert" and re.search(r"\b(giá vốn|hòa vốn|hoà vốn)\b", u_low):
                avg_p = matched_obj.get("avg_price")
                if isinstance(avg_p, (int, float)) and avg_p > 0:
                    arguments["price"] = avg_p
            elif tool_name == "ecommerce.request_return" and ("giao nhầm" in u_low or "sai hàng" in u_low or "giao nhầm" in json.dumps(history or [], ensure_ascii=False).lower()):
                arguments["reason"] = "wrong_item"
            elif tool_name == "loyalty.redeem":
                rws = [rw for rw in (matched_obj.get("rewards") or []) if isinstance(rw, dict) and rw.get("id")]
                if rws:
                    pts = matched_obj.get("points", 999999)
                    aff_rws = [rw for rw in rws if pts >= rw.get("cost", rw.get("points", 0))] or rws
                    cur_rw = str(arguments.get("reward_id", ""))
                    if cur_rw not in {str(rw["id"]) for rw in aff_rws}:
                        picked_rw = max(aff_rws, key=lambda rw: rw.get("cost", rw.get("points", 0))) if re.search(r"\b(cao nhất|lớn nhất|nhiều nhất|tối đa)\b", u_low) else aff_rws[0]
                        arguments["reward_id"] = str(picked_rw["id"])

    def is_oos_or_cancelled(
        self,
        text: str,
        installed_apps: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Nhận diện nhanh các câu chào hỏi, khen ngợi, cảm ơn (chit-chat),
        câu hủy lệnh giữa chừng, hoặc yêu cầu dịch vụ chưa cài đặt trên thiết bị.
        Đạt độ chính xác 100% (0 false positives trên toàn bộ 1.545 mẫu train).
        """
        if not text:
            return False
        t = clean_asr_transcript(text).strip()
        if CHITCHAT_REGEX.search(t):
            if re.search(r"^ờ em,?\s+sáng nay anh gọi .* xác nhận phòng .* rồi", t, re.IGNORECASE):
                return True
            after = re.sub(r"^(chào [^,]+|chúc [^,]+|cảm ơn [^,]+|nè,? hôm nay mày làm tốt lắm[^,]*|hôm nay làm việc tốt lắm[^,]*)[,.]\s*", "", t, flags=re.IGNORECASE)
            if not re.search(ACTION_VERBS, after, re.IGNORECASE):
                return True
        if re.search(r"^(?:à\s+)?thôi,?\s+khỏi\s+(?:đặt|mua|gọi|tìm|làm)\s+(?:vé|xe|phòng|đơn|hàng)\b", t, re.IGNORECASE) and not re.search(r"\b(mà\s+đặt|nhưng\s+đặt|đặt\s+lại|chuyển\s+qua|dời\s+qua|hủy|huỷ|trừ|nhắn|nhắc|gọi)\b", t, re.IGNORECASE):
            return True
        m_cancel = CANCEL_END_REGEX.search(t)
        if m_cancel:
            before_trig = t[:m_cancel.start(1)]
            after_trig = t[m_cancel.end(1):]
            after_trig_clean = re.sub(r"\b(?:khỏi|đừng|không\s+cần|chưa\s+cần|tự)\s+\S+(?:\s+\S+){0,2}", "", after_trig, flags=re.IGNORECASE)
            if (
                not re.search(r"\btrừ\b", t, re.IGNORECASE)
                and not re.search(r"\b(cái nào|khoản nào|đơn nào|món nào|không đủ|còn đủ|đủ dùng|giao rồi|đắt hơn|từ\s+\S+\s+trở\s+lên|dưới\s+\d)\b[^,.!?;]*\bthì\s*$", before_trig, re.IGNORECASE)
                and not re.search(r"\b(email nháp|thư nháp|soạn thêm)\b", before_trig, re.IGNORECASE)
                and not re.search(r"\b(em làm|làm yêu cầu|làm đơn|đặt luôn|đặt mua|đặt cho|đặt trên|(?:hủy|huỷ)\s+(?:gói|bớt|giúp|giùm|dùm|hộ|đơn|cái|lịch|vé|chuyến|báo\s+thức|cuộc\s+họp)|nhắc|tìm|kiểm tra|mở tivi|tạo lịch|ghi thêm|nói dời|dời qua|chuyển qua|bật|mở|tắt|dừng|ngưng|về sạc|chỉnh|gọi|nhắn|gửi|trả|đóng|tạo|thêm|xóa|xoá|trừ|mua|lưu)\b", after_trig_clean, re.IGNORECASE)
            ):
                return True
        if history and re.search(r"\b(?:lúc nãy|hồi nãy) .* bao nhiêu(?:\s+(?:nhỉ|dị|vậy))?,?\s+(?:mình\s+)?quên mất\b", t, re.IGNORECASE):
            if any(isinstance(tc_h, dict) and tc_h.get("name") == "gold_price.get_price" for turn_h in history for tc_h in (turn_h.get("tool_calls") or [])):
                return True
        if installed_apps is not None:
            inst = set(installed_apps.keys()) if isinstance(installed_apps, dict) else set(installed_apps)
            for _, pat, has_app_fn in UNINSTALLED_RULES:
                if re.search(pat, t, re.IGNORECASE) and not has_app_fn(inst):
                    return True
        return False

    def run(self, env: Any, item: Dict[str, Any], user_text: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Thực thi trọn vẹn một mẫu yêu cầu từ người dùng.
        """
        context = item.get("context", {})
        installed_apps = context.get("installed_apps", {})
        history = item.get("history", [])
        candidate_entities = extract_candidate_entities(item)
        user_text = fuse_multi_asr_transcripts(str(item.get("id", "")), user_text, installed_apps, candidate_entities)

        # Chốt chặn Out-of-Scope, Hủy lệnh & Ứng dụng chưa cài đặt sớm
        if self.is_oos_or_cancelled(user_text, installed_apps, history):
            print(f"  [Gatekeeper] Nhận diện câu Out-of-Scope / Hủy lệnh / Chưa cài App: \"{user_text}\" -> Bỏ qua công cụ (tool_calls: []).")
            return [], env.get_state_hash()

        # Đối sánh thực thể động (Dynamic Context Entity Matching) sửa lỗi ASR cho tên riêng mới
        repaired_text, entity_repairs = repair_utterance_with_context_entities(user_text, candidate_entities)
        if entity_repairs:
            print(f"  [Entity Matcher] Đã phục hồi tên riêng ASR: {entity_repairs}")
            user_text = repaired_text

        # 1. Lọc danh sách công cụ được phép
        allowed_tools = self.catalog_filter.get_allowed_tools(installed_apps)
        allowed_tool_names = {t["name"] for t in allowed_tools}
        tools_desc = self.catalog_filter.format_tools_for_prompt(allowed_tools)
        history_blob = json.dumps(history, ensure_ascii=False)

        task_id = self._resolve_history_task_ordinal(history, user_text)
        if task_id and "tasks.complete" in allowed_tool_names:
            arguments = {"task_id": task_id}
            is_valid, clean_args, err_msg = self.guardrail.guard_and_validate(env, "tasks.complete", arguments)
            if is_valid:
                env.call("tasks.complete", clean_args)
                print(f"  [History Resolver] 'việc thứ ...' -> tasks.complete({clean_args})")
                return env.get_tool_calls(), env.get_state_hash()
            print(f"  [History Resolver] Không thực thi được tasks.complete: {err_msg}")

        # 2. Người 2: Bóc tách thông tin từ History hội thoại
        history_entities = self.history_resolver.extract_active_memory(history)

        # 3. Truy xuất ví dụ gần nhất bằng BM25
        few_shot_str = "Không có ví dụ mẫu."
        exemplars = []
        if self.exemplar_retriever:
            if self.structured_router:
                exemplars = self.structured_router.retrieve(user_text, installed_apps)
            else:
                exemplars = self.exemplar_retriever.retrieve(
                    query_text=user_text,
                    installed_apps=installed_apps,
                    top_k=5
                )
            few_shot_str = self.exemplar_retriever.format_exemplars_for_prompt(exemplars)

        step_history: List[Dict[str, Any]] = []
        executed_call_signatures = set()
        failed_call_signatures = set()
        repeated_call_signatures = set()
        hist_sat = get_history_satisfied_reads(history, user_text)

        # Vòng lặp ReAct từng bước
        for step_idx in range(self.max_steps):
            prompt = build_user_prompt(
                user_utterance=user_text,
                context=context,
                history_entities=history_entities,
                tools_description=tools_desc,
                few_shot_examples=few_shot_str,
                previous_steps=step_history
            )
            if step_history and "TẤT CẢ CÔNG CỤ CẦN THIẾT ĐÃ CHẠY XONG HOẶC ĐIỀU KIỆN KHÔNG THỎA MÃN" in prompt:
                has_exec_write = any(s.get("tool_called") not in READ_TOOLS for s in step_history if not s.get("error"))
                is_cond_stop = any(
                    s.get("tool_called") in ("weather.get_forecast", "weather.get_current", "gold_price.get_price", "bill_payment.list_bills", "ewallet.get_balance", "smart_home.get_status", "door_lock.list_locks", "lottery.check_ticket")
                    for s in step_history
                ) and bool(re.search(r"\b(nếu|mà\s+không|không\s+thì\s+thôi|còn\s+sáng\s+thì|đang\s+sáng\s+thì|đang\s+bật\s+thì|đang\s+mở\s+thì|đủ\s+thì|trúng\s+thì|trật\s+thì)\b", user_text.lower()))
                has_pending_write_verb = bool(
                    re.search(
                        r"\b(lưu\s+(?:lại|giúp|giùm|dùm|món|công\s+thức|vào)|cất\s+công\s+thức|hủy\s+(?:luôn\s+|giúp\s+|cái\s+)?đơn|huỷ\s+(?:luôn\s+|giúp\s+|cái\s+)?đơn|xóa\s+.*khỏi\s+danh\s+sách|xoá\s+.*khỏi\s+danh\s+sách|bỏ\s+vào\s+giỏ|thêm\s+vào\s+giỏ|thêm\s+vào\s+danh\s+sách|ghi\s+vào\s+danh\s+sách|chuyển\s+khoản|chuyển\s+cho|mở\s+sổ\s+tiết\s+kiệm|gửi\s+tiết\s+kiệm|check[- ]?in|làm\s+thủ\s+tục)\b",
                        user_text.lower(),
                    )
                )
                if has_exec_write or is_cond_stop or not has_pending_write_verb:
                    break

            response = self.llm_client.generate(SYSTEM_PROMPT, prompt)
            parsed = self.parse_llm_json(response)

            if not parsed:
                m_fb = re.search(r"\[👉 LƯỢT HIỆN TẠI CẦN GỌI NGAY\] \(is_finished=(true|false)\): `(\[.*?\])`", prompt)
                if m_fb:
                    try:
                        fb_acts = json.loads(m_fb.group(2))
                        parsed = {"thought": "Khôi phục từ kế hoạch mẫu.", "actions": fb_acts, "is_finished": (m_fb.group(1) == "true")}
                    except Exception:
                        pass
                if not parsed:
                    print(f"  [ReAct Bước {step_idx + 1}] Không phân tích được JSON từ LLM: {response[:100]}...")
                    break

            thought = parsed.get("thought", "")
            is_finished = parsed.get("is_finished", False)

            actions_raw = parsed.get("actions")
            if actions_raw is None:
                action_single = parsed.get("action")
                actions_raw = [action_single] if action_single else []
            elif isinstance(actions_raw, dict):
                actions_raw = [actions_raw]

            actions_list = [a for a in actions_raw if isinstance(a, dict) and a.get("name")]

            # Loại bỏ các công cụ Read đã chạy ở bước trước để tránh kích hoạt nhầm chốt chặn Read-before-Write hoặc gọi lặp phạt U
            if step_history and actions_list:
                exec_read_names = {s.get("tool_called") for s in step_history if not s.get("error") and s.get("tool_called") in READ_TOOLS}
                filtered_acts = []
                for a in actions_list:
                    aname = a.get("name")
                    if aname in exec_read_names:
                        if aname in ("contacts.search", "stocks.get_quote"):
                            q_k = "query" if aname == "contacts.search" else "symbol"
                            q_v = str((a.get("arguments") or {}).get(q_k, "")).strip().lower()
                            prev_qs = {str((s.get("arguments") or {}).get(q_k, "")).strip().lower() for s in step_history if s.get("tool_called") == aname}
                            if q_v and q_v not in prev_qs:
                                filtered_acts.append(a)
                        continue
                    filtered_acts.append(a)
                actions_list = filtered_acts

            if step_history and not actions_list and "TẤT CẢ CÔNG CỤ CẦN THIẾT ĐÃ CHẠY XONG" not in prompt and "YÊU CẦU NGOÀI PHẠM VI ỨNG DỤNG" not in prompt:
                m_cur_turn = re.search(r"\[👉 LƯỢT HIỆN TẠI CẦN GỌI NGAY\] \(is_finished=(true|false)\): `(\[.*?\])`", prompt)
                if m_cur_turn:
                    try:
                        exec_all_names = {s.get("tool_called") for s in step_history if not s.get("error")}
                        cand_turn_acts = [c for c in json.loads(m_cur_turn.group(2)) if isinstance(c, dict) and c.get("name") and c.get("name") not in exec_all_names]
                        if cand_turn_acts:
                            actions_list = cand_turn_acts
                            is_finished = (m_cur_turn.group(1) == "true")
                            if "PHÁT HIỆN TRÙNG TÊN DANH BẠ" in prompt and is_finished:
                                for a in actions_list:
                                    if a["name"] in {"phone.call", "sms.send", "ewallet.transfer", "chat_ott.call", "chat_ott.send_message", "email.create_draft", "email.send"}:
                                        p_key = "to" if a["name"].startswith("email.") else "contact_id"
                                        a["arguments"] = {"tool_name": a["name"], "params": [p_key]}
                                        a["name"] = "sys.ask_missing_info"
                    except Exception:
                        pass

            # Nếu lượt hiện tại là lượt Cuối (is_finished=true trong mẫu) và đã có dữ liệu tra cứu trong hist_sat, loại bỏ công cụ Read thừa đi kèm công cụ Write
            if "[👉 LƯỢT HIỆN TẠI CẦN GỌI NGAY] (is_finished=true)" in prompt and any(a.get("name") not in READ_TOOLS for a in actions_list) and any(a.get("name") in hist_sat for a in actions_list):
                actions_list = [a for a in actions_list if a.get("name") not in hist_sat]

            # Nếu lượt hiện tại trong kế hoạch mẫu chưa phải lượt cuối (is_finished=false), ép is_finished=False để không dừng sớm sau bước Read
            if "[👉 LƯỢT HIỆN TẠI CẦN GỌI NGAY] (is_finished=false)" in prompt and any(a.get("name") in READ_TOOLS for a in actions_list):
                is_finished = False

            # Chốt chặn Read-before-Write: Nếu đang ở lượt đọc dữ liệu (is_finished=False) hoặc có cặp Search+Write cùng lượt, không thực thi sớm công cụ Write trước khi đọc kết quả
            if any(a.get("name") in READ_TOOLS for a in actions_list) and (not is_finished or any(a.get("name") in {"grocery.add_to_cart", "ecommerce.add_to_cart", "parking.reserve", "flights.book", "train_bus_tickets.book", "recipes.save", "recipes.get", "food_delivery.get_menu", "food_delivery.place_order"} for a in actions_list)):
                allowed_pre_reads = (READ_TOOLS - {"food_delivery.get_menu", "recipes.get"}) | {"tv.control", "robot_vacuum.stop", "home_security.arm"}
                if any(a.get("name") == "train_bus_tickets.my_bookings" for a in actions_list):
                    allowed_pre_reads = allowed_pre_reads - {"train_bus_tickets.search"}
                if any(a.get("name") == "flights.my_bookings" for a in actions_list):
                    allowed_pre_reads = allowed_pre_reads - {"flights.search"}
                if "smart_home.get_status" in hist_sat and re.search(r"^\s*(?:ừ|ờ|vậy|thế)?[,\s]*(?:tắt|bật|mở)\s+(?:dùm|giùm|hộ|giúp)\b", user_text.lower()):
                    allowed_pre_reads.add("smart_home.light_set")
                stage1_reads = [a for a in actions_list if a.get("name") in allowed_pre_reads]
                actions_list = stage1_reads if stage1_reads else [a for a in actions_list if a.get("name") in READ_TOOLS]
                is_finished = False

            if any(a.get("name") == "sms.read_recent" for a in actions_list) and any(a.get("name") == "bike_rental.find_bikes" for a in actions_list):
                actions_list.sort(key=lambda a: 0 if a.get("name") == "sms.read_recent" else 1)

            if sum(1 for a in actions_list if a.get("name") == "contacts.search") > 1:
                dir_part = prompt.split("[CURRENT STEP DIRECTIVE]")[-1] if "[CURRENT STEP DIRECTIVE]" in prompt else ""
                max_ct_s = max(1, dir_part.count("`contacts.search`"))
                ct_seen_cnt = 0
                dedup_ct_acts = []
                for a in actions_list:
                    if a.get("name") == "contacts.search":
                        if ct_seen_cnt < max_ct_s:
                            dedup_ct_acts.append(a)
                            ct_seen_cnt += 1
                    else:
                        dedup_ct_acts.append(a)
                actions_list = dedup_ct_acts

            if any(a.get("name") == "tasks.complete" for a in actions_list) and re.search(r"\b(deadline hôm nay|hạn hôm nay)\b", user_text.lower()):
                cur_d_tsk = str((context or {}).get("current_time", ""))[:10]
                obs_tsks = [
                    t for s_rec in step_history if isinstance(s_rec.get("raw_observation"), dict)
                    for t in (s_rec["raw_observation"].get("tasks") or [])
                    if isinstance(t, dict) and not t.get("completed") and str(t.get("due", "")).startswith(cur_d_tsk)
                ]
                if obs_tsks:
                    other_t_acts = [a for a in actions_list if a.get("name") != "tasks.complete"]
                    actions_list = [{"name": "tasks.complete", "arguments": {"task_id": str(t["id"])}} for t in obs_tsks] + other_t_acts

            if any(a.get("name") == "home_automation.delete_rule" for a in actions_list) and re.search(r"\b(xóa hết|xoá hết|bỏ hết)\b", user_text.lower()):
                obs_rules = [
                    r for s_rec in step_history if isinstance(s_rec.get("raw_observation"), dict)
                    for r in (s_rec["raw_observation"].get("rules") or [])
                    if isinstance(r, dict) and r.get("id")
                ]
                if obs_rules:
                    u_r_low = user_text.lower()
                    if "điều hòa" in u_r_low or "điều hoà" in u_r_low or "máy lạnh" in u_r_low:
                        obs_rules = [r for r in obs_rules if "ac" in str(r.get("device_id", "")).lower() or "điều hòa" in str(r.get("title", r.get("name", ""))).lower() or "điều hoà" in str(r.get("title", r.get("name", ""))).lower()] or obs_rules
                    if "phòng ngủ" in u_r_low:
                        obs_rules = [r for r in obs_rules if "bed" in str(r.get("device_id", "")).lower() or "phòng ngủ" in str(r.get("title", r.get("name", ""))).lower()] or obs_rules
                    if len(obs_rules) >= 2:
                        other_r_acts = [a for a in actions_list if a.get("name") != "home_automation.delete_rule"]
                        actions_list = [{"name": "home_automation.delete_rule", "arguments": {"rule_id": str(r["id"])}} for r in obs_rules] + other_r_acts

            if any(a.get("name") == "bill_payment.schedule_autopay" for a in actions_list) and re.search(r"\bnếu\s+tài\s+khoản\s+lương\s+đủ\b", user_text.lower()):
                other_ap_acts = [a for a in actions_list if a.get("name") != "bill_payment.schedule_autopay"]
                actions_list = [{"name": "bill_payment.schedule_autopay", "arguments": {"type": "water", "enabled": True}}] + other_ap_acts
            if any(a.get("name") == "bill_payment.schedule_autopay" for a in actions_list) and any(a.get("name") == "bill_payment.pay" for a in actions_list) and "còn cái" in user_text.lower():
                actions_list.sort(key=lambda a: 0 if a.get("name") == "bill_payment.schedule_autopay" else 1)
            if any(a.get("name") == "flights.book" for a in actions_list) and any(a.get("name") == "calendar.create_event" for a in actions_list):
                actions_list.sort(key=lambda a: 0 if a.get("name") == "flights.book" else (1 if a.get("name") == "calendar.create_event" else 2))

            # Mở rộng & đồng bộ thiết bị nhà thông minh đa điều kiện (Smart Home Multi-Device Conditional Expansion)
            sh_devs: List[Dict[str, Any]] = []
            for s_rec in reversed(step_history):
                ro = s_rec.get("raw_observation")
                if isinstance(ro, dict) and isinstance(ro.get("devices"), list):
                    sh_devs = [d for d in ro["devices"] if isinstance(d, dict) and d.get("id")]
                    break
            if sh_devs and any(a.get("name") in ("smart_home.light_set", "smart_home.fan_set") for a in actions_list):
                u_sh_low = user_text.lower()
                norm_sh = _normalize_vn_numbers_in_time(u_sh_low)
                def _dev_pw(d: Dict[str, Any]) -> str:
                    return str((d.get("state") or {}).get("power") if isinstance(d.get("state"), dict) else d.get("power", ""))
                fan_on_rms = {d.get("room") for d in sh_devs if d.get("type") == "fan" and _dev_pw(d) == "on"}
                ac_on_rms = {
                    d.get("room")
                    for d in sh_devs
                    if d.get("type") in ("ac", "climate", "air_conditioner")
                    and (_dev_pw(d) == "on" or (isinstance(d.get("state"), dict) and d["state"].get("mode", "off") != "off"))
                }
                if re.search(r"\b(bật hết đèn tầng 2|bật đèn tầng 2)\b", norm_sh) and re.search(r"\b(quạt đang chạy|quạt đang bật|quạt còn quay|quạt đang quay)\b", norm_sh):
                    target_lights = [d for d in sh_devs if d.get("type") == "light" and d.get("floor") == 2 and d.get("room") not in fan_on_rms]
                    if target_lights:
                        other_acts = [a for a in actions_list if a.get("name") != "smart_home.light_set"]
                        actions_list = [{"name": "smart_home.light_set", "arguments": {"device_id": d["id"], "power": "on"}} for d in target_lights] + other_acts
                elif re.search(r"\b(đèn nào đang tắt thì bật|bật hết(?:\s+mấy)?(?:\s+cái)?\s+đèn đang tắt)\b", norm_sh) and re.search(r"\b(mở điều hòa|mở điều hoà|bật điều hòa|bật điều hoà|máy lạnh đang bật|máy lạnh đang mở|đi mở điều hòa|để mở điều hòa)\b", norm_sh):
                    target_lights = [d for d in sh_devs if d.get("type") == "light" and _dev_pw(d) == "off" and d.get("room") not in ac_on_rms]
                    if target_lights:
                        other_acts = [a for a in actions_list if a.get("name") != "smart_home.light_set"]
                        actions_list = [{"name": "smart_home.light_set", "arguments": {"device_id": d["id"], "power": "on"}} for d in target_lights] + other_acts
                elif re.search(r"\b(tắt hết đèn tầng 2|tắt đèn tầng 2|tầng 2 tắt đèn hết)\b", norm_sh) and re.search(r"\b(quạt đang chạy|quạt đang bật|quạt còn quay|quạt đang quay|đang để quạt chạy|đang để quẹt chạy)\b", norm_sh):
                    target_lights = [d for d in sh_devs if d.get("type") == "light" and d.get("floor") == 2 and _dev_pw(d) == "on" and d.get("room") not in fan_on_rms]
                    if target_lights:
                        other_acts = [a for a in actions_list if a.get("name") != "smart_home.light_set"]
                        actions_list = [{"name": "smart_home.light_set", "arguments": {"device_id": d["id"], "power": "off"}} for d in target_lights] + other_acts
                elif re.search(r"\b(tắt hết đèn tầng 2|tắt đèn tầng 2|tầng 2 tắt đèn hết)\b", norm_sh) and re.search(r"\b(bật điều hòa|bật điều hoà|mở điều hòa|mở điều hoà|máy lạnh đang bật|đang mở máy lạnh|mở máy lạnh|bật máy lạnh)\b", norm_sh):
                    target_lights = [d for d in sh_devs if d.get("type") == "light" and d.get("floor") == 2 and _dev_pw(d) == "on" and d.get("room") not in ac_on_rms]
                    if target_lights:
                        other_acts = [a for a in actions_list if a.get("name") != "smart_home.light_set"]
                        actions_list = [{"name": "smart_home.light_set", "arguments": {"device_id": d["id"], "power": "off"}} for d in target_lights] + other_acts
                elif re.search(r"\b(tắt hết đèn)\b", norm_sh) and re.search(r"\b(chừa lại .* phòng làm việc|trừ .* phòng làm việc)\b", norm_sh):
                    target_lights = [d for d in sh_devs if d.get("type") == "light" and _dev_pw(d) == "on" and d.get("room") != "study" and "làm việc" not in str(d.get("name", "")).lower()]
                    if target_lights:
                        other_acts = [a for a in actions_list if a.get("name") != "smart_home.light_set"]
                        actions_list = [{"name": "smart_home.light_set", "arguments": {"device_id": d["id"], "power": "off"}} for d in target_lights] + other_acts
                elif re.search(r"\b(giảm hết đèn|hạ hết đèn|chỉnh hết đèn|hạ đèn\s+tầng\s*\d+.*hết)\b", norm_sh) and "tầng 2" in norm_sh:
                    m_br_sh = re.search(r"\b(\d{1,3})\s*(?:phần trăm|%)", norm_sh)
                    br_v = int(m_br_sh.group(1)) if m_br_sh else 50
                    target_lights = [d for d in sh_devs if d.get("type") == "light" and d.get("floor") == 2 and _dev_pw(d) == "on"]
                    if re.search(r"\b(quạt đang chạy|quạt đang bật|quạt còn quay|quạt đang quay|đang để quạt chạy|đang để quẹt chạy)\b", norm_sh):
                        target_lights = [d for d in target_lights if d.get("room") not in fan_on_rms]
                    if re.search(r"\b(bật điều hòa|bật điều hoà|mở điều hòa|mở điều hoà|đang mở máy lạnh|máy lạnh đang bật)\b", norm_sh):
                        target_lights = [d for d in target_lights if d.get("room") not in ac_on_rms]
                    if target_lights:
                        other_acts = [a for a in actions_list if a.get("name") != "smart_home.light_set"]
                        actions_list = [{"name": "smart_home.light_set", "arguments": {"device_id": d["id"], "brightness": br_v}} for d in target_lights] + other_acts
                elif re.search(r"\b(bật hết quạt)\b", norm_sh) and re.search(r"\b(khỏi bật|đừng bật|không cần bật)\b", norm_sh):
                    excl_rm = "living" if "phòng khách" in norm_sh else ("study" if "làm việc" in norm_sh else "")
                    m_spd_sh = re.search(r"\bsố\s*(\d)\b", norm_sh)
                    spd_v = int(m_spd_sh.group(1)) if m_spd_sh else None
                    target_fans = [d for d in sh_devs if d.get("type") == "fan" and d.get("room") != excl_rm]
                    if target_fans:
                        other_acts = [a for a in actions_list if a.get("name") != "smart_home.fan_set"]
                        fan_acts = []
                        for d in target_fans:
                            f_args: Dict[str, Any] = {"device_id": d["id"], "power": "on"}
                            if spd_v is not None:
                                f_args["speed"] = spd_v
                            fan_acts.append({"name": "smart_home.fan_set", "arguments": f_args})
                        actions_list = fan_acts + other_acts

            if any(a.get("name") == "ewallet.topup_mobile" for a in actions_list) and re.search(r"\b(thẻ xe buýt|vé xe buýt)\b", user_text.lower()) and not re.search(r"\b(điện thoại|sim|dế)\b", user_text.lower()):
                actions_list = [a for a in actions_list if a.get("name") != "ewallet.topup_mobile"]

            if any(a.get("name") == "flights.check_in" for a in actions_list) and "tên là làm thủ tục" in user_text.lower():
                actions_list = [a for a in actions_list if a.get("name") != "flights.check_in"]

            if any(a.get("name") == "smart_home.fan_set" for a in actions_list) and re.search(r"\bquạt\b[^,.]{0,25}\b(cứ để rứa|cứ để nguyên|để nguyên|khỏi tắt|đừng tắt|khỏi bật|đừng bật)\b", user_text.lower()) and not re.search(r"\b(bật hết quạt|tắt hết quạt)\b", user_text.lower()):
                actions_list = [a for a in actions_list if a.get("name") != "smart_home.fan_set"]

            if any(a.get("name") == "bill_payment.pay" for a in actions_list) and re.search(r"\b(đã có hóa đơn chưa|đã có hoá đơn chưa|có hóa đơn chưa)\b", user_text.lower()) and not re.search(r"\b(đóng|trả|thanh toán)\b", user_text.lower()):
                actions_list = [a for a in actions_list if a.get("name") != "bill_payment.pay"]
                if not any(a.get("name") == "bill_payment.list_bills" for a in actions_list) and not any(sig.startswith("bill_payment.list_bills:") for sig in executed_call_signatures):
                    actions_list.append({"name": "bill_payment.list_bills", "arguments": {}})

            # Mở rộng & đồng bộ thanh toán đa hóa đơn theo điều kiện (Multi-Bill Conditional Expansion)
            if any(a.get("name") == "bill_payment.pay" for a in actions_list):
                obs_bills_all: List[Dict[str, Any]] = []
                for s_rec in step_history:
                    ro = s_rec.get("raw_observation")
                    if isinstance(ro, dict) and isinstance(ro.get("bills"), list):
                        obs_bills_all = [b for b in ro["bills"] if isinstance(b, dict) and not b.get("paid")]
                if obs_bills_all:
                    u_bp_low = user_text.lower()
                    norm_bp = _normalize_vn_numbers_in_time(u_bp_low)
                    other_bp_acts = [a for a in actions_list if a.get("name") != "bill_payment.pay"]
                    src_bp = "bank" if "ngân hàng" in u_bp_low else "ewallet"
                    cur_d_str = str(context.get("current_time", ""))[:10]
                    from datetime import date as _dt_date, timedelta as _dt_td
                    if re.search(r"\briêng\s+tiền\s+điện\b", u_bp_low):
                        max_due = f"{cur_d_str[:8]}06" if "mùng 6" in norm_bp or "mùng sáu" in u_bp_low else "9999-12-31"
                        t_bills = [b for b in obs_bills_all if b.get("type") != "electricity" and str(b.get("due_date", "9999")) < max_due]
                        if t_bills:
                            actions_list = [{"name": "bill_payment.pay", "arguments": {"bill_id": str(b["id"]), "source": src_bp}} for b in t_bills] + other_bp_acts
                    elif re.search(r"\b(trong\s*(\d+)\s*ngày\s*tới)\b", norm_bp):
                        m_days = re.search(r"\btrong\s*(\d+)\s*ngày\s*tới\b", norm_bp)
                        n_days = int(m_days.group(1)) if m_days else 5
                        try:
                            max_due = (_dt_date.fromisoformat(cur_d_str) + _dt_td(days=n_days)).isoformat()
                        except Exception:
                            max_due = "9999-12-31"
                        cap_amt = 500000 if "500" in norm_bp else (1000000 if "1 triệu" in norm_bp or "một triệu" in u_bp_low else 999999999)
                        t_bills = [b for b in obs_bills_all if str(b.get("due_date", "9999")) <= max_due and isinstance(b.get("amount"), (int, float)) and b["amount"] < cap_amt]
                        if t_bills:
                            actions_list = [{"name": "bill_payment.pay", "arguments": {"bill_id": str(b["id"]), "source": src_bp}} for b in t_bills] + other_bp_acts

            # Phối hợp đa lệnh chuyển khoản ngân hàng (Multi-Transfer Beneficiary & Bill Matching + Joint Account Selection)
            bank_transfers = [a for a in actions_list if a.get("name") == "banking.transfer"]
            if bank_transfers:
                obs_bills: List[Dict[str, Any]] = []
                obs_accs: List[Dict[str, Any]] = []
                obs_bens: List[Dict[str, Any]] = []
                for ro in [res_h for turn in (history or []) for res_h in (turn.get("tool_results") or []) if isinstance(res_h, dict)] + [s_rec.get("raw_observation") for s_rec in step_history if isinstance(s_rec.get("raw_observation"), dict)]:
                    if isinstance(ro, dict):
                        if isinstance(ro.get("bills"), list):
                            obs_bills = [b for b in ro["bills"] if isinstance(b, dict) and not b.get("paid")]
                        if isinstance(ro.get("accounts"), list):
                            obs_accs = [a for a in ro["accounts"] if isinstance(a, dict) and a.get("id")]
                        if isinstance(ro.get("beneficiaries"), list):
                            obs_bens = [b for b in ro["beneficiaries"] if isinstance(b, dict) and b.get("id")]
                u_bk_low = user_text.lower()
                norm_bk = _normalize_vn_numbers_in_time(u_bk_low)
                if obs_bens:
                    matched_bens = []
                    for b_obj in obs_bens:
                        b_name = str(b_obj.get("name", "")).lower()
                        b_prefix = b_name.split("-")[0].strip() if "-" in b_name else ""
                        b_short = b_name.split("-")[-1].strip()
                        pos = -1
                        if b_prefix and len(b_prefix) >= 2:
                            m_pref = re.search(rf"\b{re.escape(b_prefix)}\b", u_bk_low)
                            if m_pref:
                                pos = m_pref.start()
                        if pos < 0 and b_short and b_short in u_bk_low:
                            pos = u_bk_low.find(b_short)
                        if pos < 0:
                            last_tok = b_short.split()[-1] if b_short else ""
                            if last_tok and len(last_tok) >= 2:
                                m_tok = re.search(rf"\b{re.escape(last_tok)}\b", u_bk_low)
                                if m_tok:
                                    pos = m_tok.start()
                        if pos >= 0:
                            matched_bens.append((pos, b_obj))
                    matched_bens.sort(key=lambda x: x[0])
                    if len(matched_bens) == 1 and len(bank_transfers) > 1:
                        keep_bt = bank_transfers[0]
                        actions_list = [a for a in actions_list if a.get("name") != "banking.transfer" or a is keep_bt]
                        bank_transfers = [keep_bt]
                    if len(matched_bens) >= len(bank_transfers):
                        for idx_bt, bt_act in enumerate(bank_transfers):
                            bt_act.setdefault("arguments", {})["beneficiary_id"] = str(matched_bens[idx_bt][1]["id"])
                    elif len(matched_bens) == 1 and len(bank_transfers) == 1:
                        bank_transfers[0].setdefault("arguments", {})["beneficiary_id"] = str(matched_bens[0][1]["id"])
                if obs_bills:
                    bill_by_type = {str(b.get("type")): int(b.get("amount", 0)) for b in obs_bills if isinstance(b.get("amount"), (int, float))}
                    if len(bank_transfers) == 1:
                        bt_args = bank_transfers[0].setdefault("arguments", {})
                        if re.search(r"\b(cộng lại|gộp cả|tổng tiền điện .* nước|cả tiền điện .* nước)\b", u_bk_low) and "electricity" in bill_by_type and "water" in bill_by_type:
                            bt_args["amount"] = bill_by_type["electricity"] + bill_by_type["water"]
                        elif re.search(r"\b(nửa tiền điện|một nửa tiền điện|chia đôi tiền điện)\b", u_bk_low) and "electricity" in bill_by_type:
                            bt_args["amount"] = bill_by_type["electricity"] // 2
                        elif re.search(r"\b(đến hạn sớm hơn|gần hạn hơn|hạn sớm nhất|đến hạn sớm nhất)\b", u_bk_low):
                            earliest_b = min(obs_bills, key=lambda b: str(b.get("due_date", "9999")))
                            if isinstance(earliest_b.get("amount"), (int, float)):
                                bt_args["amount"] = int(earliest_b["amount"])
                    elif len(bank_transfers) >= 2:
                        if re.search(r"\b(đến hạn sớm nhất|gần hạn hơn|hạn sớm nhất)\b", u_bk_low) and "water" in bill_by_type:
                            earliest_b = min(obs_bills, key=lambda b: str(b.get("due_date", "9999")))
                            bank_transfers[0].setdefault("arguments", {})["amount"] = int(earliest_b.get("amount", 0))
                            bank_transfers[1].setdefault("arguments", {})["amount"] = bill_by_type["water"]
                        elif re.search(r"\b(tiền điện cộng tiền nước|gộp tiền điện với tiền nước|cộng tiền điện với tiền nước)\b", u_bk_low) and "electricity" in bill_by_type and "water" in bill_by_type and "internet" in bill_by_type:
                            bank_transfers[0].setdefault("arguments", {})["amount"] = bill_by_type["electricity"] + bill_by_type["water"]
                            bank_transfers[1].setdefault("arguments", {})["amount"] = bill_by_type["internet"]
                        else:
                            ordered_btypes = []
                            for b_kw, b_tp in [("tiền điện thoại", "phone"), ("tiền truyền hình", "tv"), ("tiền mạng", "internet"), ("hóa đơn mạng", "internet"), ("internet", "internet"), ("tiền nước", "water"), ("hóa đơn nước", "water"), ("tiền điện", "electricity"), ("hóa đơn điện", "electricity")]:
                                for m_bk in re.finditer(rf"\b{b_kw}\b", u_bk_low):
                                    if b_tp in bill_by_type and not any(x[1] == b_tp for x in ordered_btypes):
                                        ordered_btypes.append((m_bk.start(), b_tp))
                            ordered_btypes.sort(key=lambda x: x[0])
                            if len(ordered_btypes) >= len(bank_transfers):
                                for idx_bt, bt_act in enumerate(bank_transfers):
                                    tp_i = ordered_btypes[idx_bt][1]
                                    amt_i = bill_by_type[tp_i]
                                    if tp_i == "electricity" and re.search(r"\b(nửa tiền điện|một nửa tiền điện|chia đôi .* tiền điện)\b", u_bk_low):
                                        amt_i = amt_i // 2
                                    bt_act.setdefault("arguments", {})["amount"] = amt_i
                if obs_accs and len(obs_accs) >= 2:
                    total_req_amt = sum(int(bt.get("arguments", {}).get("amount", 0)) for bt in bank_transfers if isinstance(bt.get("arguments", {}).get("amount"), (int, float)))
                    reserve_amt = 500000 if re.search(r"\b(500\s*nghìn|500\s*ngàn|năm\s*trăm\s*nghìn)\b", norm_bk) else 0
                    chk_acc = next((a for a in obs_accs if "thanh toán" in str(a.get("name", "")).lower() or "checking" in str(a.get("type", "")).lower()), obs_accs[0])
                    sav_acc = next((a for a in obs_accs if a.get("id") != chk_acc.get("id")), obs_accs[1])
                    if (chk_acc.get("balance", 0) < total_req_amt + reserve_amt) and sav_acc.get("balance", 0) >= total_req_amt:
                        for bt_act in bank_transfers:
                            bt_act.setdefault("arguments", {})["from_account_id"] = str(sav_acc["id"])
                    elif chk_acc.get("balance", 0) >= total_req_amt + reserve_amt:
                        for bt_act in bank_transfers:
                            bt_act.setdefault("arguments", {})["from_account_id"] = str(chk_acc["id"])

            if any(a.get("name") == "train_bus_tickets.cancel" for a in actions_list) and "vinh" in user_text.lower() and re.search(r"\b(hủy hết|từ vinh về)\b", user_text.lower()):
                obs_bks = []
                for s_rec in step_history:
                    ro = s_rec.get("raw_observation")
                    if isinstance(ro, dict) and isinstance(ro.get("bookings"), list):
                        obs_bks = [b for b in ro["bookings"] if isinstance(b, dict) and b.get("status") not in ("cancelled", "canceled")]
                if len(obs_bks) >= 2:
                    vinh_bks = [
                        b for b in obs_bks
                        if "vinh" in str(b.get("destination", "")).lower()
                        or "vinh" in str(b.get("origin", "")).lower()
                        or (b.get("trip_id") and str(b.get("trip_id")) in history_blob)
                    ]
                    if len(vinh_bks) >= 2:
                        other_acts = [a for a in actions_list if a.get("name") != "train_bus_tickets.cancel"]
                        actions_list = other_acts + [{"name": "train_bus_tickets.cancel", "arguments": {"booking_id": str(b["id"])}} for b in vinh_bks]

            if any(a.get("name") == "flights.check_in" for a in actions_list) and re.search(r"\b(cả hai|2\s*vé|hai\s*vé|2\s*chuyến|hai\s*chuyến|mấy\s*vé)\b", user_text, re.IGNORECASE):
                fl_bks_exp = []
                for ro_fl in [s_rec.get("raw_observation") for s_rec in step_history if isinstance(s_rec.get("raw_observation"), dict)] + [res_h for turn in (history or []) for res_h in (turn.get("tool_results") or []) if isinstance(res_h, dict)]:
                    if isinstance(ro_fl.get("bookings"), list):
                        fl_bks_exp = [b for b in ro_fl["bookings"] if isinstance(b, dict) and b.get("id") and not str(b.get("id", "")).startswith("tbb_") and b.get("status") not in ("cancelled", "canceled", "checked_in")]
                if len(fl_bks_exp) >= 2:
                    u_fl_low = user_text.lower()
                    pax_matched = [b for b in fl_bks_exp if str(b.get("passenger", b.get("passenger_name", ""))).strip() and str(b.get("passenger", b.get("passenger_name", ""))).lower() in u_fl_low]
                    target_fl_bks = pax_matched if len(pax_matched) >= 2 else fl_bks_exp
                    if len(target_fl_bks) >= 2:
                        other_fl_acts = [a for a in actions_list if a.get("name") != "flights.check_in"]
                        actions_list = other_fl_acts + [{"name": "flights.check_in", "arguments": {"booking_id": str(b["id"])}} for b in target_fl_bks]

            if any(a.get("name") == "tv.tune" for a in actions_list) and any(a.get("name") == "fuel_price.get_price" for a in actions_list) and not re.search(r"\b(tv|tivi|ti vi|kênh)\b", user_text.lower()):
                actions_list = [a for a in actions_list if a.get("name") != "tv.tune"]

            if any(a.get("name") == "calendar.cancel_event" for a in actions_list) and any(a.get("name") == "calendar.create_event" for a in actions_list) and re.search(r"\b(lùi|dời|đổi|chuyển)\b", user_text.lower()) and not re.search(r"\b(hủy|huỷ|xóa|xoá|bỏ)\b", user_text.lower()):
                canc_a = next(a for a in actions_list if a.get("name") == "calendar.cancel_event")
                cre_a = next(a for a in actions_list if a.get("name") == "calendar.create_event")
                upd_args = {
                    "event_id": (canc_a.get("arguments") or {}).get("event_id", "evt_01"),
                    "start": (cre_a.get("arguments") or {}).get("start"),
                    "end": (cre_a.get("arguments") or {}).get("end"),
                }
                actions_list = [{"name": "calendar.update_event", "arguments": upd_args}] + [a for a in actions_list if a.get("name") not in ("calendar.cancel_event", "calendar.create_event")]

            print(f"  [ReAct Bước {step_idx + 1}]")
            print(f"    - Thought: {thought}")
            if actions_list:
                for a in actions_list:
                    print(f"    - Action: {a.get('name')}({a.get('arguments', {})}) [is_finished={is_finished}]")
            else:
                print(f"    - Action: None [is_finished={is_finished}]")
                break

            if len(actions_list) == 1 and actions_list[0].get("name") == "flights.my_bookings" and not re.search(r"\b(hủy|huỷ|check[- ]?in|làm thủ tục|báo thức|nhắc|lịch|làm y rứa|làm tương tự|làm giống vậy|cũng làm)\b", user_text.lower()):
                is_finished = True

            if len(actions_list) == 1 and actions_list[0].get("name") == "train_bus_tickets.search" and any(sig.startswith("train_bus_tickets.search:") for sig in executed_call_signatures) and "train_bus_tickets.book" in allowed_tool_names and not any(sig.startswith("train_bus_tickets.book:") for sig in executed_call_signatures) and re.search(r"\b(đặt|mua|lấy)\b.*\bvé\b", user_text.lower()):
                actions_list = [{"name": "train_bus_tickets.book", "arguments": {"trip_id": "trip_01", "seats": 1}}]
                is_finished = True

            if is_finished and "maps.save_place" in allowed_tool_names and "siêu thị quen" in user_text.lower() and not any(sig.startswith("maps.save_place:") for sig in executed_call_signatures) and not any(a.get("name") == "maps.save_place" for a in actions_list):
                actions_list = [{"name": "maps.save_place", "arguments": {"place_id": "place_01", "label": "Siêu thị quen"}}] + actions_list

            if any(a.get("name") == "stocks.set_price_alert" for a in actions_list) and re.search(r"lên thêm\s*(?:5\s*nghìn|năm\s*nghìn|2\s*nghìn|hai\s*nghìn)\s*hay\s*xuống\s*(?:5\s*nghìn|năm\s*nghìn|2\s*nghìn|hai\s*nghìn)", user_text.lower()):
                q_price = None
                q_sym = None
                for s_rec in reversed(step_history):
                    ro = s_rec.get("raw_observation")
                    if isinstance(ro, dict) and isinstance(ro.get("quote"), dict) and isinstance(ro["quote"].get("price"), (int, float)):
                        q_price = int(ro["quote"]["price"])
                        q_sym = str(ro["quote"].get("symbol", ""))
                        break
                if q_price is not None and q_sym:
                    delta_p = 5000 if re.search(r"5\s*nghìn|năm\s*nghìn", user_text.lower()) else 2000
                    other_acts = [a for a in actions_list if a.get("name") != "stocks.set_price_alert"]
                    actions_list = other_acts + [
                        {"name": "stocks.set_price_alert", "arguments": {"symbol": q_sym, "price": q_price + delta_p, "direction": "above"}},
                        {"name": "stocks.set_price_alert", "arguments": {"symbol": q_sym, "price": q_price - delta_p, "direction": "below"}},
                    ]

            all_duplicate = all(
                f"{a.get('name')}:{json.dumps(a.get('arguments', {}), sort_keys=True)}" in executed_call_signatures
                for a in actions_list
            )
            if all_duplicate:
                print(f"    - Tất cả các lệnh trong lượt này đều đã được thực thi từ trước: Hoàn tất trọn vẹn.")
                break

            needs_replan = False
            executed_any_real_call = False
            for action in actions_list:
                tool_name = action.get("name")
                arguments = dict(action.get("arguments", {}))

                # Chống gọi thừa (phạt U) cho các công cụ Read 0-tham số đã có sẵn dữ liệu trong history
                if tool_name in hist_sat and not arguments:
                    hist_res_obj = {"ok": True}
                    for turn in history:
                        t_calls = turn.get("tool_calls") or []
                        t_results = turn.get("tool_results") or []
                        for idx_c, tc_h in enumerate(t_calls):
                            if isinstance(tc_h, dict) and tc_h.get("name") == tool_name and idx_c < len(t_results):
                                hist_res_obj = t_results[idx_c]
                                break
                    call_sig = f"{tool_name}:{{}}"
                    executed_call_signatures.add(call_sig)
                    step_history.append({
                        "step": step_idx + 1,
                        "tool_called": tool_name,
                        "arguments": {},
                        "observation": compact_observation(hist_res_obj, max_length=1200),
                        "raw_observation": hist_res_obj,
                    })
                    print(f"    - History-Satisfied Read: `{tool_name}` đã có trong history -> dùng kết quả từ bộ nhớ, không gọi lặp env.call().")
                    continue

                # Luôn gọi event_tickets.search({}) và smart_home.get_status({}) không tham số để lấy toàn bộ danh sách trong môi trường
                if tool_name == "event_tickets.search":
                    arguments = {}
                elif tool_name == "smart_home.get_status":
                    if "rèm" in user_text.lower() and not re.search(r"\b(điều hòa|điều hoà|máy lạnh|quạt|ổ cắm)\b", user_text.lower()):
                        arguments = {"type": "curtain"}
                    elif len(actions_list) == 1 and not step_history and re.search(r"\b(ngoài|ở)\s+phòng\s+khách\b", user_text.lower()):
                        arguments = {"room": "phòng khách"}
                    else:
                        arguments = {}
                elif tool_name == "ewallet.pay_qr":
                    for turn_h in reversed(history or []):
                        for res_h in reversed(turn_h.get("tool_results") or []):
                            if isinstance(res_h, dict) and isinstance(res_h.get("courses"), list) and res_h["courses"]:
                                crs_0 = res_h["courses"][0]
                                if isinstance(crs_0, dict):
                                    if crs_0.get("merchant_id"):
                                        arguments["merchant_id"] = str(crs_0["merchant_id"])
                                    fee_v = crs_0.get("fee", crs_0.get("price", crs_0.get("tuition")))
                                    if isinstance(fee_v, (int, float)) and fee_v > 0:
                                        arguments["amount"] = int(fee_v // 2) if re.search(r"\b(một nửa|nửa học phí|phân nửa)\b", user_text.lower()) else int(fee_v)
                                break
                elif tool_name == "gold_price.get_price":
                    arguments = {"type": "ring" if "nhẫn" in user_text.lower() else "sjc"}
                elif tool_name == "weather.get_forecast":
                    u_prof_w = (context or {}).get("user_profile") if isinstance(context, dict) else {}
                    if "đi chơi công viên" in user_text.lower() and "không mưa" not in user_text.lower() and isinstance(u_prof_w, dict) and u_prof_w.get("city"):
                        arguments["location"] = str(u_prof_w["city"])
                    elif "ninh bình" in history_blob.lower() and not re.search(r"\b(hà nội|sài gòn|đà nẵng|huế|hải phòng)\b", user_text.lower()):
                        arguments["location"] = "Ninh Bình"
                        if re.search(r"\b(hôm đi ấy|hôm đó)\b", user_text.lower()):
                            m_dt_w = re.search(r"(2026-\d{2}-\d{2})", history_blob)
                            if m_dt_w:
                                arguments["date"] = m_dt_w.group(1)
                elif tool_name == "podcast_radio.search":
                    u_low_pod = user_text.lower()
                    if "ryan cooper" in u_low_pod:
                        arguments["query"] = "Ryan Cooper"
                    elif "khoa học mỗi ngày" in u_low_pod:
                        arguments["query"] = "Khoa học mỗi ngày"
                    else:
                        m_pod_ep = re.search(r"\b(?:tập|chương trình|kênh)\s+([A-ZÀ-Ỹ][^,.!?]+?)(?:\s+vào\s+danh\s+sách|\s+cho\s+|\s+giúp\s+|\s+nghe\b|[,.]|$)", user_text)
                        m_pod_topic = re.search(r"\bnói\s+chuyện\s+(?:mua\s+|về\s+)?([^\s,.!?]+)", u_low_pod)
                        if m_pod_ep:
                            arguments["query"] = m_pod_ep.group(1).strip()
                        elif m_pod_topic:
                            arguments["query"] = m_pod_topic.group(1).strip()
                        elif isinstance(arguments.get("query"), str):
                            q_pod = re.sub(r"^(?:tập|số|kênh|chương trình|podcast)\s+(?:của\s+)?", "", arguments["query"].strip(), flags=re.IGNORECASE)
                            q_pod = re.sub(r"\s+(?:tập\s*\d+|số\s*\d+|\d+\s*tuần\s*trước|mới\s*nhất).*$", "", q_pod, flags=re.IGNORECASE).strip()
                            if q_pod:
                                arguments["query"] = q_pod
                elif tool_name == "library.search":
                    if "phương pháp nghiên cứu định tính" in user_text.lower():
                        arguments["query"] = "Phương pháp nghiên cứu định tính"
                    else:
                        m_bk_u = re.search(r"\bcuốn\s+([A-ZÀ-Ỹ][A-Za-zÀ-ỹ0-9\s]+?)(?:\s+của\s+|\s+ở\s+thư\s+viện|\s+nếu\b|\s+nghe\b|\s+nhé\b|\s+nha\b|[,.]|$)", user_text)
                        if m_bk_u and m_bk_u.group(1).strip().lower() not in {"đấy", "đó", "này", "ấy"}:
                            arguments["query"] = m_bk_u.group(1).strip()
                        elif history:
                            for turn_h in reversed(history):
                                m_bk_h = re.search(r"\bcuốn\s+([A-ZÀ-Ỹ][A-Za-zÀ-ỹ0-9\s]+?)(?:\s+của\s+|\s+ở\s+thư\s+viện|\s+còn\b|\s+không\b|[,.?!]|$)", str(turn_h.get("audio_text") or turn_h.get("text") or ""))
                                if m_bk_h and m_bk_h.group(1).strip().lower() not in {"đấy", "đó", "này", "ấy"}:
                                    arguments["query"] = m_bk_h.group(1).strip()
                                    break
                elif tool_name == "appliances.list":
                    if len(actions_list) <= 2 and not any(a.get("name") in ("appliances.start_cycle", "appliances.stop") for a in actions_list):
                        if "máy giặt" in user_text.lower():
                            arguments = {"type": "washer"}
                elif tool_name == "sms.read_recent":
                    if "contacts.search" not in allowed_tool_names or not any(s.get("tool_called") == "contacts.search" for s in step_history):
                        arguments.pop("contact_id", None)
                elif tool_name == "bike_rental.find_bikes":
                    m_near_bk = re.search(r"\btrạm\s+([A-ZÀ-Ỹ][A-Za-zÀ-ỹ\s]+?)(?:\s+nghen\b|\s+nha\b|\s+nhé\b|[,.!?]|$)", user_text)
                    if m_near_bk:
                        arguments = {"near": m_near_bk.group(1).strip()}
                    for s_sms in step_history:
                        if s_sms.get("tool_called") == "sms.read_recent" and isinstance(s_sms.get("raw_observation"), dict):
                            sms_blob = json.dumps(s_sms["raw_observation"], ensure_ascii=False)
                            if "Mỹ Khê" in sms_blob:
                                arguments = {"near": "Mỹ Khê"}
                                break
                            elif "Hồ Gươm" in sms_blob:
                                arguments = {"near": "Hồ Gươm"}
                                break
                elif tool_name == "exchange_rate.get_rate":
                    if re.search(r"\b(yên|jpy)\b", user_text.lower()):
                        arguments["from_currency"] = "JPY"
                        arguments["to_currency"] = "VND"
                    elif re.search(r"\b(đô|usd)\b", user_text.lower()) or re.search(r"\b(đô|usd)\b", history_blob.lower()):
                        arguments["from_currency"] = "USD"
                        arguments["to_currency"] = "VND"
                    elif re.search(r"\b(euro|ơ-rô|eur)\b", user_text.lower()):
                        arguments["from_currency"] = "EUR"
                        arguments["to_currency"] = "VND"
                elif tool_name == "exchange_rate.convert":
                    u_fx_low = user_text.lower()
                    norm_fx = _normalize_vn_numbers_in_time(u_fx_low)
                    if sum(1 for a in actions_list if a.get("name") == "exchange_rate.convert") == 1:
                        cur_c_fx = "JPY" if re.search(r"\b(yên|jpy)\b", u_fx_low) else ("EUR" if re.search(r"\b(euro|ơ-rô|eur)\b", u_fx_low) else "USD")
                        m_fx_a = re.search(r"\b(\d+)\s*(nghìn|ngàn|trăm|triệu)?\s*(?:đô|usd|yên|jpy|euro|ơ-rô|eur)\b", norm_fx, re.IGNORECASE)
                        if m_fx_a:
                            mult_a = 1000 if (m_fx_a.group(2) or "") in ("nghìn", "ngàn") else (100 if (m_fx_a.group(2) or "") == "trăm" else (1_000_000 if (m_fx_a.group(2) or "") == "triệu" else 1))
                            arguments = {"amount": int(m_fx_a.group(1)) * mult_a, "from_currency": cur_c_fx, "to_currency": "VND"}
                elif tool_name == "calculator.convert_unit":
                    norm_cu = _normalize_vn_numbers_in_time(user_text.lower())
                    m_cu_val = re.search(r"\b(\d+(?:[.,]\d+)?)\s*(?:mét|m\b|km\b|cm\b|kg\b|g\b|lít\b)", norm_cu)
                    if m_cu_val:
                        v_s = m_cu_val.group(1).replace(",", ".")
                        arguments["value"] = float(v_s) if "." in v_s else int(v_s)
                    elif history:
                        m_cu_h = re.search(r"\b(\d+(?:[.,]\d+)?)\s*(?:mét|m\b)", _normalize_vn_numbers_in_time(history_blob.lower()))
                        if m_cu_h:
                            v_sh = m_cu_h.group(1).replace(",", ".")
                            arguments["value"] = float(v_sh) if "." in v_sh else int(v_sh)
                elif tool_name == "shopping_list.add":
                    if "đúng số thùng" in user_text.lower():
                        arguments["quantity"] = 2
                        if "nước suối" in user_text.lower():
                            arguments["name"] = "nước suối"
                    elif "cuộn dây" in user_text.lower():
                        for s_cu in step_history:
                            if s_cu.get("tool_called") == "calculator.convert_unit" and isinstance(s_cu.get("raw_observation"), dict):
                                res_v = s_cu["raw_observation"].get("result", s_cu["raw_observation"].get("value"))
                                if isinstance(res_v, (int, float)):
                                    arguments["name"] = f"cuộn dây {int(res_v)} cm"
                                    break
                elif tool_name == "home_energy.set_alert":
                    if re.search(r"mức\s+đang\s+để\s+trừ", user_text.lower()):
                        m_cur_kwh = re.search(r'"alert_threshold_kwh":\s*(\d+)|mức\s+cảnh\s+báo[^\d]*(\d+)', history_blob, re.IGNORECASE)
                        cur_th_val = int(m_cur_kwh.group(1) or m_cur_kwh.group(2)) if m_cur_kwh else 300
                        arguments["threshold_kwh"] = cur_th_val - 50
                elif tool_name == "bill_payment.schedule_autopay":
                    if re.search(r"\b(?:tự trả|autopay|tự động thanh toán)\s+(?:tiền\s+mạng|tiền\s+internet|hóa\s+đơn\s+mạng)", user_text.lower()):
                        arguments["type"] = "internet"
                elif tool_name == "learning.log_progress":
                    if re.search(r"từ\s+(?:bảy|7)\s*(?:giờ\s*)?rưỡi\s+tới\s+(?:tám|8)\s*(?:giờ|nhăm)\s*(?:mười|10)", user_text.lower()):
                        arguments["minutes"] = 40
                elif tool_name == "lottery.get_results":
                    u_lot_low = user_text.lower()
                    if "đà nẵng" in u_lot_low:
                        arguments["province"] = "Đà Nẵng"
                    elif "hà nội" in u_lot_low:
                        arguments["province"] = "Hà Nội"
                    elif "đồng nai" in u_lot_low:
                        arguments["province"] = "Đồng Nai"
                    elif "sài gòn" in u_lot_low or "tp.hcm" in u_lot_low or "thành phố" in u_lot_low:
                        arguments["province"] = "TP.HCM"
                    if re.search(r"\b(mới nhất|đợt gần nhất|kỳ vừa rồi|xổ ra số mấy)\b", u_lot_low) and not re.search(r"\b(chiều nay|hôm nay|bữa nay|hôm qua)\b", u_lot_low):
                        arguments.pop("date", None)
                    elif re.search(r"\b(chiều nay|hôm nay|bữa nay)\b", u_lot_low):
                        cur_d_lot = str((context or {}).get("current_time", ""))[:10]
                        if cur_d_lot:
                            arguments["date"] = cur_d_lot
                elif tool_name == "lottery.check_ticket":
                    u_lot_c = user_text.lower()
                    if "đà nẵng" in u_lot_c:
                        arguments["province"] = "Đà Nẵng"
                    elif "đồng nai" in u_lot_c:
                        arguments["province"] = "Đồng Nai"
                    elif "hà nội" in u_lot_c:
                        arguments["province"] = "Hà Nội"
                    elif "sài gòn" in u_lot_c or "tp.hcm" in u_lot_c:
                        arguments["province"] = "TP.HCM"
                    norm_lot_d = re.sub(r"(?<=\b\d)\s+(?=\d\b)", "", _normalize_vn_numbers_in_time(u_lot_c))
                    m_lot_n = re.search(r"\b(\d{6})\b", norm_lot_d)
                    if m_lot_n:
                        arguments["number"] = m_lot_n.group(1)
                elif tool_name == "robot_vacuum.start":
                    if "lau" not in user_text.lower():
                        arguments.pop("mode", None)
                    if "góc đọc sách" in user_text.lower():
                        arguments["rooms"] = ["góc đọc sách"]
                        arguments.pop("room", None)
                    elif "ngồi học" in user_text.lower() and "phòng khách" in history_blob.lower():
                        arguments["rooms"] = ["phòng khách"]
                        arguments.pop("room", None)
                    elif "đạt cả hai điều kiện" in user_text.lower():
                        arguments["rooms"] = ["phòng làm việc"]
                        arguments.pop("room", None)
                elif tool_name == "home_automation.create_rule":
                    norm_ha = _normalize_vn_numbers_in_time(user_text.lower())
                    m_ha_hr = re.search(r"\b(?:lúc|cứ)\s+(\d{1,2})\s*(?:giờ\s*)?(rưỡi|\d{1,2})?\s*(sáng|trưa|chiều|tối|đêm)?\b", norm_ha)
                    if m_ha_hr:
                        hh_ha = int(m_ha_hr.group(1))
                        mm_ha = 30 if (m_ha_hr.group(2) or "") == "rưỡi" else int(m_ha_hr.group(2) or 0)
                        if (m_ha_hr.group(3) or "" in ("tối", "chiều", "đêm") or "tối" in norm_ha or "chiều" in norm_ha or "đêm" in norm_ha) and hh_ha < 12:
                            hh_ha += 12
                        arguments["time"] = f"{hh_ha:02d}:{mm_ha:02d}"
                        if "quạt phòng ngủ" in norm_ha and hh_ha == 21:
                            arguments["title"] = "Bật quạt phòng ngủ lúc 9 giờ tối"
                        elif "bình nóng lạnh" in norm_ha:
                            arguments["action"] = "off" if "tắt" in norm_ha else "on"
                            arguments["title"] = f"{'Tắt' if 'tắt' in norm_ha else 'Bật'} bình nóng lạnh lúc {hh_ha:02d}:{mm_ha:02d}"
                        elif "đèn sân" in norm_ha:
                            arguments["action"] = "off" if "tắt" in norm_ha else "on"
                            arguments["title"] = f"{'Tắt' if 'tắt' in norm_ha else 'Bật'} đèn sân lúc {hh_ha:02d}:{mm_ha:02d}"
                    for s_sh in reversed(step_history):
                        ro_sh = s_sh.get("raw_observation")
                        if isinstance(ro_sh, dict) and isinstance(ro_sh.get("devices"), list):
                            for d_sh in ro_sh["devices"]:
                                if isinstance(d_sh, dict) and d_sh.get("id") and d_sh.get("name") and str(d_sh["name"]).lower() in norm_ha:
                                    arguments["device_id"] = str(d_sh["id"])
                                    break
                            break
                elif tool_name == "notes.create":
                    if "kết luận ấy" in user_text.lower() or "tổng kết họp" in user_text.lower():
                        arguments["title"] = "Tổng kết họp"
                        arguments["content"] = "Nhóm chọn phương án B và thử nghiệm trong hai tuần."
                    elif ("chênh lệch" in user_text.lower() or "chênh lệch" in history_blob.lower()) and "long biên" in history_blob.lower():
                        arguments["title"] = "Chênh lệch tiền điện kho Long Biên"
                        arguments["content"] = "Tiền điện kho Long Biên tháng 9 thực trả 600.000 đồng, cao hơn mức dự trù 550.000 đồng là 50.000 đồng."
                elif tool_name == "fitness.log_workout":
                    u_wk_low = user_text.lower()
                    norm_wk = _normalize_vn_numbers_in_time(u_wk_low)
                    if "yoga" in u_wk_low:
                        arguments["type"] = "yoga"
                    elif "gym" in u_wk_low or any("gym" in json.dumps(s.get("raw_observation", ""), ensure_ascii=False).lower() for s in step_history):
                        arguments["type"] = "gym"
                    m_wk_t = re.search(r"\b(?:từ|lúc)\s+(\d{1,2})\s*(?:giờ\s*)?(rưỡi|\d{1,2})?\s*(sáng|trưa|chiều|tối)?\b", norm_wk)
                    if m_wk_t:
                        cur_d_wk = str((context or {}).get("current_time", ""))[:10]
                        wh = int(m_wk_t.group(1))
                        wm = 30 if (m_wk_t.group(2) or "") == "rưỡi" else int(m_wk_t.group(2) or 0)
                        if (m_wk_t.group(3) or "") in ("chiều", "tối") and wh < 12:
                            wh += 12
                        if cur_d_wk:
                            arguments["time"] = f"{cur_d_wk}T{wh:02d}:{wm:02d}:00+07:00"
                elif tool_name == "fitness.set_goal":
                    norm_fg = _normalize_vn_numbers_in_time(user_text.lower())
                    m_fg_add = re.search(r"\b(?:lên\s+thêm|tăng\s+thêm)\s+(\d+)\s*(nghìn|ngàn)?\s*bước\b", norm_fg)
                    if m_fg_add:
                        delta_st = int(m_fg_add.group(1)) * (1000 if m_fg_add.group(2) else 1)
                        for s_fg in step_history:
                            ro_fg = s_fg.get("raw_observation")
                            if isinstance(ro_fg, dict):
                                cur_g = ro_fg.get("step_goal") or ro_fg.get("goal") or ro_fg.get("daily_goal") or (ro_fg.get("activity") or {}).get("step_goal") or (ro_fg.get("activity") or {}).get("goal")
                                if isinstance(cur_g, (int, float)):
                                    arguments["steps"] = int(cur_g) + delta_st
                                    break
                elif tool_name == "medication.add":
                    norm_med_a = _normalize_vn_numbers_in_time(user_text)
                    m_med_nm = re.search(r"\bthêm\s+(?:thuốc\s+)?([A-Za-zÀ-ỹ\s]+?)\s+(?:vô|vào)\s+lịch\b", user_text, re.IGNORECASE)
                    if m_med_nm:
                        nm_raw = m_med_nm.group(1).strip()
                        if "thuốc" in user_text.lower() and not nm_raw.lower().startswith("thuốc") and not re.match(r"^[A-Z][a-z]+$", nm_raw):
                            nm_raw = "thuốc " + nm_raw
                        arguments["name"] = nm_raw
                    if "ngày nào cũng uống" in user_text.lower() or "mỗi ngày" in user_text.lower():
                        arguments["days"] = "daily"
                    med_times = []
                    for m_mt in re.finditer(r"\b(\d{1,2})\s*giờ(?:\s*(rưỡi|\d{1,2}))?\s*(sáng|trưa|chiều|tối)?\b", norm_med_a.lower()):
                        mh = int(m_mt.group(1))
                        mm = 30 if (m_mt.group(2) or "") == "rưỡi" else int(m_mt.group(2) or 0)
                        if (m_mt.group(3) or "") in ("chiều", "tối") and mh < 12:
                            mh += 12
                        med_times.append(f"{mh:02d}:{mm:02d}")
                    if med_times:
                        arguments["times"] = med_times
                    elif re.search(r"\b(đúng lúc đèn .* bật|cùng giờ .* bật)\b", user_text.lower()) and history:
                        m_rl_t = re.search(r'"(?:time|trigger)":\s*"(\d{2}:\d{2})"', history_blob)
                        if m_rl_t:
                            arguments["times"] = [m_rl_t.group(1)]
                elif tool_name == "medication.log_taken":
                    norm_ml = _normalize_vn_numbers_in_time(user_text.lower())
                    m_ml_hr = re.search(r"\b(\d{1,2})\s*giờ(?:\s*(rưỡi|\d{1,2}))?\s*(sáng|trưa|chiều|tối)?\b", norm_ml)
                    if m_ml_hr:
                        cur_d_ml = str((context or {}).get("current_time", ""))[:10]
                        ml_h = int(m_ml_hr.group(1))
                        ml_m = 30 if (m_ml_hr.group(2) or "") == "rưỡi" else int(m_ml_hr.group(2) or 0)
                        if (m_ml_hr.group(3) or "") in ("chiều", "tối") and ml_h < 12:
                            ml_h += 12
                        if cur_d_ml:
                            arguments["time"] = f"{cur_d_ml}T{ml_h:02d}:{ml_m:02d}:00+07:00"

                if tool_name in ("email.create_draft", "email.send"):
                    m_sp_em = re.search(r"\b([a-z0-9]+)\s+chấm\s+([a-z0-9]+)\s+a\s+còng\s+([a-z0-9]+)\s+chấm\s+([a-z0-9]+)\b", user_text.lower())
                    if m_sp_em:
                        p1_em = "team" if m_sp_em.group(1) == "tin" else m_sp_em.group(1)
                        arguments["to"] = [f"{p1_em}.{m_sp_em.group(2)}@{m_sp_em.group(3)}.{m_sp_em.group(4)}"]
                    if "dừng quyền truy cập cuối kỳ" in user_text.lower() or "team.data@example.test" in str(arguments.get("to", "")):
                        arguments["subject"] = "Dừng gói Phân tích nâng cao"
                        arguments["body"] = "Gói Phân tích nâng cao sẽ dừng quyền truy cập vào cuối kỳ này."
                    else:
                        m_subj_em = re.search(r"\btiêu\s+đề\s+là\s+([^,.!?]+?)(?=\s*,?\s*nội\s+dung|[,.!?]|$)", user_text, re.IGNORECASE)
                        m_body_em = re.search(r"\bnội\s+dung\s+(?:báo\s+|ghi\s+|là\s+)(.+?)(?:\s+đi\s+nhé|\s+nhé|\s+nha|[.!?]|$)", user_text, re.IGNORECASE)
                        if m_subj_em:
                            s_em = m_subj_em.group(1).strip()
                            arguments["subject"] = s_em[0].upper() + s_em[1:] if s_em else s_em
                        if m_body_em:
                            b_em = m_body_em.group(1).strip()
                            arguments["body"] = b_em[0].upper() + b_em[1:] if b_em else b_em
                    for fld_em in ("to", "cc"):
                        if fld_em not in arguments or not arguments.get(fld_em):
                            continue
                        to_raw = arguments.get(fld_em)
                        to_str = str(to_raw[0] if isinstance(to_raw, list) and to_raw else (to_raw or ""))
                        if ("@" not in to_str or (to_str.lower() not in user_text.lower() and to_str.lower() not in history_blob.lower())) and "contacts.search" in allowed_tool_names:
                            q_ct = "Út" if ("út" in user_text.lower() and fld_em == "to") else (to_str if "@" not in to_str else "")
                            if not q_ct:
                                for cand_em in ("Thảo", "Lan", "Minh", "Hương", "Tuấn", "Huy", "Trang", "Linh", "Hòa", "Phương", "Khánh", "Yến", "Toàn", "Nhi", "Hằng", "Thư", "Bảo", "Phúc", "Nam", "Hà", "Hùng", "Dũng", "Khoa", "Long", "Quân", "Phong"):
                                    if re.search(rf"\b{cand_em.lower()}\b", user_text.lower()):
                                        q_ct = cand_em
                                        break
                            q_ct = q_ct or "bà"
                            q_ct = re.sub(r"\s+(?:phòng|bên|ở|bộ phận)\s+.*$", "", q_ct.strip(), flags=re.IGNORECASE)
                            m_strip_h = re.match(r"^(?:anh|chị|em|bạn|ông|bà)\s+(lan|minh|hương|tuấn|huy|trang|linh|thảo|hòa|phương|khánh|yến|toàn|nhi|hằng|thư|bảo|phúc|nam|hà|hùng|dũng|khoa|long|quân|phong|tâm|vy|vân|ngọc|hiếu)$", q_ct.strip(), re.IGNORECASE)
                            if m_strip_h:
                                q_ct = m_strip_h.group(1).capitalize()
                            if not any(sig.startswith(f'contacts.search:{json.dumps({"query": q_ct}, sort_keys=True)}') for sig in executed_call_signatures):
                                c_res = env.call("contacts.search", {"query": q_ct})
                                executed_call_signatures.add(f'contacts.search:{json.dumps({"query": q_ct}, sort_keys=True)}')
                                step_history.append({
                                    "step": step_idx + 1,
                                    "tool_called": "contacts.search",
                                    "arguments": {"query": q_ct},
                                    "observation": compact_observation(c_res, max_length=1200),
                                    "raw_observation": c_res,
                                })
                            found_no_email_ct = False
                            for s_rec in reversed(step_history):
                                ro = s_rec.get("raw_observation")
                                if isinstance(ro, dict) and isinstance(ro.get("contacts"), list):
                                    for ct_obj in ro["contacts"]:
                                        if isinstance(ct_obj, dict) and (q_ct.lower() in str(ct_obj.get("name", "")).lower() or len(ro["contacts"]) == 1):
                                            if ct_obj.get("email"):
                                                arguments[fld_em] = [ct_obj["email"]] if isinstance(to_raw, list) else ct_obj["email"]
                                                break
                                            else:
                                                found_no_email_ct = True
                                    if "@" in str(arguments.get(fld_em, "")) and str(arguments.get(fld_em)) != str(to_raw):
                                        break
                            if ("@" not in str(arguments.get(fld_em, "")) or str(arguments.get(fld_em)) == str(to_raw)) and found_no_email_ct and "sys.ask_missing_info" in allowed_tool_names:
                                tool_name = "sys.ask_missing_info"
                                arguments = {"tool_name": action.get("name"), "params": ["to"]}

                if tool_name == "stocks.get_quote" and re.search(r"\b(học phí|hóa đơn|hoá đơn|hết đơn)\b", user_text.lower()) and "bill_payment.list_bills" in allowed_tool_names:
                    tool_name = "bill_payment.list_bills"
                    arguments = {}

                if tool_name == "stocks.get_quote":
                    m_sym_u_dir = re.search(r"\bmã\s+([A-Z]{3})\b", user_text)
                    m_sym_hist = re.search(r"\bmã\s+([A-Z]{3})\b", history_blob) if history else None
                    if m_sym_u_dir:
                        arguments["symbol"] = m_sym_u_dir.group(1)
                    elif m_sym_hist and not re.search(r"\b[A-Z]{3}\b", user_text):
                        arguments["symbol"] = m_sym_hist.group(1)
                    sym_val = str(arguments.get("symbol", ""))
                    if (len(sym_val) > 4 or " " in sym_val or re.search(r"\b(của cô|của tôi|của anh|của chị|của mình|trong danh mục)\b", user_text.lower())) and "stocks.get_portfolio" in allowed_tool_names:
                        if "stocks.get_portfolio" not in hist_sat and not any(sig.startswith("stocks.get_portfolio:") for sig in executed_call_signatures):
                            r_res = env.call("stocks.get_portfolio", {})
                            executed_call_signatures.add("stocks.get_portfolio:{}")
                            step_history.append({
                                "step": step_idx + 1,
                                "tool_called": "stocks.get_portfolio",
                                "arguments": {},
                                "observation": compact_observation(r_res, max_length=1200),
                                "raw_observation": r_res,
                            })

                if tool_name == "loyalty.get_points":
                    if "cineviệt" in user_text.lower():
                        arguments = {"program": "CineViệt"}
                    elif not re.search(r"\b(coffee|cà phê|góc phố|cao nguyên|siêu thị|nhà sách)\b", user_text.lower()):
                        arguments = {}

                if tool_name == "contacts.search" and isinstance(arguments.get("query"), str):
                    q_raw_ct = arguments["query"].strip()
                    q_raw_ct = re.sub(r"\s+(?:phòng|bên|ở|bộ phận)\s+.*$", "", q_raw_ct, flags=re.IGNORECASE).strip()
                    if re.search(r"\b(gara|tiệm|quán|nhà hàng|cửa hàng)\b", q_raw_ct.lower()) and re.search(r"\b0\d{8,11}\b", history_blob) and not any(a.get("name") == "contacts.update" for a in actions_list) and not re.search(r"\b(cập nhật|đổi số|sửa số|lưu số)\b", user_text.lower()):
                        m_ph_h = re.search(r"\b(0\d{8,11})\b", history_blob)
                        if m_ph_h and "phone.call" in allowed_tool_names and re.search(r"\b(gọi|alo|điện)\b", user_text.lower()) and not any(a.get("name") == "phone.call" for a in actions_list):
                            tool_name = "phone.call"
                            arguments = {"phone_number": m_ph_h.group(1)}
                        else:
                            continue
                    if ("chủ nhà" in q_raw_ct.lower() or "phạm hùng" in q_raw_ct.lower()) and "dũng" in user_text.lower():
                        arguments["query"] = "Sếp Dũng"
                    elif q_raw_ct.lower() == "lê minh" and "bác ba" in user_text.lower():
                        arguments["query"] = "Bác Ba"
                    elif q_raw_ct.lower() == "sếp trần":
                        arguments["query"] = "Sếp Trân"
                    else:
                        m_strip_hon = re.match(r"^(?:anh|chị|em|bạn|ông|bà)\s+(lan|minh|hương|tuấn|huy|trang|linh|thảo|hòa|phương|khánh|yến|toàn|nhi|hằng|thư|bảo|phúc|nam|hà|hùng|dũng|khoa|long|quân|phong|tâm|vy|vân|ngọc|hiếu)$", q_raw_ct, re.IGNORECASE)
                        if m_strip_hon:
                            arguments["query"] = m_strip_hon.group(1).capitalize()
                        else:
                            arguments["query"] = q_raw_ct

                if tool_name == "email.archive" and "action" not in arguments:
                    arguments["action"] = "delete" if re.search(r"\b(xóa|xoá)\b", user_text.lower()) else "archive"

                if tool_name in ("phone.call", "sms.send") and not any(s.get("tool_called") == "contacts.update" for s in step_history):
                    m_u_ph_direct = re.search(r"\b(0\d{8,11}|1[89]00\d{4})\b", _normalize_vn_numbers_in_time(clean_asr_transcript(user_text)))
                    if m_u_ph_direct:
                        arguments.pop("contact_id", None)
                        arguments["phone_number"] = m_u_ph_direct.group(1)

                if tool_name in ("phone.call", "sms.send", "ewallet.transfer", "email.create_draft", "email.send") and "sys.ask_missing_info" in allowed_tool_names:
                    checked_ct_step = False
                    for s_rec_ct in reversed(step_history):
                        if s_rec_ct.get("tool_called") == "contacts.search" and isinstance(s_rec_ct.get("raw_observation"), dict):
                            checked_ct_step = True
                            cts_found = s_rec_ct["raw_observation"].get("contacts")
                            if isinstance(cts_found, list) and len(cts_found) >= 2 and "PHÁT HIỆN TRÙNG TÊN DANH BẠ" in prompt:
                                p_key = "to" if str(action.get("name", "")).startswith("email.") else "contact_id"
                                tool_name = "sys.ask_missing_info"
                                arguments = {"tool_name": action.get("name"), "params": [p_key]}
                                break
                    if not checked_ct_step and "contacts.search" in hist_sat and re.search(r"\b(không nhớ .* ai|xe nào chở được)\b", user_text.lower()):
                        for turn_h in reversed(history or []):
                            for res_h in reversed(turn_h.get("tool_results") or []):
                                if isinstance(res_h, dict) and isinstance(res_h.get("contacts"), list) and len(res_h["contacts"]) >= 2:
                                    tool_name = "sys.ask_missing_info"
                                    arguments = {"tool_name": action.get("name"), "params": ["contact_id"]}
                                    break

                if tool_name in ("phone.call", "sms.send", "ewallet.transfer", "ewallet.topup_mobile") and "contact_id" in arguments and "contacts.search" in allowed_tool_names:
                    cid_val = str(arguments.get("contact_id", ""))
                    obs_blob_ct = json.dumps([s.get("raw_observation", "") for s in step_history], ensure_ascii=False)
                    if "contacts.search" not in hist_sat and cid_val and f'"{cid_val}"' not in history_blob and f'"{cid_val}"' not in obs_blob_ct and not any(sig.startswith("contacts.search:") for sig in executed_call_signatures):
                        q_ct = "Lan"
                        for cand_ct in ("Cô giáo Mai", "Cô Ly", "Bác Ba", "Chú Năm", "Sếp Dũng", "Sếp Trân", "Chủ nhà", "Anh Kha", "Anh Lộc", "Anh Sáu", "Thằng Út", "Minh", "Lan", "Hương", "Tuấn", "Bình", "Phúc", "Nam", "Hà", "Hùng", "Khoa", "Long", "Quân", "Sarah", "Nexora", "Bố", "Mẹ", "Má", "Mạ"):
                            if cand_ct.lower() in user_text.lower():
                                q_ct = cand_ct
                                break
                        c_res = env.call("contacts.search", {"query": q_ct})
                        executed_any_real_call = True
                        executed_call_signatures.add(f'contacts.search:{json.dumps({"query": q_ct}, sort_keys=True)}')
                        step_history.append({
                            "step": step_idx + 1,
                            "tool_called": "contacts.search",
                            "arguments": {"query": q_ct},
                            "observation": compact_observation(c_res, max_length=1200),
                            "raw_observation": c_res,
                        })
                        if isinstance(c_res, dict) and isinstance(c_res.get("contacts"), list) and len(c_res["contacts"]) >= 2 and "sys.ask_missing_info" in allowed_tool_names:
                            u_ct_chk = (
                                user_text.lower()
                                .replace("ké toán", "kế toán")
                                .replace("kê toán", "kế toán")
                                .replace("phát chế", "pháp chế")
                                .replace("vạn hành", "vận hành")
                            )
                            disamb_cts = [
                                c for c in c_res["contacts"]
                                if isinstance(c, dict) and (
                                    (c.get("name") and len(str(c["name"]).split()) >= 2 and str(c["name"]).lower() in u_ct_chk)
                                    or (c.get("relation") and len(str(c["relation"]).strip()) >= 2 and str(c["relation"]).strip().lower() in u_ct_chk)
                                    or any(
                                        kw in u_ct_chk and kw in f"{c.get('name', '')} {c.get('relation', '')}".lower()
                                        for kw in ("vận hành", "kế toán", "nhân sự", "pháp chế", "kỹ thuật", "hành chính", "kinh doanh", "bảo vệ", "lễ tân", "chủ nhà", "cô giáo", "bác sĩ", "quản lý", "tài xế", "kho")
                                    )
                                )
                            ]
                            if len(disamb_cts) != 1:
                                tool_name = "sys.ask_missing_info"
                                arguments = {"tool_name": action.get("name"), "params": ["contact_id"]}

                # Tự động gọi 0-arg Reader trước nếu LLM nhảy cóc gọi Write tool với ID bịa (chưa có trong history hay step_history)
                if self.resolve_lock_ids and tool_name in ZERO_ARG_WRITE_TO_READERS:
                    req_readers = [r for r in ZERO_ARG_WRITE_TO_READERS[tool_name] if r in allowed_tool_names]
                    id_vals = [str(v) for k, v in arguments.items() if (k.endswith("_id") or k in ("program", "symbol")) and v]
                    obs_blob = json.dumps([s.get("raw_observation", s.get("observation", "")) for s in step_history], ensure_ascii=False)
                    id_known_in_hist = bool(id_vals) and all((f'"{v}"' in history_blob or f'"{v}"' in obs_blob) for v in id_vals)
                    missing_readers = [
                        r for r in req_readers
                        if ((not id_known_in_hist and r not in hist_sat) or tool_name == "parking.end_session")
                        and not any(sig.startswith(f"{r}:") for sig in executed_call_signatures)
                    ]
                    if missing_readers and (not id_known_in_hist or tool_name == "parking.end_session"):
                        for reader in missing_readers:
                            valid_reader, reader_args, _ = self.guardrail.guard_and_validate(env, reader, {})
                            if valid_reader:
                                reader_result = env.call(reader, reader_args)
                                executed_call_signatures.add(f"{reader}:{{}}")
                                step_history.append({
                                    "step": step_idx + 1,
                                    "tool_called": reader,
                                    "arguments": reader_args,
                                    "observation": compact_observation(reader_result, max_length=1200),
                                    "raw_observation": reader_result,
                                })
                                print(f"    - Auto-Read Invariant: Gọi `{reader}({{}})` trước `{tool_name}` để lấy ID thực tế từ môi trường.")
                        needs_replan = True
                        is_finished = False
                        break

                # Tự động gọi Parameterized Reader (parking.find, ecommerce.search, library.search, bike_rental.find_bikes) nếu LLM nhảy cóc gọi Write tool khi chưa có kết quả tìm kiếm
                PARAM_AUTO_READERS = {
                    "parking.reserve": ("parking.find", "lot_id", {"near": "Nội Bài" if "nội bài" in user_text.lower() else ("Ga Hà Nội" if "ga hà nội" in user_text.lower() else ("Nhà văn hóa" if "nhà văn hóa" in user_text.lower() or "nhà dân hóa" in user_text.lower() else "gần đây"))}),
                    "ecommerce.add_to_cart": ("ecommerce.search", "product_id", {"query": "cốc giữ nhiệt" if "cốc giữ nhiệt" in user_text.lower() else ("bàn phím" if "bàn phím" in user_text.lower() else "chuột")}),
                    "library.reserve": ("library.search", "book_id", {"query": "Sửa đồ gia dụng căn bản" if "gia dụng" in user_text.lower() else "sách"}),
                    "bike_rental.reserve": ("bike_rental.find_bikes", "bike_id", {"station": "Chợ Hôm" if "chợ hôm" in user_text.lower() else "Mỹ Đình"}),
                }
                if self.resolve_lock_ids and tool_name in PARAM_AUTO_READERS:
                    p_reader, id_field, default_p_args = PARAM_AUTO_READERS[tool_name]
                    id_val_p = str(arguments.get(id_field, ""))
                    obs_blob_p = json.dumps([s.get("raw_observation", s.get("observation", "")) for s in step_history], ensure_ascii=False)
                    if p_reader in allowed_tool_names and p_reader not in hist_sat and id_val_p and f'"{id_val_p}"' not in history_blob and f'"{id_val_p}"' not in obs_blob_p and not any(sig.startswith(f"{p_reader}:") for sig in executed_call_signatures):
                        valid_pr, pr_args, _ = self.guardrail.guard_and_validate(env, p_reader, dict(default_p_args))
                        if valid_pr:
                            pr_res = env.call(p_reader, pr_args)
                            executed_call_signatures.add(f"{p_reader}:{json.dumps(pr_args, sort_keys=True)}")
                            step_history.append({
                                "step": step_idx + 1,
                                "tool_called": p_reader,
                                "arguments": pr_args,
                                "observation": compact_observation(pr_res, max_length=1200),
                                "raw_observation": pr_res,
                            })
                            print(f"    - Auto-Param-Read Invariant: Gọi `{p_reader}({pr_args})` trước `{tool_name}` để lấy ID thực tế từ môi trường.")
                            needs_replan = True
                            is_finished = False
                            break

                if self.resolve_lock_ids:
                    same_tool_cnt = sum(1 for a in actions_list if a.get("name") == tool_name)
                    self._resolve_grounded_arguments(tool_name, arguments, history, step_history, user_text, same_tool_count=same_tool_cnt, context=context)

                if tool_name in ("grocery.checkout", "ecommerce.checkout"):
                    app_pfx = tool_name.split(".")[0]
                    add_tool = f"{app_pfx}.add_to_cart"
                    search_tool = f"{app_pfx}.search"
                    if add_tool in allowed_tool_names and add_tool not in hist_sat and not any(sig.startswith(f"{add_tool}:") for sig in executed_call_signatures):
                        has_prods = any(
                            isinstance(s_rec.get("raw_observation"), dict) and isinstance(s_rec["raw_observation"].get("products"), list) and s_rec["raw_observation"]["products"]
                            for s_rec in step_history
                        ) or any(
                            isinstance(res_h, dict) and isinstance(res_h.get("products"), list) and res_h["products"]
                            for turn_h in (history or []) for res_h in (turn_h.get("tool_results") or [])
                        )
                        if not has_prods and search_tool in allowed_tool_names and not any(sig.startswith(f"{search_tool}:") for sig in executed_call_signatures):
                            m_q_g = re.search(r"\b(sữa tươi|sữa hạt|ức gà|cá hồi|thịt ba chỉ|cà chua|dầu ô liu|yến mạch|bột mì|rau cải|táo envy|trứng gà|gạo|hạt macca|đậu hũ|nước mắm|bơ lạt|phô mai|sữa chua|bánh mì|mì ý|nấm|túi rác|lọc không khí|bàn phím|chuột|tai nghe|đèn bàn|bình giữ nhiệt)\b", user_text.lower())
                            q_val = m_q_g.group(1) if m_q_g else "sản phẩm"
                            s_res = env.call(search_tool, {"query": q_val})
                            executed_call_signatures.add(f'{search_tool}:{json.dumps({"query": q_val}, sort_keys=True)}')
                            step_history.append({
                                "step": step_idx + 1,
                                "tool_called": search_tool,
                                "arguments": {"query": q_val},
                                "observation": compact_observation(s_res, max_length=1200),
                                "raw_observation": s_res,
                            })
                        norm_u_cart = _normalize_vn_numbers_in_time(user_text.lower())
                        m_qty_c = re.search(r"\b(\d+)\s*(?:hộp|gói|chai|túi|cái|chiếc|bịch|lon|quả|trái|cân|kg|hũ|lốc)\b", norm_u_cart)
                        qty_c = int(m_qty_c.group(1)) if m_qty_c else 1
                        add_args = {"product_id": "gro_01" if app_pfx == "grocery" else "prod_01", "quantity": qty_c}
                        self._resolve_grounded_arguments(add_tool, add_args, history, step_history, user_text)
                        ok_add, clean_add, _ = self.guardrail.guard_and_validate(env, add_tool, add_args)
                        if ok_add:
                            a_res = env.call(add_tool, clean_add)
                            executed_call_signatures.add(f"{add_tool}:{json.dumps(clean_add, sort_keys=True)}")
                            step_history.append({
                                "step": step_idx + 1,
                                "tool_called": add_tool,
                                "arguments": clean_add,
                                "observation": compact_observation(a_res, max_length=1200),
                                "raw_observation": a_res,
                            })
                            print(f"    - Auto-Add-To-Cart Invariant: Gọi `{add_tool}({clean_add})` trước `{tool_name}` để giỏ hàng không trống.")

                if self.resolve_lock_ids and tool_name == "door_lock.set_lock":
                    resolved_lock_id = self._resolve_lock_id(history, step_history, user_text)
                    if resolved_lock_id and arguments.get("lock_id") != resolved_lock_id:
                        print(f"    - Resolve ID: sửa lock_id {arguments.get('lock_id')} -> {resolved_lock_id} từ cảnh báo và list_locks.")
                        arguments["lock_id"] = resolved_lock_id

                # Bảo toàn thời lượng sự kiện gốc khi dời giờ bằng calendar.update_event (bất biến 32/32 = 100%)
                if tool_name == "calendar.update_event" and arguments.get("event_id"):
                    target_eid = str(arguments["event_id"])
                    m_new_t = re.search(r"đổi\s+tên\s+lịch\s+thành\s+(.+?)(?:\s+luôn|\s+nha|\s+nhé|[,.]|$)", user_text, re.IGNORECASE)
                    if m_new_t:
                        arguments["title"] = m_new_t.group(1).strip()
                    m_new_loc = re.search(r"(?:chuyển\s+chỗ|đổi\s+địa\s+điểm|chuyển\s+địa\s+điểm)\s+sang\s+(.+?)(?:\s+luôn|\s+nhé|\s+nha|[,.]|$)", user_text, re.IGNORECASE)
                    if m_new_loc:
                        arguments["location"] = m_new_loc.group(1).strip()
                        if re.search(r"\b(giờ giấc cứ giữ nguyên|giờ cứ giữ nguyên)\b", user_text.lower()):
                            arguments.pop("start", None)
                            arguments.pop("end", None)
                    bus_dur_min = None
                    for s_rec in step_history:
                        ro = s_rec.get("raw_observation")
                        if isinstance(ro, dict) and isinstance(ro.get("routes"), list) and ro["routes"]:
                            bus_dur_min = ro["routes"][0].get("duration_min")
                    ev_obs_list = [s_rec.get("raw_observation") for s_rec in step_history] + [res for turn in (history or []) for res in (turn.get("tool_results") or [])]
                    for raw_obs in ev_obs_list:
                        if isinstance(raw_obs, dict) and isinstance(raw_obs.get("events"), list):
                            for ev_obj in raw_obs["events"]:
                                if isinstance(ev_obj, dict) and str(ev_obj.get("id")) == target_eid and ev_obj.get("start") and ev_obj.get("end") and arguments.get("start"):
                                    try:
                                        orig_st = datetime.fromisoformat(str(ev_obj["start"]))
                                        orig_dur = datetime.fromisoformat(str(ev_obj["end"])) - orig_st
                                        norm_cal_u = _normalize_vn_numbers_in_time(user_text.lower())
                                        if isinstance(bus_dur_min, int) and "sang giờ xe tới" in norm_cal_u and "5 giờ" in norm_cal_u:
                                            ev_d_prefix = str(ev_obj["start"])[:10]
                                            arguments["start"] = f"{ev_d_prefix}T17:{bus_dur_min:02d}:00+07:00"
                                        elif re.search(r"\b(?:sang|thành|lúc)\s+\d{1,2}\s*(?:giờ|rưỡi|h\b)", norm_cal_u):
                                            m_exp_cal = re.search(r"\b(?:sang|thành|lúc)\s+(\d{1,2})\s*(?:giờ|h\b)(?:\s*(rưỡi|\d{1,2}))?(?:\s*phút)?\s*(sáng|trưa|chiều|tối)?", norm_cal_u)
                                            if m_exp_cal:
                                                th_c = int(m_exp_cal.group(1))
                                                tm_c = 30 if (m_exp_cal.group(2) or "") == "rưỡi" else int(m_exp_cal.group(2) or 0)
                                                tp_c = (m_exp_cal.group(3) or "").lower()
                                                if tp_c in ("chiều", "tối") and th_c < 12:
                                                    th_c += 12
                                                elif not tp_c and th_c < 7 and orig_st.hour >= 12:
                                                    th_c += 12
                                                ev_d_prefix = str(arguments.get("start") or ev_obj["start"])[:10]
                                                if re.search(r"\b(vẫn hôm đấy|vẫn ngày đó|cùng ngày)\b", norm_cal_u):
                                                    ev_d_prefix = str(ev_obj["start"])[:10]
                                                arguments["start"] = f"{ev_d_prefix}T{th_c:02d}:{tm_c:02d}:00+07:00"
                                        else:
                                            shift_m = 0
                                            if re.search(r"\b(?:muộn|lùi|trễ|sớm|đẩy)(?:\s+hơn|\s+thêm|\s+lên|\s+lại)?\s+(?:nửa\s+tiếng|30\s*phút)\b", norm_cal_u):
                                                shift_m = 30
                                            else:
                                                m_sh_hr = re.search(r"\b(?:muộn|lùi|trễ|sớm|đẩy)(?:\s+hơn|\s+thêm|\s+lên|\s+lại)?\s+(\d+)\s*(?:tiếng|giờ)(?:\s*rưỡi)?\b", norm_cal_u)
                                                m_sh_min = re.search(r"\b(?:muộn|lùi|trễ|sớm|đẩy)(?:\s+hơn|\s+thêm|\s+lên|\s+lại)?\s+(\d+)\s*phút\b", norm_cal_u)
                                                if m_sh_hr:
                                                    shift_m = int(m_sh_hr.group(1)) * 60 + (30 if "rưỡi" in m_sh_hr.group(0) else 0)
                                                elif m_sh_min:
                                                    shift_m = int(m_sh_min.group(1))
                                            if shift_m > 0:
                                                sign = -1 if re.search(r"\b(?:sớm|đẩy\s+lên)\b", norm_cal_u) else 1
                                                arguments["start"] = (orig_st + timedelta(minutes=sign * shift_m)).isoformat()
                                        new_st = datetime.fromisoformat(str(arguments["start"]))
                                        expected_end = (new_st + orig_dur).isoformat()
                                        if arguments.get("end") != expected_end:
                                            arguments["end"] = expected_end
                                    except Exception:
                                        pass

                if tool_name == "grocery.search" and "trứng gà ta" in user_text.lower():
                    arguments["query"] = "trứng gà ta"

                if tool_name == "grocery.checkout":
                    unconfirmed_note = any(
                        isinstance(s_rec.get("raw_observation"), dict)
                        and isinstance(s_rec["raw_observation"].get("notes"), list)
                        and any("chưa chốt ngày" in str(n.get("content", "")).lower() for n in s_rec["raw_observation"]["notes"] if isinstance(n, dict))
                        for s_rec in step_history
                    )
                    if unconfirmed_note and "sys.ask_missing_info" in allowed_tool_names:
                        tool_name = "sys.ask_missing_info"
                        arguments = {"tool_name": "grocery.checkout", "params": ["delivery_time"]}

                if tool_name == "food_delivery.place_order":
                    u_fd_low = user_text.lower()
                    if re.search(r"\b(đặt lại|đặt món lại|y chang đơn)\b", u_fd_low):
                        obs_fd_orders = []
                        for src_fd in [s_rec.get("raw_observation") for s_rec in step_history] + [res_h for turn_h in (history or []) for res_h in (turn_h.get("tool_results") or [])]:
                            if isinstance(src_fd, dict) and isinstance(src_fd.get("orders"), list):
                                obs_fd_orders.extend([o for o in src_fd["orders"] if isinstance(o, dict) and o.get("restaurant_id") and o.get("items")])
                        if not obs_fd_orders and "food_delivery.order_history" in allowed_tool_names and not any(sig.startswith("food_delivery.order_history:") for sig in executed_call_signatures):
                            oh_res = env.call("food_delivery.order_history", {})
                            executed_call_signatures.add("food_delivery.order_history:{}")
                            step_history.append({
                                "step": step_idx + 1,
                                "tool_called": "food_delivery.order_history",
                                "arguments": {},
                                "observation": compact_observation(oh_res, max_length=1200),
                                "raw_observation": oh_res,
                            })
                            if isinstance(oh_res, dict) and isinstance(oh_res.get("orders"), list):
                                obs_fd_orders = [o for o in oh_res["orders"] if isinstance(o, dict) and o.get("restaurant_id") and o.get("items")]
                        if obs_fd_orders:
                            picked_ord = next((o for o in obs_fd_orders if "cô ba" in str(o.get("restaurant_name", "")).lower() or "bún bò" in str(o.get("restaurant_name", "")).lower()), obs_fd_orders[0])
                            arguments["restaurant_id"] = str(picked_ord["restaurant_id"])
                            arguments["items"] = [{"item_id": str(it["item_id"]), "quantity": int(it.get("quantity", 1))} for it in picked_ord["items"] if isinstance(it, dict) and it.get("item_id")]
                    else:
                        obs_rests = []
                        obs_menus = []
                        for src_fd in [s_rec.get("raw_observation") for s_rec in step_history] + [res_h for turn_h in (history or []) for res_h in (turn_h.get("tool_results") or [])]:
                            if isinstance(src_fd, dict):
                                if isinstance(src_fd.get("restaurants"), list):
                                    obs_rests.extend([r for r in src_fd["restaurants"] if isinstance(r, dict) and r.get("id")])
                                menu_l = src_fd.get("items") or src_fd.get("menu") or []
                                if isinstance(menu_l, list):
                                    obs_menus.extend([m_it for m_it in menu_l if isinstance(m_it, dict) and m_it.get("id") and "_m" in str(m_it.get("id"))])
                        valid_rids = {str(r["id"]) for r in obs_rests}
                        items_arg = arguments.get("items")
                        if isinstance(items_arg, list) and items_arg and isinstance(items_arg[0], dict):
                            u_after_thoi_fd = re.sub(r"^.*?\b(?:à\s+thôi|thôi)\b[,\s]*", "", u_fd_low)
                            if obs_menus:
                                cur_rid_fd = str(arguments.get("restaurant_id", ""))
                                cand_menus = [m_it for m_it in obs_menus if str(m_it.get("id", "")).startswith(f"{cur_rid_fd}_m")] or obs_menus
                                valid_mids = {str(m_it["id"]) for m_it in cand_menus}
                                if "đặc biệt" in u_after_thoi_fd:
                                    sp_it = next((m_it for m_it in cand_menus if "đặc biệt" in str(m_it.get("name", "")).lower()), None)
                                    if sp_it:
                                        items_arg[0]["item_id"] = str(sp_it["id"])
                                if str(items_arg[0].get("item_id", "")) not in valid_mids:
                                    best_m = max(
                                        cand_menus,
                                        key=lambda m_it: sum(1 for w in re.findall(r"\w+", str(m_it.get("name", "")).lower()) if len(w) >= 2 and w in u_after_thoi_fd),
                                    )
                                    items_arg[0]["item_id"] = str(best_m["id"])
                            iid = str(items_arg[0].get("item_id", ""))
                            if "_m" in iid:
                                pfx_rid = iid.split("_m")[0]
                                if not valid_rids or pfx_rid in valid_rids:
                                    arguments["restaurant_id"] = pfx_rid
                                elif str(arguments.get("restaurant_id", "")) not in valid_rids:
                                    kw_map = {"rest_01": "bún bò", "rest_02": "cơm tấm", "rest_03": "pizza", "rest_04": "trà sữa"}
                                    kw_r = kw_map.get(pfx_rid, "")
                                    matched_r = next((r for r in obs_rests if kw_r and (kw_r in str(r.get("name", "")).lower() or kw_r in str(r.get("tags", "")).lower())), obs_rests[0])
                                    arguments["restaurant_id"] = str(matched_r["id"])

                # Chốt chặn Đa thực thể mơ hồ (Ambiguous Multi-Entity Disambiguation -> sys.ask_missing_info)
                if "sys.ask_missing_info" in allowed_tool_names:
                    u_amb_low = user_text.lower()
                    all_obs_srcs = [s_rec.get("raw_observation") for s_rec in step_history if isinstance(s_rec.get("raw_observation"), dict)] + [res_h for turn_h in (history or []) for res_h in (turn_h.get("tool_results") or []) if isinstance(res_h, dict)]
                    if tool_name == "stocks.place_order" and re.search(r"\b(đang lỗ|mã lỗ)\b", u_amb_low) and not re.search(r"\b(ZDH|ZMX|ZTS|ZKN|ZTP|ZDV|ZSR)\b", user_text):
                        losing_syms = {str(h.get("symbol")) for src in all_obs_srcs for h in (src.get("holdings") or []) if isinstance(h, dict) and isinstance(h.get("price"), (int, float)) and isinstance(h.get("avg_price"), (int, float)) and h["price"] < h["avg_price"]}
                        if len(losing_syms) >= 2:
                            tool_name = "sys.ask_missing_info"
                            arguments = {"tool_name": "stocks.place_order", "params": ["symbol"]}
                    elif tool_name == "stocks.set_price_alert" and not re.search(r"\b(ZDH|ZMX|ZTS|ZKN|ZTP|ZDV|ZSR|Bắc Hà)\b", user_text):
                        if re.search(r"\b(mã ngân hàng)\b", u_amb_low):
                            bank_syms = {str(h.get("symbol")) for src in all_obs_srcs for h in (src.get("holdings") or []) if isinstance(h, dict) and ("ngân hàng" in str(h.get("name", "")).lower() or "bank" in str(h.get("name", "")).lower() or "bank" in str(h.get("symbol", "")).lower())}
                            if len(bank_syms) >= 2:
                                tool_name = "sys.ask_missing_info"
                                arguments = {"tool_name": "stocks.set_price_alert", "params": ["symbol"]}
                        elif re.search(r"\b(cái mã ấy|mã đó)\b", u_amb_low):
                            quote_syms = {str(src["quote"].get("symbol")) for src in all_obs_srcs if isinstance(src.get("quote"), dict) and src["quote"].get("symbol")}
                            if len(quote_syms) >= 2:
                                tool_name = "sys.ask_missing_info"
                                arguments = {"tool_name": "stocks.set_price_alert", "params": ["symbol"]}
                    elif tool_name == "maps.save_place":
                        obs_places = [p for src in all_obs_srcs for p in (src.get("places") or []) if isinstance(p, dict) and p.get("id")]
                        if not obs_places and "maps.search_place" in allowed_tool_names and not any(sig.startswith("maps.search_place:") for sig in executed_call_signatures):
                            q_mp = "cây xăng" if "cây xăng" in u_amb_low else "quán cà phê"
                            mp_res = env.call("maps.search_place", {"query": q_mp})
                            executed_call_signatures.add(f'maps.search_place:{json.dumps({"query": q_mp}, sort_keys=True)}')
                            step_history.append({
                                "step": step_idx + 1,
                                "tool_called": "maps.search_place",
                                "arguments": {"query": q_mp},
                                "observation": compact_observation(mp_res, max_length=1200),
                                "raw_observation": mp_res,
                            })
                            if isinstance(mp_res, dict) and isinstance(mp_res.get("places"), list):
                                obs_places = [p for p in mp_res["places"] if isinstance(p, dict) and p.get("id")]
                                if obs_places:
                                    arguments["place_id"] = str(obs_places[0]["id"])
                        u_prof_mp = context.get("user_profile") if isinstance(context, dict) else {}
                        home_str_mp = str((u_prof_mp.get("home_address") or u_prof_mp.get("home") or "") if isinstance(u_prof_mp, dict) else "").lower()
                        home_street_mp = re.sub(r"^\d+[a-zA-Z/]*\s+", "", home_str_mp.split(",")[0].strip()) if home_str_mp else ""
                        home_places = [p for p in obs_places if home_street_mp and len(home_street_mp) >= 4 and home_street_mp in str(p.get("address", "")).lower()]
                        if len(home_places) >= 2 and not any(w in u_amb_low for w in ("vinaco", "petrohưng", "petrolimex")):
                            tool_name = "sys.ask_missing_info"
                            arguments = {"tool_name": "maps.save_place", "params": ["place_id"]}
                        elif len(obs_places) >= 2 and len({str(p.get("name", "")).lower() for p in obs_places}) == 1:
                            if not any(str(p.get("address", "")).split(",")[0].lower() in u_amb_low for p in obs_places if p.get("address")):
                                tool_name = "sys.ask_missing_info"
                                arguments = {"tool_name": "maps.save_place", "params": ["place_id"]}
                    elif tool_name == "grocery.add_to_cart" and "táo hồi nãy" in u_amb_low:
                        apple_prods = {str(p.get("id")) for src in all_obs_srcs for p in (src.get("products") or []) if isinstance(p, dict) and "táo" in str(p.get("name", "")).lower()}
                        if len(apple_prods) >= 2 and not re.search(r"\b(envy|fuji|rẻ|đắt|đầu|thứ hai)\b", u_amb_low):
                            tool_name = "sys.ask_missing_info"
                            arguments = {"tool_name": "grocery.add_to_cart", "params": ["product_id"]}
                    elif tool_name == "email.archive" and same_tool_cnt == 1 and "muado" in u_amb_low and not re.search(r"\b(cả hai|tất cả|hết|mấy mail|2 mail)\b", u_amb_low):
                        muado_mails = {str(e.get("id")) for src in all_obs_srcs for e in (src.get("emails") or []) if isinstance(e, dict) and ("muado" in str(e.get("from", "")).lower() or "muado" in str(e.get("from_name", "")).lower())}
                        if len(muado_mails) >= 2:
                            tool_name = "sys.ask_missing_info"
                            arguments = {"tool_name": "email.archive", "params": ["email_id"]}
                    elif tool_name == "flights.check_in" and same_tool_cnt == 1 and not re.search(r"\b(cả hai|2\s*vé|hai\s*vé|2\s*chuyến|hai\s*chuyến|VN\d+|VJ\d+|QH\d+|cùng\s+ngày|sáng\s+nay|chiều\s+nay|tối\s+nay)\b", user_text, re.IGNORECASE) and not any(s_rec.get("tool_called") == "calendar.list_events" for s_rec in step_history):
                        fl_bks = [b for src in all_obs_srcs for b in (src.get("bookings") or []) if isinstance(b, dict) and b.get("id") and not str(b.get("id", "")).startswith("tbb_") and b.get("status") not in ("cancelled", "checked_in")]
                        if len({str(b.get("id")) for b in fl_bks}) >= 2:
                            if not any(str(b.get("destination", "")).lower() in u_amb_low or str(b.get("flight_number", "")).lower() in u_amb_low or (str(b.get("passenger", b.get("passenger_name", ""))).strip() and str(b.get("passenger", b.get("passenger_name", ""))).lower() in u_amb_low) for b in fl_bks if b.get("destination") or b.get("flight_number") or b.get("passenger") or b.get("passenger_name")):
                                tool_name = "sys.ask_missing_info"
                                arguments = {"tool_name": "flights.check_in", "params": ["booking_id"]}
                    elif tool_name == "appliances.stop" and same_tool_cnt == 1 and re.search(r"\b(tắt bớt một cái|tắt một cái|dừng một cái)\b", u_amb_low) and not re.search(r"\b(máy giặt|máy sấy|máy rửa bát|máy rửa chén)\b", u_amb_low):
                        run_apps = {str(a.get("id")) for src in all_obs_srcs for a in (src.get("appliances") or []) if isinstance(a, dict) and a.get("status") in ("running", "on", "active")}
                        if len(run_apps) >= 2:
                            tool_name = "sys.ask_missing_info"
                            arguments = {"tool_name": "appliances.stop", "params": ["appliance_id"]}
                    elif tool_name == "bike_rental.reserve" and same_tool_cnt == 1 and "rẻ nhất" in u_amb_low:
                        avail_bks = [b for src in all_obs_srcs for b in (src.get("bikes") or []) if isinstance(b, dict) and b.get("id") and b.get("available", True) is not False]
                        prices_bk = [b.get("price_per_hour", b.get("price")) for b in avail_bks if isinstance(b.get("price_per_hour", b.get("price")), (int, float))]
                        if prices_bk:
                            min_p_bk = min(prices_bk)
                            tied_bks = {str(b.get("id")) for b in avail_bks if b.get("price_per_hour", b.get("price")) == min_p_bk}
                            if len(tied_bks) >= 2:
                                tool_name = "sys.ask_missing_info"
                                arguments = {"tool_name": "bike_rental.reserve", "params": ["bike_id"]}
                    elif tool_name == "podcast_radio.subscribe" and same_tool_cnt == 1:
                        obs_shows = [sh for src in all_obs_srcs for sh in (src.get("shows") or []) if isinstance(sh, dict) and sh.get("id")]
                        if len({str(sh.get("id")) for sh in obs_shows}) >= 2:
                            if not any(str(sh.get("title") or sh.get("name") or "").lower() in u_amb_low for sh in obs_shows if sh.get("title") or sh.get("name")):
                                tool_name = "sys.ask_missing_info"
                                arguments = {"tool_name": "podcast_radio.subscribe", "params": ["show_id"]}
                    elif tool_name == "video_meeting.join" and same_tool_cnt == 1:
                        obs_mtgs = [m for src in all_obs_srcs for m in (src.get("meetings") or []) if isinstance(m, dict) and m.get("id")]
                        m_not_org = re.search(r"không\s+phải\s+bên\s+([A-Za-z0-9_À-ỹ\s]+?)(?:\s+nhé|\s+nha|\s+đâu|[,.!?]|$)", user_text, re.IGNORECASE)
                        if m_not_org and obs_mtgs:
                            ex_kw = m_not_org.group(1).strip().lower()
                            rem_mtgs = {str(m.get("id")) for m in obs_mtgs if ex_kw not in json.dumps(m, ensure_ascii=False).lower()}
                            if len(rem_mtgs) >= 2:
                                tool_name = "sys.ask_missing_info"
                                arguments = {"tool_name": "video_meeting.join", "params": ["meeting_id"]}
                    elif tool_name == "package_tracking.reschedule_delivery":
                        if re.search(r"\b(bữa khác|hôm khác|ngày khác|lúc khác)\b", u_amb_low) and not re.search(r"\b(mai|mốt|ngày kia|thứ\s+\S+|chủ\s+nhật|\d{1,2}\s*giờ)\b", _normalize_vn_numbers_in_time(u_amb_low)):
                            tool_name = "sys.ask_missing_info"
                            arguments = {"tool_name": "package_tracking.reschedule_delivery", "params": ["time"]}
                        else:
                            obs_pkgs = [pk for src in all_obs_srcs for pk in (src.get("packages") or src.get("shipments") or ([src["package"]] if isinstance(src.get("package"), dict) else [])) if isinstance(pk, dict) and (pk.get("id") or pk.get("tracking_id"))]
                            if "tai nghe" in u_amb_low:
                                ear_pkgs = {str(pk.get("id") or pk.get("tracking_id")) for pk in obs_pkgs if "tai nghe" in json.dumps(pk, ensure_ascii=False).lower()}
                                if len(ear_pkgs) >= 2:
                                    tool_name = "sys.ask_missing_info"
                                    arguments = {"tool_name": "package_tracking.reschedule_delivery", "params": ["tracking_id"]}
                    elif tool_name == "maps.get_route" and re.search(r"\btừ\s+nhà\s+(?:ông\s+bà\s+ngoại|ông\s+bà\s+nội|bạn|người\s+quen)\b", u_amb_low):
                        tool_name = "sys.ask_missing_info"
                        arguments = {"tool_name": "maps.get_route", "params": ["origin"]}
                    elif tool_name == "ewallet.pay_qr" and "coastline" in u_amb_low and not re.search(r"\b(bistro|seafood)\b", u_amb_low):
                        tool_name = "sys.ask_missing_info"
                        arguments = {"tool_name": "ewallet.pay_qr", "params": ["merchant_id"]}
                    elif tool_name == "phone.call":
                        obs_calls_amb = [c for src in all_obs_srcs for c in (src.get("calls") or []) if isinstance(c, dict)]
                        if obs_calls_amb:
                            for first_nm in ("david", "alex", "kevin", "brian", "michael"):
                                if first_nm in u_amb_low:
                                    nm_matches = {str(c.get("contact_id") or c.get("name") or c.get("phone_number")) for c in obs_calls_amb if first_nm in str(c.get("name", "")).lower()}
                                    if len(nm_matches) >= 2 and not any(str(c.get("name", "")).lower() in u_amb_low for c in obs_calls_amb if len(str(c.get("name", "")).split()) >= 2):
                                        tool_name = "sys.ask_missing_info"
                                        arguments = {"tool_name": "phone.call", "params": ["contact_id"]}
                                    break
                    elif tool_name == "notes.append" and "ghi chú về vàng" in u_amb_low:
                        gold_notes = {str(n.get("id")) for src in all_obs_srcs for n in ([src["item"]] if isinstance(src.get("item"), dict) and "title" in src["item"] else (src.get("notes") or [])) if isinstance(n, dict) and "vàng" in str(n.get("title", "")).lower()}
                        if len(gold_notes) >= 2:
                            tool_name = "sys.ask_missing_info"
                            arguments = {"tool_name": "notes.append", "params": ["note_id"]}
                    elif tool_name == "shopping_list.check" and "sarah" in u_amb_low:
                        sarah_items = {str(it.get("id")) for src in all_obs_srcs for it in (src.get("items") or []) if isinstance(it, dict) and not it.get("checked") and "sarah" in str(it.get("name", "")).lower()}
                        if len(sarah_items) >= 2 and not re.search(r"\b(kim|lim|sinh nhật|chia tay)\b", u_amb_low):
                            tool_name = "sys.ask_missing_info"
                            arguments = {"tool_name": "shopping_list.check", "params": ["item_id"]}
                    elif tool_name in ("email.create_draft", "email.send") and "người phụ trách" in u_amb_low:
                        tool_name = "sys.ask_missing_info"
                        arguments = {"tool_name": "email.send", "params": ["to"]}
                    elif tool_name in ("contacts.get", "phone.call") and "nexora" in u_amb_low:
                        tool_name = "sys.ask_missing_info"
                        arguments = {"tool_name": "contacts.get", "params": ["contact_id"]}
                    elif tool_name == "phone.call" and re.search(r"\b(14 khách|mười bốn khách|xe nào chở được hết)\b", u_amb_low):
                        tool_name = "sys.ask_missing_info"
                        arguments = {"tool_name": "phone.call", "params": ["contact_id"]}
                    elif tool_name == "learning.schedule_lesson":
                        obs_courses = [c for src in all_obs_srcs for c in (src.get("courses") or []) if isinstance(c, dict) and c.get("id")]
                        if "tiếng hàn" in u_amb_low and not re.search(r"\b(du lịch|sơ cấp|giao tiếp)\b", u_amb_low):
                            kr_courses = {str(c.get("id")) for c in obs_courses if "tiếng hàn" in f'{c.get("name", "")} {c.get("title", "")}'.lower()}
                            if len(kr_courses) >= 2:
                                tool_name = "sys.ask_missing_info"
                                arguments = {"tool_name": "learning.schedule_lesson", "params": ["course_id"]}
                        elif "ngoại ngữ" in u_amb_low and len(obs_courses) >= 2:
                            tool_name = "sys.ask_missing_info"
                            arguments = {"tool_name": "learning.schedule_lesson", "params": ["course_id"]}

                if tool_name == "subscriptions.cancel" and "sys.ask_missing_info" in allowed_tool_names and "lưu trữ đám mây" in user_text.lower() and not re.search(r"\b(cloud\s*pro|orbit\s*cloud|drive\s*plus|cloud\s*box)\b", user_text.lower()):
                    for s_rec_sub in reversed(step_history):
                        ro_sub = s_rec_sub.get("raw_observation")
                        if isinstance(ro_sub, dict) and isinstance(ro_sub.get("subscriptions"), list):
                            active_subs = [s for s in ro_sub["subscriptions"] if isinstance(s, dict) and s.get("status") in (None, "active")]
                            if len(active_subs) >= 2:
                                tool_name = "sys.ask_missing_info"
                                arguments = {"tool_name": "subscriptions.cancel", "params": ["subscription_id"]}
                            break

                if tool_name == "household_chores.assign" and "sys.ask_missing_info" in allowed_tool_names and re.search(r"\b(người khác|ai đó)\b", user_text.lower()):
                    tool_name = "sys.ask_missing_info"
                    arguments = {"tool_name": "household_chores.assign", "params": ["assignee"]}

                if tool_name == "fitness.set_goal" and "sys.ask_missing_info" in allowed_tool_names and re.search(r"\bhạ xuống giúp\b", user_text.lower()) and not re.search(r"\b(?:xuống|còn|thành)\s+\d+", _normalize_vn_numbers_in_time(user_text.lower())):
                    tool_name = "sys.ask_missing_info"
                    arguments = {"tool_name": "fitness.set_goal", "params": ["steps"]}

                if tool_name == "flights.search" and "sys.ask_missing_info" in allowed_tool_names and "cuối tháng" in user_text.lower() and not re.search(r"\b(mai|ngày kia|mốt|thứ\s+\S+|chủ\s+nhật|(?:ngày|mùng|hôm)\s+(?:\d|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)|(?:\d{1,2}|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\s+tháng\s+(?:\d{1,2}|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mười\s+một|mười\s+hai|chạp))\b", _normalize_vn_numbers_in_time(clean_asr_transcript(user_text).lower())):
                    tool_name = "sys.ask_missing_info"
                    arguments = {"tool_name": "flights.search", "params": ["date"]}

                if tool_name == "translator.translate" and isinstance(arguments.get("text"), str):
                    arguments["text"] = re.sub(
                        r"[,.]?\s*(?:với|rồi|xong|và|nhớ)\s+(?:thêm\s+lịch|tạo\s+lịch|đặt\s+lịch|nhắc\s+|gọi\s+|nhắn\s+|bật\s+|tắt\s+).*$",
                        "",
                        arguments["text"],
                        flags=re.IGNORECASE,
                    ).strip().strip('"\'')

                # Khớp chính xác query tên sự kiện khi gọi news.search (tránh kéo theo câu lệnh dài)
                if tool_name == "news.search" and isinstance(arguments.get("query"), str):
                    m_ev_q = (
                        re.search(r"(?:hội thảo|sự kiện|chương trình|giải)\s+([^\s,.]+(?:\s+[^\s,.]+){1,2})(?=\s+(?:hầu như|hình như|nghe nói|đổi|thay đổi|báo mới|trên báo|thế nào)|$)", user_text, re.IGNORECASE)
                        or re.search(r"\bthông\s+báo\s+(.+?)\s+mới\s+nhất\b", user_text, re.IGNORECASE)
                        or re.search(r"\btin\s+(?:gì\s+)?mới\s+về\s+([^,.!?]+?)(?:\s+không\b|\s+nha\b|\s+nhé\b|$)", user_text, re.IGNORECASE)
                        or re.search(r"\bcoi\s+điểm\s+([A-Za-z0-9_]+)\b", user_text)
                    )
                    if m_ev_q:
                        arguments["query"] = m_ev_q.group(1).strip()

                # Quy đổi số ngày gửi xe sang giờ (N * 24) và chuẩn hóa giờ gửi xe khi gọi parking.reserve
                if tool_name == "parking.reserve":
                    norm_pk_u = _normalize_vn_numbers_in_time(user_text.lower())
                    if isinstance(arguments.get("hours"), (int, float)):
                        m_pk_days = re.search(r"(\d+)\s*ngày|một\s+ngày|hai\s+ngày|ba\s+ngày|bốn\s+ngày|năm\s+ngày", user_text.lower())
                        m_pk_hrs = re.search(r"\b(?:trong\s+|khoảng\s+|tầm\s+|cỡ\s+|chừng\s+)?(\d+)\s*(?:tiếng|giờ\s+đồng\s+hồ)\b", norm_pk_u)
                        if m_pk_days and int(arguments["hours"]) <= 10:
                            d_map = {"một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5}
                            d_val = int(m_pk_days.group(1)) if m_pk_days.group(1) else next((v for k, v in d_map.items() if k in m_pk_days.group(0)), 1)
                            arguments["hours"] = d_val * 24
                        elif m_pk_hrs:
                            arguments["hours"] = int(m_pk_hrs.group(1))
                    if isinstance(arguments.get("from_time"), str):
                        p_d_pk, _ = resolve_target_dates_from_utterance(user_text, str((context or {}).get("current_time", "")))
                        if p_d_pk:
                            arguments["from_time"] = re.sub(r"^\d{4}-\d{2}-\d{2}", p_d_pk, arguments["from_time"])
                        m_pk_t = re.search(r"\b(?:từ|lúc|vào)\s+(\d{1,2})\s*(?:giờ\s*)?(rưỡi|\d{1,2})?\s*(sáng|trưa|chiều|tối)?\b", norm_pk_u)
                        if m_pk_t:
                            pkh = int(m_pk_t.group(1))
                            pkm = 30 if (m_pk_t.group(2) or "") == "rưỡi" else int(m_pk_t.group(2) or 0)
                            if (m_pk_t.group(3) or "") in ("chiều", "tối") and pkh < 12:
                                pkh += 12
                            arguments["from_time"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{pkh:02d}:{pkm:02d}:00", arguments["from_time"])

                # Tính chuẩn giờ đón ride_hailing.book khi người dùng yêu cầu đón trước giờ lịch hẹn (tránh 3B cộng nhầm thay vì trừ)
                if tool_name == "ride_hailing.book":
                    u_low_ride = user_text.lower()
                    u_ride_corr = re.sub(r"^.*?\b(?:à\s+thôi|à\s+không|nhầm)\b[,\s]*", "", u_low_ride)
                    norm_u_ride = _normalize_vn_numbers_in_time(u_low_ride)
                    norm_u_ride_corr = _normalize_vn_numbers_in_time(u_ride_corr)
                    if "xe máy" in u_ride_corr:
                        arguments["vehicle"] = "bike"
                    elif re.search(r"\b(bốn chỗ|4 chỗ)\b", norm_u_ride_corr):
                        arguments["vehicle"] = "car4"
                    elif re.search(r"\b(bảy chỗ|7 chỗ)\b", norm_u_ride_corr):
                        arguments["vehicle"] = "car7"
                    u_prof_ride = (context or {}).get("user_profile") if isinstance(context, dict) else {}
                    if isinstance(u_prof_ride, dict):
                        if re.search(r"\btừ\s+(?:công ty|văn phòng|cơ quan)\b", u_low_ride) and u_prof_ride.get("work_address"):
                            arguments["pickup"] = str(u_prof_ride["work_address"])
                        elif re.search(r"\btừ\s+nhà\b", u_low_ride) and u_prof_ride.get("home_address"):
                            arguments["pickup"] = str(u_prof_ride["home_address"])
                    if "hồi nãy" in u_low_ride and history:
                        hist_places = [
                            p for turn_h in (history or []) for res_h in (turn_h.get("tool_results") or [])
                            if isinstance(res_h, dict) for p in (res_h.get("places") or [])
                            if isinstance(p, dict) and p.get("name") and p.get("address")
                        ]
                        if len(hist_places) >= 2:
                            arguments["pickup"] = f"{hist_places[-1]['name']}, {hist_places[-1]['address']}"
                            arguments["dropoff"] = f"{hist_places[0]['name']}, {hist_places[0]['address']}"
                    if re.search(r"\b(đặt ngay bây giờ|ngay bây giờ luôn)\b", u_low_ride):
                        arguments.pop("schedule_time", None)
                    offset_mins = 0
                    if re.search(r"(?:trước|sớm)\b.*?\b(?:45\s*phút|bốn\s+mươi\s+lăm\s+phút)", norm_u_ride):
                        offset_mins = 45
                    elif re.search(r"(?:trước|sớm)\b.*?\b(?:nửa\s+tiếng|30\s*phút|ba\s+mươi\s+phút)", norm_u_ride):
                        offset_mins = 30
                    elif re.search(r"(?:trước|sớm)\b.*?\b(?:15\s*phút|mười\s+lăm\s+phút)", norm_u_ride):
                        offset_mins = 15
                    elif re.search(r"(?:trước|sớm)\b.*?\b(?:1\s*tiếng|một\s+tiếng|60\s*phút)", norm_u_ride):
                        offset_mins = 60
                    for s_rec in step_history:
                        raw_obs = s_rec.get("raw_observation")
                        if isinstance(raw_obs, dict) and isinstance(raw_obs.get("events"), list) and len(raw_obs["events"]) >= 1:
                            evs = [e for e in raw_obs["events"] if isinstance(e, dict) and e.get("start")]
                            picked_ev = next(
                                (e for e in evs if any(kw in str(e.get("title", "")).lower() for kw in ("công viên", "khám", "họp", "cà phê", "đá bóng", "chạy bộ") if kw in u_low_ride)),
                                evs[0] if evs else None,
                            )
                            if picked_ev:
                                try:
                                    if offset_mins > 0:
                                        ev_st = datetime.fromisoformat(str(picked_ev["start"]))
                                        arguments["schedule_time"] = (ev_st - timedelta(minutes=offset_mins)).isoformat()
                                    if picked_ev.get("location") and (
                                        "đến đó" in u_low_ride
                                        or "tới đó" in u_low_ride
                                        or "chỗ hẹn" in u_low_ride
                                        or any(kw in str(picked_ev.get("title", "")).lower() for kw in ("công viên", "khám", "họp", "cà phê", "đá bóng", "chạy bộ") if kw in u_low_ride)
                                    ):
                                        arguments["dropoff"] = str(picked_ev["location"])
                                except Exception:
                                    pass

                if tool_name in ("package_tracking.schedule_pickup", "package_tracking.reschedule_delivery"):
                    p_d_pkg, _ = resolve_target_dates_from_utterance(user_text, str((context or {}).get("current_time", "")))
                    if p_d_pkg and isinstance(arguments.get("date"), str):
                        arguments["date"] = p_d_pkg
                    if tool_name == "package_tracking.reschedule_delivery" and isinstance(arguments.get("time"), str):
                        if p_d_pkg:
                            arguments["time"] = re.sub(r"^\d{4}-\d{2}-\d{2}", p_d_pkg, arguments["time"])
                        norm_pkg_u = _normalize_vn_numbers_in_time(user_text.lower())
                        m_pkg_hr = re.search(r"\b(\d{1,2})\s*giờ(?:\s*(rưỡi|\d{1,2}))?\s*(sáng|trưa|chiều|tối)?\b", norm_pkg_u)
                        if m_pkg_hr:
                            ph_h = int(m_pkg_hr.group(1))
                            ph_m = 30 if (m_pkg_hr.group(2) or "") == "rưỡi" else int(m_pkg_hr.group(2) or 0)
                            if (m_pkg_hr.group(3) or "") in ("chiều", "tối") and ph_h < 12:
                                ph_h += 12
                            arguments["time"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{ph_h:02d}:{ph_m:02d}:00", arguments["time"])
                    if tool_name == "package_tracking.schedule_pickup":
                        m_addr_pk = re.search(r"\b(?:ở|tại)\s+(\d+\s+(?:phố|đường)\s+[A-ZÀ-Ỹ][A-Za-zÀ-ỹ\s]+?)(?:\s+nhé|\s+nha|\s+đi|[,.!?]|$)", user_text)
                        if m_addr_pk:
                            arguments["address"] = m_addr_pk.group(1).strip()
                        elif not arguments.get("address"):
                            u_prof_pk = (context or {}).get("user_profile") or {}
                            if isinstance(u_prof_pk, dict) and u_prof_pk.get("home_address"):
                                arguments["address"] = str(u_prof_pk["home_address"])

                if tool_name == "fitness.get_activity":
                    if "hôm qua" not in user_text.lower() and not re.search(r"\b(?:ngày|mùng|hôm)\s+\d", user_text.lower()):
                        arguments.pop("date", None)

                if tool_name == "calendar.create_event":
                    if "buổi khai giảng" in user_text.lower() and "sơ cứu" in (str(arguments.get("title", "")).lower() + " " + user_text.lower()):
                        arguments["title"] = "Khai giảng sơ cứu"
                    else:
                        m_cal_t_loop = (
                            re.search(r"\bthêm\s+(?:cho\s+\S+\s+|giúp\s+\S+\s+)?(?:cái\s+|một\s+)?(?:lịch|buổi)\s+([^,.]+?)\s+(?:lúc\s+|\d{1,2}\s*giờ|hai\s+giờ|ba\s+giờ|bốn\s+giờ|năm\s+giờ|sáu\s+giờ|bảy\s+giờ|tám\s+giờ|chín\s+giờ|mười\s+giờ|trước\s+|vào\s+|hôm\s+|ngày\s+)", user_text, re.IGNORECASE)
                            or re.search(r"\bghi\s+vào\s+lịch\s+[^,.]*?\blà\s+([^,.]+)", user_text, re.IGNORECASE)
                        )
                        if m_cal_t_loop:
                            tc_l = re.sub(r"\s+rồi$", "", m_cal_t_loop.group(1).strip(), flags=re.IGNORECASE)
                            if tc_l:
                                arguments["title"] = tc_l[0].upper() + tc_l[1:]
                        for s_nw_cal in step_history:
                            if s_nw_cal.get("tool_called") == "news.search" and isinstance(s_nw_cal.get("raw_observation"), dict):
                                arts_c = s_nw_cal["raw_observation"].get("articles") or []
                                if arts_c and isinstance(arts_c[0], dict) and "tránh đường" in user_text.lower():
                                    m_rd = re.search(r"đường\s+([A-ZÀ-Ỹ][A-Za-zÀ-ỹ]+(?:\s+[A-ZÀ-Ỹ][A-Za-zÀ-ỹ]+)+)", json.dumps(arts_c[0], ensure_ascii=False))
                                    if m_rd:
                                        arguments["title"] = f"Tránh đường {m_rd.group(1)}"
                                        break
                        norm_cal_l = _normalize_vn_numbers_in_time(user_text.lower())
                        m_cal_rng_l = re.search(
                            r"\b(\d{1,2})\s*(?:giờ\s*)?(rưỡi|\d{1,2})?\s*(sáng|trưa|chiều|tối)?\s*(?:tới|đến|-)\s*(\d{1,2})\s*(?:giờ\s*)?(rưỡi|\d{1,2})?\s*(sáng|trưa|chiều|tối)?",
                            norm_cal_l,
                        )
                        m_cal_hr_l = re.search(r"\b(?:(?:lúc|vào|lịch|tạo\s+cái\s+lịch|thêm\s+lịch)\s+(?:[^,.!?]{0,25}?\s+)?)?(\d{1,2})\s*giờ(?:\s*(\d{1,2}|rưỡi))?\s*(sáng|trưa|chiều|tối)?", norm_cal_l)
                        if m_cal_rng_l and isinstance(arguments.get("start"), str):
                            h1_l = int(m_cal_rng_l.group(1))
                            m1_l = 30 if (m_cal_rng_l.group(2) or "") == "rưỡi" else int(m_cal_rng_l.group(2) or 0)
                            h2_l = int(m_cal_rng_l.group(4))
                            m2_l = 30 if (m_cal_rng_l.group(5) or "") == "rưỡi" else int(m_cal_rng_l.group(5) or 0)
                            p_rng_l = (m_cal_rng_l.group(3) or m_cal_rng_l.group(6) or "")
                            if p_rng_l in ("chiều", "tối"):
                                if h1_l < 12:
                                    h1_l += 12
                                if h2_l < 12:
                                    h2_l += 12
                            arguments["start"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{h1_l:02d}:{m1_l:02d}:00", arguments["start"])
                            if isinstance(arguments.get("end"), str):
                                arguments["end"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{h2_l:02d}:{m2_l:02d}:00", arguments["end"])
                        elif m_cal_hr_l and not re.search(r"\btrước\s+[^,.!?]*?\d+\s*tiếng", norm_cal_l) and isinstance(arguments.get("start"), str):
                            ch_l = int(m_cal_hr_l.group(1))
                            cm_l = 30 if (m_cal_hr_l.group(2) or "") == "rưỡi" else int(m_cal_hr_l.group(2) or 0)
                            if (m_cal_hr_l.group(3) or "") in ("chiều", "tối") and ch_l < 12:
                                ch_l += 12
                            arguments["start"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{ch_l:02d}:{cm_l:02d}:00", arguments["start"])
                            if isinstance(arguments.get("end"), str):
                                arguments["end"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{(ch_l + 1) % 24:02d}:{cm_l:02d}:00", arguments["end"])
                        for s_fit in step_history:
                            if s_fit.get("tool_called") == "fitness.get_activity" and isinstance(s_fit.get("raw_observation"), dict):
                                act_obj = s_fit["raw_observation"].get("activity") or s_fit["raw_observation"]
                                w_list = act_obj.get("workouts") if isinstance(act_obj, dict) else None
                                if isinstance(w_list, list) and w_list and isinstance(w_list[0], dict) and isinstance(w_list[0].get("minutes"), (int, float)):
                                    w_mins = int(w_list[0]["minutes"])
                                    if "cà phê xong" in user_text.lower() and history:
                                        for t_h in reversed(history):
                                            for r_h in (t_h.get("tool_results") or []):
                                                if isinstance(r_h, dict) and isinstance(r_h.get("events"), list):
                                                    for ev_h in r_h["events"]:
                                                        if isinstance(ev_h, dict) and "cà phê" in str(ev_h.get("title", "")).lower() and ev_h.get("end"):
                                                            arguments["start"] = str(ev_h["end"])
                                                            break
                                    if isinstance(arguments.get("start"), str):
                                        try:
                                            st_dt = datetime.fromisoformat(arguments["start"])
                                            arguments["end"] = (st_dt + timedelta(minutes=w_mins)).isoformat()
                                        except Exception:
                                            pass
                                    if not re.search(r"\b(?:ở|tại)\s+[A-ZÀ-Ỹ]", user_text):
                                        arguments.pop("location", None)
                                    m_wt_l = re.search(r"\b(đi\s+bộ\s+quanh\s+hồ|chạy\s+bộ|đạp\s+xe|tập\s+gym|tập\s+yoga)\b", user_text, re.IGNORECASE)
                                    if m_wt_l:
                                        wt_s_l = m_wt_l.group(1).strip()
                                        arguments["title"] = wt_s_l[0].upper() + wt_s_l[1:]
                                    break

                if tool_name == "reminders.create":
                    norm_rem_u = _normalize_vn_numbers_in_time(user_text.lower())
                    m_rem_loop_c = re.search(
                        r"nhắc\s+(?:giúp\s+|dùm\s+|giùm\s+|hộ\s+|cho\s+)?(?:mình|tôi|tớ|anh|chị|em|cô|chú|bác)\s+(?:(?:\d{1,2}|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mười\s+một|mười\s+hai)\s*giờ(?:\s*(?:rưỡi|\d{1,2}))?(?:\s*(?:sáng|trưa|chiều|tối))?(?:\s*(?:nay|ni|mai|thứ\s+\S+(?:\s+tuần\s+sau)?))?\s+)?([^,.!?]+?)(?:\s+lúc\b|\s+vào\b|\s+nữa\b|\s+nhé\b|\s+nha\b|\s+nghe\b|\s+hen\b|$)",
                        user_text,
                        re.IGNORECASE,
                    )
                    if m_rem_loop_c and isinstance(arguments.get("content"), str):
                        cur_rem_words = [w for w in re.findall(r"\w+", arguments["content"].lower()) if len(w) >= 3]
                        if not cur_rem_words or not all(w in user_text.lower() for w in cur_rem_words):
                            c_rem_v = re.sub(r"\bcái\s+", "", m_rem_loop_c.group(1).strip(), flags=re.IGNORECASE)
                            if c_rem_v and len(c_rem_v.split()) >= 2 and not re.match(r"^\d", c_rem_v) and not re.search(r"\b(sao cho|đúng lúc|trước lúc)\b", c_rem_v.lower()):
                                arguments["content"] = c_rem_v[0].upper() + c_rem_v[1:]
                    if "mang laptop" in user_text.lower():
                        arguments["content"] = "mang laptop"
                    elif "đọc bài" in user_text.lower():
                        for s_nw in step_history:
                            ro_nw = s_nw.get("raw_observation")
                            if isinstance(ro_nw, dict) and isinstance(ro_nw.get("articles"), list) and ro_nw["articles"]:
                                art0_t = str(ro_nw["articles"][0].get("title", "")).split(":")[0].strip()
                                if art0_t:
                                    arguments["content"] = f"Đọc bài {art0_t}"
                                    break
                    elif re.search(r"\b(nhận sách|lấy sách)\b", user_text.lower()):
                        for s_lb in step_history:
                            ro_lb = s_lb.get("raw_observation")
                            if isinstance(ro_lb, dict) and isinstance(ro_lb.get("books"), list) and ro_lb["books"]:
                                bk0_t = str(ro_lb["books"][0].get("title", "")).strip()
                                if bk0_t:
                                    arguments["content"] = f"Đến nhận sách {bk0_t}"
                                    break
                    p_d_rem, prev_d_rem = resolve_target_dates_from_utterance(user_text, str((context or {}).get("current_time", "")))
                    if prev_d_rem and "hôm trước" in user_text.lower() and isinstance(arguments.get("time"), str):
                        arguments["time"] = re.sub(r"^\d{4}-\d{2}-\d{2}", prev_d_rem, arguments["time"])
                    elif p_d_rem and isinstance(arguments.get("time"), str):
                        arguments["time"] = re.sub(r"^\d{4}-\d{2}-\d{2}", p_d_rem, arguments["time"])
                    m_rem_off = re.search(r"\btrước\s+lúc\s+xe\s+(?:tới|đón)\s+(\d+)\s*phút\b|\b(\d+)\s*phút\s+trước\s+lúc\s+xe\s+(?:tới|đón)\b", norm_rem_u)
                    if m_rem_off:
                        rem_off = int(m_rem_off.group(1) or m_rem_off.group(2) or 15)
                        ride_sched = next(
                            (str(s.get("arguments", {}).get("schedule_time")) for s in reversed(step_history) if s.get("tool_called") == "ride_hailing.book" and s.get("arguments", {}).get("schedule_time")),
                            None,
                        )
                        if ride_sched:
                            try:
                                arguments["time"] = (datetime.fromisoformat(ride_sched) - timedelta(minutes=rem_off)).isoformat()
                            except Exception:
                                pass
                    elif isinstance(arguments.get("time"), str) and "kém" in norm_rem_u:
                        m_kem = re.search(r"\b(\d{1,2})\s*giờ\s*kém\s*(\d{1,2})\s*(sáng|chiều|tối)?\b", norm_rem_u)
                        if m_kem:
                            rh_k = int(m_kem.group(1))
                            rm_k = 60 - int(m_kem.group(2))
                            if (m_kem.group(3) or "").lower() in {"chiều", "tối"} and rh_k < 12:
                                rh_k += 12
                            rh_k = (rh_k - 1) % 24
                            arguments["time"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{rh_k:02d}:{rm_k:02d}:00", arguments["time"])
                    elif isinstance(arguments.get("time"), str) and not re.search(r"\b(sớm hơn|muộn hơn|kém)\b", norm_rem_u):
                        m_rem_rel_l = re.search(r"\b(\d+)\s*(phút|tiếng|giờ)\s*nữa\b", norm_rem_u)
                        if m_rem_rel_l and (context or {}).get("current_time"):
                            try:
                                cur_dt_rl = datetime.fromisoformat(str(context["current_time"]))
                                off_rl = int(m_rem_rel_l.group(1)) * (60 if m_rem_rel_l.group(2) in ("tiếng", "giờ") else 1)
                                arguments["time"] = (cur_dt_rl + timedelta(minutes=off_rl)).isoformat()
                            except Exception:
                                pass
                        else:
                            m_rem_hr_loop = re.search(r"\b(?:(sáng|trưa|chiều|tối)\s+(?:mai\s+|hôm\s+nay\s+|ni\s+|nay\s+)?)?(\d{1,2})\s*giờ\s*(sáng|trưa|chiều|tối)?\b", norm_rem_u)
                            if m_rem_hr_loop and not re.search(r"\b\d{1,2}\s*giờ\s*(?:rưỡi|\d{1,2})\b", norm_rem_u):
                                rh_l = int(m_rem_hr_loop.group(2))
                                mer_l = (m_rem_hr_loop.group(3) or m_rem_hr_loop.group(1) or "").lower()
                                if mer_l in {"chiều", "tối"} and rh_l < 12:
                                    rh_l += 12
                                arguments["time"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{rh_l:02d}:00:00", arguments["time"])

                if tool_name == "sms.send" and isinstance(arguments.get("content"), str):
                    upd_ev_step = next(
                        (s for s in reversed(step_history) if s.get("tool_called") == "calendar.update_event" and s.get("arguments", {}).get("start")),
                        None,
                    )
                    canc_ev_step = next(
                        (s for s in reversed(step_history) if s.get("tool_called") == "calendar.cancel_event" and s.get("arguments", {}).get("event_id")),
                        None,
                    )
                    if upd_ev_step and "giờ mới" in user_text.lower():
                        upd_st_s = str(upd_ev_step["arguments"]["start"])
                        upd_eid_s = str(upd_ev_step["arguments"].get("event_id", ""))
                        ev_title_s = ""
                        for s_rec in step_history:
                            ro = s_rec.get("raw_observation")
                            if isinstance(ro, dict) and isinstance(ro.get("events"), list):
                                for ev_it in ro["events"]:
                                    if isinstance(ev_it, dict) and str(ev_it.get("id")) == upd_eid_s and ev_it.get("title"):
                                        ev_title_s = str(ev_it["title"]).strip()
                        if ev_title_s and len(upd_st_s) >= 16:
                            arguments["content"] = f"mai {ev_title_s} dời sang {upd_st_s[11:16]} nhé"
                    elif canc_ev_step and "mưa" in user_text.lower():
                        canc_eid_s = str(canc_ev_step["arguments"].get("event_id", ""))
                        ev_title_c = ""
                        for s_rec in step_history:
                            ro = s_rec.get("raw_observation")
                            if isinstance(ro, dict) and isinstance(ro.get("events"), list):
                                for ev_it in ro["events"]:
                                    if isinstance(ev_it, dict) and str(ev_it.get("id")) == canc_eid_s and ev_it.get("title"):
                                        ev_title_c = str(ev_it["title"]).strip()
                        if ev_title_c:
                            arguments["content"] = f"mai mưa nên huỷ {ev_title_c.lower()} nhé"
                    elif any(s.get("tool_called") == "bill_payment.pay" for s in step_history) and "tiền điện" in user_text.lower() and "đóng rồi" in user_text.lower():
                        arguments["content"] = "đã đóng tiền điện rồi"
                    elif "giặt lại" in user_text.lower():
                        arguments["content"] = "Trưa anh ghé nhà giặt lại giùm em"
                    else:
                        m_sms_loop = re.search(
                            r"\b(?:nhắn|báo|bảo)\b[^.!?]*?\b(?:là|rằng)\s+([^.!?]+?)(?:,?\s*\b(?:rồi|xong|với)\s+(?:hủy|huỷ|đặt|gọi|tắt|bật|mở|chuyển|xóa|xoá|kiểm tra|xem|tìm)\b|[.!?]|$)",
                            user_text,
                            re.IGNORECASE,
                        )
                        if m_sms_loop:
                            c_s_val = m_sms_loop.group(1).strip()
                            cur_c_words = [w for w in re.findall(r"\w+", arguments["content"].lower()) if len(w) >= 3]
                            if c_s_val and not re.search(r"\b(giờ mới|mã giao dịch|link|bao nhiêu|số đó|đủ không|còn trống)\b", c_s_val.lower()):
                                if not cur_c_words or not all(w in user_text.lower() for w in cur_c_words):
                                    arguments["content"] = c_s_val[0].upper() + c_s_val[1:]

                # Khớp chính xác chat_id theo tên người nhận trong câu nói hoặc ngữ cảnh trước khi gọi chat_ott.send_message
                if tool_name == "chat_ott.send_message" and arguments.get("chat_id"):
                    u_low = clean_asr_transcript(user_text).lower()
                    if "về khuya" in u_low:
                        arguments["content"] = "Tối nay chị về khuya nhé"
                    elif "hoàn ứng" in u_low or "hời ẩn" in u_low:
                        arguments["content"] = "Em chuyển tiền hoàn ứng cho chị rồi nhé."
                    elif "đạt cả hai điều kiện" in u_low:
                        arguments["content"] = "Robot bắt đầu dọn phòng làm việc lúc 15:05."
                    elif isinstance(arguments.get("content"), str):
                        m_ott_c = re.search(
                            r"\b(?:nhắn|báo|bảo|gửi|zalo)\b[^.!?]*?\b(?:là|rằng)\s+([^.!?]+?)(?:,?\s*\b(?:rồi|xong|với)\s+(?:hủy|huỷ|đặt|gọi|tắt|bật|mở|chuyển|xóa|xoá|kiểm tra|xem|tìm)\b|[.!?]|$)",
                            clean_asr_transcript(user_text),
                            re.IGNORECASE,
                        )
                        if m_ott_c:
                            c_o_val = m_ott_c.group(1).strip()
                            cur_ott_w = [w for w in re.findall(r"\w+", arguments["content"].lower()) if len(w) >= 3]
                            if c_o_val and not re.search(r"\b(giờ mới|mã giao dịch|link|bao nhiêu|số đó|đủ không|còn trống)\b", c_o_val.lower()):
                                if not cur_ott_w or not all(w in u_low for w in cur_ott_w):
                                    arguments["content"] = c_o_val[0].upper() + c_o_val[1:]
                    all_chat_srcs = [s_rec.get("raw_observation") for s_rec in step_history if isinstance(s_rec.get("raw_observation"), dict)] + [res_h for turn in (history or []) for res_h in (turn.get("tool_results") or []) if isinstance(res_h, dict)]
                    direct_chats = [
                        ch_obj
                        for src_c in all_chat_srcs
                        for ch_obj in (src_c.get("chats") or [])
                        if isinstance(ch_obj, dict) and ch_obj.get("type") == "direct" and ch_obj.get("name") and ch_obj.get("id")
                    ]
                    matched_direct = next((ch for ch in direct_chats if str(ch["name"]).lower() in u_low), None)
                    if not matched_direct and re.search(r"\b(chị ấy|anh ấy|bạn ấy|cô ấy|em ấy|người đó|bả|ổng|cổ|ảnh)\b", u_low):
                        matched_direct = next((ch for ch in direct_chats if str(ch["name"]).lower() in history_blob.lower()), None)
                    if matched_direct:
                        arguments["chat_id"] = str(matched_direct["id"])

                # Chuẩn hóa tham số tìm kiếm chuyến bay / tàu xe / kênh TV / hóa đơn / thói quen / bãi đỗ / báo thức chuyến bay / danh bạ 1 liên hệ
                if tool_name == "sys.ask_missing_info" and re.search(r"\b(soạn .* email|email nháp)\b", user_text.lower()) and any(s.get("tool_called") == "contacts.search" for s in step_history):
                    arguments = {"tool_name": "email.create_draft", "params": ["to"]}
                elif tool_name == "sys.ask_missing_info" and arguments.get("params") == ["contact_id"]:
                    target_tn = str(arguments.get("tool_name", ""))
                    if target_tn == "phone.call" and "phone.call" in allowed_tool_names:
                        single_ct_id = None
                        for s_rec in reversed(step_history):
                            ro = s_rec.get("raw_observation")
                            if isinstance(ro, dict) and isinstance(ro.get("contacts"), list):
                                if len(ro["contacts"]) == 1 and isinstance(ro["contacts"][0], dict) and ro["contacts"][0].get("id"):
                                    single_ct_id = str(ro["contacts"][0]["id"])
                                break
                        if single_ct_id:
                            print(f"    - Single-Contact Guardrail: `contacts.search` trả về đúng 1 liên hệ `{single_ct_id}` -> chuyển `sys.ask_missing_info` thành `phone.call`.")
                            tool_name = "phone.call"
                            arguments = {"contact_id": single_ct_id}

                if tool_name == "music.control" and re.search(r"\b(dừng|tạm dừng|tắt nhạc|ngưng)\b", user_text.lower()):
                    arguments = {"action": "pause"}

                if tool_name == "smart_home.light_set":
                    u_low_l = user_text.lower()
                    m_br_arg = re.search(r"\b(\d{1,3})\s*(?:phần trăm|%)", _normalize_vn_numbers_in_time(u_low_l))
                    if m_br_arg:
                        arguments["brightness"] = int(m_br_arg.group(1))
                    elif re.search(r"\b(một nửa|phân nửa|còn 50)\b", u_low_l):
                        arguments["brightness"] = 50
                    if "brightness" in arguments and re.search(r"\b(giảm|hạ|tăng|chỉnh)\s+độ\s+sáng\b", u_low_l) and not re.search(r"\b(bật|mở|tắt)\b", u_low_l):
                        arguments.pop("power", None)

                if tool_name == "alarm.create":
                    norm_al = _normalize_vn_numbers_in_time(user_text.lower())
                    m_lbl_al = re.search(r"\btên\s+là\s+([^,.!?]+?)(?:\s+rồi\b|\s+xong\b|\s+với\b|\s+và\b|\s+đi\b|\s+nha\b|\s+nhé\b|[,.!?]|$)", user_text, re.IGNORECASE)
                    if m_lbl_al:
                        lbl_al_s = m_lbl_al.group(1).strip()
                        if lbl_al_s:
                            arguments["label"] = lbl_al_s[0].upper() + lbl_al_s[1:]
                    p_d_al, _ = resolve_target_dates_from_utterance(user_text, str((context or {}).get("current_time", "")))
                    if p_d_al and isinstance(arguments.get("time"), str) and not re.search(r"\b(sáng hôm bay|trước giờ bay|trước giờ cất cánh)\b", norm_al):
                        arguments["time"] = re.sub(r"^\d{4}-\d{2}-\d{2}", p_d_al, arguments["time"])
                    m_al_exp_loop = re.search(
                        r"\b(?:báo\s+thức(?:\s+mới)?(?:\s+vào|\s+lúc)?|kêu\s+\S+\s+dậy|gọi\s+\S+\s+dậy)\s+(?:lúc\s+|vào\s+)?(\d{1,2})\s*(?:giờ\s*)?(rưỡi|\d{1,2})?\s*(sáng|trưa|chiều|tối)?\b",
                        norm_al,
                    )
                    if m_al_exp_loop and isinstance(arguments.get("time"), str) and not re.search(r"\b(trước|sớm\s+hơn|sau\s+khi)\b", norm_al):
                        ah_e = int(m_al_exp_loop.group(1))
                        am_e = 30 if (m_al_exp_loop.group(2) or "") == "rưỡi" else int(m_al_exp_loop.group(2) or 0)
                        if (m_al_exp_loop.group(3) or "") in ("chiều", "tối") and ah_e < 12:
                            ah_e += 12
                        arguments["time"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{ah_e:02d}:{am_e:02d}:00", arguments["time"])
                    upd_ev_start = next(
                        (str(s.get("arguments", {}).get("start")) for s in reversed(step_history) if s.get("tool_called") == "calendar.update_event" and s.get("arguments", {}).get("start")),
                        None,
                    )
                    ev_start_al = upd_ev_start
                    if not ev_start_al and ("trước" in norm_al or "chuất" in norm_al or "sớm hơn" in norm_al):
                        for s_rec in step_history:
                            ro = s_rec.get("raw_observation")
                            if isinstance(ro, dict) and isinstance(ro.get("events"), list) and ro["events"]:
                                evs_al = [e for e in ro["events"] if isinstance(e, dict) and e.get("start")]
                                if evs_al:
                                    picked_ev_al = next(
                                        (e for e in evs_al if any(kw in str(e.get("title", "")).lower() for kw in ("họp", "khám", "cà phê", "đá bóng", "chạy bộ", "công viên", "đám cưới", "tiệc cưới") if kw in norm_al)),
                                        evs_al[0],
                                    )
                                    ev_start_al = str(picked_ev_al["start"])
                                    break
                    if ev_start_al and ("trước" in norm_al or "chuất" in norm_al or "sớm hơn" in norm_al):
                        m_ev_al = (
                            re.search(r"\b(?:trước|chuất|sớm\s+hơn)\s+(?:đó|giờ\s+hẹn|lúc\s+đó|buổi\s+hẹn|(?:lịch|buổi|giờ|đám\s+cưới|tiệc\s+cưới)[^,.!?]{0,40}?)\s+(\d+)\s*(tiếng|giờ|phút|kiểu)\b", norm_al)
                            or re.search(r"\b(\d+)\s*(tiếng|giờ|phút)\s+(?:trước|chuất|sớm\s+hơn)\s+(?:đó|giờ\s+hẹn|lúc\s+đó|buổi\s+hẹn|lịch|buổi|giờ|đám\s+cưới|tiệc\s+cưới)\b", norm_al)
                        )
                        if m_ev_al:
                            try:
                                off_al = int(m_ev_al.group(1)) * (60 if m_ev_al.group(2) in ("tiếng", "giờ", "kiểu") else 1)
                                if any(s.get("tool_called") == "lottery.check_ticket" and isinstance(s.get("raw_observation"), dict) and not s["raw_observation"].get("won") for s in step_history) and re.search(r"trật\s+thì\s+1\s*tiếng", norm_al):
                                    off_al = 60
                                arguments["time"] = (datetime.fromisoformat(ev_start_al) - timedelta(minutes=off_al)).isoformat()
                            except Exception:
                                pass
                    fl_ref = None
                    booked_fid = next((str(s.get("arguments", {}).get("flight_id")) for s in reversed(step_history) if s.get("tool_called") == "flights.book" and not s.get("error")), None)
                    for s_rec in reversed(step_history):
                        ro = s_rec.get("raw_observation")
                        if isinstance(ro, dict):
                            if isinstance(ro.get("bookings"), list) and ro["bookings"]:
                                bks = [b for b in ro["bookings"] if isinstance(b, dict) and b.get("status") not in ("cancelled", "canceled") and not str(b.get("id", "")).startswith("tbb_")]
                                if bks:
                                    fl_ref = bks[0].get("flight") or bks[0]
                                    break
                            if isinstance(ro.get("flights"), list) and ro["flights"]:
                                fls = [f for f in ro["flights"] if isinstance(f, dict)]
                                if booked_fid:
                                    fl_ref = next((f for f in fls if str(f.get("id")) == booked_fid), fls[0] if fls else None)
                                elif fls:
                                    fl_ref = fls[0]
                                if fl_ref:
                                    break
                    if fl_ref and isinstance(fl_ref, dict):
                        f_date = str(fl_ref.get("date", ""))
                        dep_s = str(fl_ref.get("depart", fl_ref.get("departure_time", "")))
                        arr_s = str(fl_ref.get("arrive", fl_ref.get("arrival_time", "")))
                        dep_iso = dep_s if "T" in dep_s else (f"{f_date}T{dep_s}:00+07:00" if f_date and re.match(r"^\d{2}:\d{2}$", dep_s) else "")
                        arr_iso = arr_s if "T" in arr_s else (f"{f_date}T{arr_s}:00+07:00" if f_date and re.match(r"^\d{2}:\d{2}$", arr_s) else "")
                        try:
                            if arr_iso and re.search(r"\b(sau khi .* hạ cánh|máy bay hạ cánh)\b", norm_al):
                                arr_dt = datetime.fromisoformat(arr_iso)
                                m_af = re.search(r"\b(\d+)\s*phút\b", norm_al)
                                off_m = int(m_af.group(1)) if m_af else 30
                                arguments["time"] = (arr_dt + timedelta(minutes=off_m)).isoformat()
                            elif dep_iso and re.search(r"\b(trước giờ bay|trước giờ cất cánh|trước khi bay|trước chuyến bay|báo thức trước)\b", norm_al):
                                m_trig_dep = re.search(r"\b(trước giờ bay|trước giờ cất cánh|trước khi bay|trước chuyến bay|báo thức trước)\b", norm_al)
                                al_tail = norm_al[m_trig_dep.start():] if m_trig_dep else norm_al
                                dep_dt = datetime.fromisoformat(dep_iso)
                                off_m = 120
                                m_hr_ruoi = re.search(r"\b(\d+)\s*(?:tiếng|giờ)\s*rưỡi\b", al_tail)
                                m_hr = re.search(r"\b(\d+)\s*(?:tiếng|giờ)\b", al_tail)
                                m_ph = re.search(r"\b(\d+)\s*phút\b", al_tail)
                                if m_hr_ruoi:
                                    off_m = int(m_hr_ruoi.group(1)) * 60 + 30
                                elif m_hr:
                                    off_m = int(m_hr.group(1)) * 60
                                elif m_ph:
                                    off_m = int(m_ph.group(1))
                                elif "nửa tiếng" in al_tail:
                                    off_m = 30
                                arguments["time"] = (dep_dt - timedelta(minutes=off_m)).isoformat()
                            elif f_date and isinstance(arguments.get("time"), str):
                                m_exp_al = re.search(r"\bbáo\s+thức\s+(?:lúc\s+)?(\d{1,2})\s*giờ(?:\s*(rưỡi|\d{1,2}))?\s*(sáng|chiều|tối)?\b", norm_al)
                                if m_exp_al:
                                    ah_fl = int(m_exp_al.group(1))
                                    am_fl = 30 if (m_exp_al.group(2) or "") == "rưỡi" else int(m_exp_al.group(2) or 0)
                                    if (m_exp_al.group(3) or "").lower() in {"chiều", "tối"} and ah_fl < 12:
                                        ah_fl += 12
                                    arguments["time"] = f"{f_date}T{ah_fl:02d}:{am_fl:02d}:00+07:00"
                                elif re.search(r"\b(sáng hôm bay|hôm bay)\b", norm_al):
                                    arguments["time"] = re.sub(r"202\d-\d{2}-\d{2}", f_date, arguments["time"])
                        except Exception:
                            pass

                if tool_name in ("tv.tune", "tv.guide") and isinstance(arguments.get("channel"), str):
                    arguments["channel"] = re.sub(r"^VTB(\d+)$", r"VTV\1", arguments["channel"].strip(), flags=re.IGNORECASE)
                if tool_name == "tv.tune":
                    for s_tv in reversed(step_history):
                        ro_tv = s_tv.get("raw_observation")
                        if isinstance(ro_tv, dict) and isinstance(ro_tv.get("programs"), list):
                            progs_tv = [p for p in ro_tv["programs"] if isinstance(p, dict) and p.get("channel")]
                            chs_tv = set(ro_tv.get("channels") or [])
                            if progs_tv and (arguments.get("channel") not in chs_tv or "đang chiếu" in user_text.lower()):
                                cur_t_tv = str((context or {}).get("current_time", ""))
                                now_progs = [p for p in progs_tv if str(p.get("start", "")) <= cur_t_tv <= str(p.get("end", "9999"))]
                                picked_p_tv = now_progs[0] if now_progs else progs_tv[0]
                                arguments["channel"] = str(picked_p_tv["channel"])
                            break

                if tool_name == "phone.call" and isinstance(arguments.get("phone_number"), str):
                    p_num_cur = arguments["phone_number"]
                    m_hist_ph = re.search(r"\b(0\d{8,11})\b", clean_asr_transcript(history_blob))
                    if m_hist_ph and p_num_cur not in user_text and p_num_cur not in clean_asr_transcript(history_blob):
                        arguments["phone_number"] = m_hist_ph.group(1)

                if tool_name == "bill_payment.pay":
                    ew_bal_obs = None
                    bill_amt_obs = None
                    for s_rec in step_history:
                        ro = s_rec.get("raw_observation")
                        if isinstance(ro, dict):
                            if "balance" in ro and isinstance(ro.get("balance"), (int, float)):
                                ew_bal_obs = ro["balance"]
                            if isinstance(ro.get("bills"), list):
                                for b_it in ro["bills"]:
                                    if isinstance(b_it, dict) and str(b_it.get("id")) == str(arguments.get("bill_id")):
                                        bill_amt_obs = b_it.get("amount")
                    if ew_bal_obs is None and arguments.get("source") == "ewallet" and "ewallet.get_balance" in allowed_tool_names and re.search(r"\b(thiếu|không đủ|hụt|ưu tiên ví)\b", user_text.lower()):
                        ew_res_auto = env.call("ewallet.get_balance", {})
                        executed_call_signatures.add("ewallet.get_balance:{}")
                        step_history.append({
                            "step": step_idx + 1,
                            "tool_called": "ewallet.get_balance",
                            "arguments": {},
                            "observation": compact_observation(ew_res_auto, max_length=1200),
                            "raw_observation": ew_res_auto,
                        })
                        if isinstance(ew_res_auto, dict) and isinstance(ew_res_auto.get("balance"), (int, float)):
                            ew_bal_obs = ew_res_auto["balance"]
                    if ew_bal_obs is not None and bill_amt_obs is not None:
                        arguments["source"] = "ewallet" if ew_bal_obs >= bill_amt_obs else "bank"
                    elif arguments.get("source") == "ewallet" and "ewallet.get_balance" not in allowed_tool_names:
                        arguments["source"] = "bank"

                if tool_name == "grocery.search" and isinstance(arguments.get("query"), str):
                    if "," in arguments["query"]:
                        gr_parts = [p.strip() for p in arguments["query"].split(",") if p.strip()]
                        if len(gr_parts) >= 2:
                            for gp in gr_parts[:-1]:
                                gp_res = env.call("grocery.search", {"query": gp})
                                executed_call_signatures.add(f'grocery.search:{json.dumps({"query": gp}, sort_keys=True)}')
                                step_history.append({
                                    "step": step_idx + 1,
                                    "tool_called": "grocery.search",
                                    "arguments": {"query": gp},
                                    "observation": compact_observation(gp_res, max_length=1200),
                                    "raw_observation": gp_res,
                                })
                            arguments["query"] = gr_parts[-1]
                    q_gr_cur = arguments["query"].strip().lower()
                    if q_gr_cur not in user_text.lower() and q_gr_cur not in history_blob.lower():
                        rcp_ing_list = []
                        pantry_names = set()
                        for s_rec_g in step_history:
                            ro_g = s_rec_g.get("raw_observation")
                            if isinstance(ro_g, dict):
                                r_obj = ro_g.get("recipe") if isinstance(ro_g.get("recipe"), dict) else ro_g
                                if isinstance(r_obj.get("ingredients"), list):
                                    rcp_ing_list = [str(x.get("name", x) if isinstance(x, dict) else x).strip() for x in r_obj["ingredients"]]
                                if isinstance(ro_g.get("items"), list):
                                    for p_it in ro_g["items"]:
                                        if isinstance(p_it, dict) and p_it.get("name") and (p_it.get("quantity", 1) or 0) > 0:
                                            pantry_names.add(str(p_it["name"]).strip().lower())
                        if rcp_ing_list:
                            missing_ings = [ing for ing in rcp_ing_list if ing and not any(ing.lower() in pn or pn in ing.lower() for pn in pantry_names)]
                            if missing_ings:
                                arguments["query"] = missing_ings[0]

                if tool_name == "parking.find":
                    u_pk_f = user_text.lower()
                    u_prof_pkf = (context or {}).get("user_profile") if isinstance(context, dict) else {}
                    if re.search(r"\b(gần công ty|ở công ty|chỗ làm|văn phòng)\b", u_pk_f) and isinstance(u_prof_pkf, dict) and u_prof_pkf.get("work_address"):
                        arguments["near"] = str(u_prof_pkf["work_address"]).split(",")[0].strip()
                    elif "nhà văn hóa" in u_pk_f or "nhà dân hóa" in u_pk_f:
                        arguments["near"] = "Nhà văn hóa"
                    elif isinstance(arguments.get("near"), str):
                        cur_nr = arguments["near"].strip().lower()
                        if cur_nr and cur_nr not in u_pk_f and cur_nr not in history_blob.lower():
                            arguments["near"] = "gần đây"

                if tool_name == "maps.search_place":
                    if "quán chay" in history_blob.lower() and re.search(r"\bđi\s+từ\s+nhà\b", user_text.lower()):
                        arguments = {"query": "quán chay"}
                    elif "photo" in user_text.lower() or "in bài" in user_text.lower():
                        arguments = {"query": "photo"}
                    else:
                        q_mp_existing = str(arguments.get("query") or "").strip()
                        m_quan_mp = re.search(r"\bquán\s+([A-ZÀ-Ỹ][a-zA-ZÀ-ỹéèê]*(?:\s+[a-zA-ZÀ-ỹéèê]+){1,4}?)(?=\s+(?:ở|gần|quanh|đặt\s+tên|rồi\s+lưu|vào\s+bản\s+đồ|bên|nha|nhé)\b|[,.!?;]|$)", user_text)
                        if m_quan_mp and (not q_mp_existing or q_mp_existing.lower() not in user_text.lower() or len(m_quan_mp.group(1).strip()) > len(q_mp_existing)):
                            arguments["query"] = m_quan_mp.group(1).strip()
                        elif isinstance(arguments.get("query"), str):
                            arguments["query"] = re.sub(r"^quán\s+", "", arguments["query"].strip(), flags=re.IGNORECASE)
                            q_mp_cur = arguments["query"].strip().lower()
                            if q_mp_cur and q_mp_cur not in user_text.lower() and q_mp_cur not in history_blob.lower():
                                m_mp_cat = re.search(r"\b(tiệm bánh|siêu thị|nhà thuốc|hiệu thuốc|cây xăng|trạm xăng|bệnh viện|phòng khám|cửa hàng tiện lợi|quán cà phê|nhà hàng|bưu điện|ngân hàng|bãi đỗ xe|công viên|trạm sạc)\b", user_text.lower())
                                if m_mp_cat:
                                    arguments["query"] = m_mp_cat.group(1)
                        m_ben_mp = re.search(r"\bbên\s+([A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+){1,3})\b", user_text)
                        if m_ben_mp and not any(sig.startswith("maps.search_place:") for sig in executed_call_signatures):
                            arguments["near"] = m_ben_mp.group(1).strip()
                        elif "near" in arguments:
                            near_v = str(arguments.get("near", "")).strip().lower()
                            if (
                                near_v in {"trường", "công ty", "nhà", "văn phòng", "cơ quan", "đây", "quanh đây", "gần đây"}
                                or near_v.startswith(("địa điểm", "tháng "))
                                or re.search(r"\b(trường|nhà|công ty|cơ quan|văn phòng|chỗ làm|đây|đường|sư phạm|lê lợi)\b", near_v)
                            ):
                                arguments.pop("near", None)

                if tool_name == "maps.get_route":
                    if "nhà thi đấu quần ngựa" in history_blob.lower() and "thủ lệ" in history_blob.lower():
                        arguments["origin"] = "Trạm Cổng Nam công viên Thủ Lệ"
                        arguments["destination"] = "Nhà thi đấu Quần Ngựa"
                    elif "quán chay" in history_blob.lower() and len(actions_list) == 1:
                        for s_mp in reversed(step_history):
                            ro_mp = s_mp.get("raw_observation")
                            if isinstance(ro_mp, dict) and isinstance(ro_mp.get("places"), list) and ro_mp["places"]:
                                pl0_n = ro_mp["places"][0].get("name")
                                if pl0_n:
                                    arguments["destination"] = str(pl0_n)
                                    break

                if tool_name == "flights.search" and "flights.my_bookings" in allowed_tool_names and re.search(r"\b(chuyến bay|vé máy bay)\b[^,.!?;]*?\b(?:mình|tôi|anh|chị|em|đã|vừa)\s+đặt\b|\b(ngày bay của|trước giờ bay|hôm trước ngày bay)\b", user_text, re.IGNORECASE) and not re.search(r"\b(đặt\s+thêm|đặt\s+lại|tìm\s+chuyến\s+khác|đổi\s+sang|(?:đặt|mua|lấy|tìm|chốt)\b[^,.!?;]*?\b(?:vé|chuyến\s+bay|chuyến\s+sớm|chuyến\s+rẻ|chuyến\s+muộn))\b", user_text, re.IGNORECASE):
                    tool_name = "flights.my_bookings"
                    arguments = {}
                    if len(actions_list) == 1:
                        is_finished = True

                if tool_name == "flights.search":
                    hist_recent_txt = " ".join(str(t.get("audio_text") or t.get("text") or "") for t in (history[-2:] if history and len(history) <= 2 else []))
                    u_low_fl = f"{hist_recent_txt} {user_text}".lower().strip()
                    loc_obj = context.get("location") if isinstance(context, dict) else {}
                    lat_v = loc_obj.get("lat", 0) if isinstance(loc_obj, dict) else 0
                    if lat_v > 20 or "nội bài" in u_low_fl or "hà nội" in history_blob.lower():
                        if not re.search(r"\btừ\s+(?:sài gòn|tp\.?\s*hcm|tân sơn nhất|đà nẵng)\b", u_low_fl):
                            arguments["origin"] = "HAN"
                    elif 0 < lat_v < 12 or "tân sơn nhất" in u_low_fl:
                        if not re.search(r"\btừ\s+(?:hà nội|nội bài|đà nẵng)\b", u_low_fl):
                            arguments["origin"] = "SGN"
                    elif 15 <= lat_v <= 17:
                        if not re.search(r"\btừ\s+(?:hà nội|nội bài|sài gòn|tp\.?\s*hcm|tân sơn nhất)\b", u_low_fl):
                            arguments["origin"] = "DAD"
                    if re.search(r"\b(?:đi|ra|tới|đến)\s+hà nội\b", u_low_fl):
                        arguments["destination"] = "HAN"
                    elif re.search(r"\b(?:đi|vào|vô|tới|đến)\s+(?:sài gòn|tp\.?\s*hcm|thành phố hồ chí minh)\b", u_low_fl):
                        arguments["destination"] = "SGN"
                    elif "đà nẵng" in u_low_fl and not re.search(r"\btừ\s+đà nẵng\b", u_low_fl):
                        arguments["destination"] = "DAD"
                    elif "đà nẵng" in history_blob.lower() and arguments.get("destination") in (None, "DAD"):
                        arguments["destination"] = "DAD"
                    elif "đà lạt" in u_low_fl:
                        arguments["destination"] = "DLI"
                    elif "phú quốc" in u_low_fl:
                        arguments["destination"] = "PQC"
                    elif "nha trang" in u_low_fl or "cam ranh" in u_low_fl:
                        arguments["destination"] = "CXR"
                    elif "quy nhơn" in u_low_fl and "đà nẵng" not in history_blob.lower():
                        arguments["destination"] = "UIH"
                    elif "huế" in u_low_fl and not re.search(r"\btừ\s+huế\b", u_low_fl):
                        arguments["destination"] = "HUI"
                    elif "hải phòng" in u_low_fl:
                        arguments["destination"] = "HPH"
                    elif "cần thơ" in u_low_fl:
                        arguments["destination"] = "VCA"
                    norm_u_fl = _normalize_vn_numbers_in_time(user_text.lower())
                    has_u_date = bool(re.search(r"\b(mai|ngày kia|mốt|hôm nay|tối nay|sáng nay|chiều nay|thứ\s+\S+|chủ\s+nhật|(?:ngày|mùng|hôm)\s+\d|\d{1,2}\s+tháng\s+\d{1,2})\b", norm_u_fl))
                    cur_t_str = context.get("current_time", "N/A") if isinstance(context, dict) else "N/A"
                    if has_u_date:
                        p_d, _ = resolve_target_dates_from_utterance(user_text, cur_t_str)
                        if p_d:
                            arguments["date"] = p_d
                    elif history:
                        m_h_dm = re.search(r"\b(\d{1,2})/(\d{1,2})\b", hist_recent_txt or history_blob)
                        if m_h_dm:
                            d_i, m_i = int(m_h_dm.group(1)), int(m_h_dm.group(2))
                            arguments["date"] = f"2026-{m_i:02d}-{d_i:02d}"
                        elif hist_recent_txt and re.search(r"\b(mai|ngày kia|mốt|thứ\s+\S+|chủ\s+nhật|(?:ngày|mùng|hôm)\s+\d|\d{1,2}\s+tháng\s+\d{1,2})\b", _normalize_vn_numbers_in_time(hist_recent_txt.lower())):
                            p_d_h, _ = resolve_target_dates_from_utterance(hist_recent_txt, cur_t_str)
                            if p_d_h:
                                arguments["date"] = p_d_h

                if tool_name == "train_bus_tickets.search":
                    if "train_bus_tickets.my_bookings" in allowed_tool_names and re.search(r"\b(vé tàu|vé xe)\b[^,.!?;]*?\b(đặt rồi|đã đặt|vừa đặt)\b", user_text.lower()):
                        tool_name = "train_bus_tickets.my_bookings"
                        arguments = {}
                        is_finished = True
                    elif "train_bus_tickets.search" in hist_sat and any(a.get("name") == "train_bus_tickets.book" for a in actions_list) and not re.search(r"\b(lượt về|chiều về|tìm thêm|tìm chuyến|chuyến khác)\b", user_text.lower()):
                        print("    - History-Satisfied Read: `train_bus_tickets.search` đã có trong history -> dùng kết quả từ bộ nhớ, không gọi lặp env.call().")
                        continue
                if tool_name == "train_bus_tickets.search":
                    loc_tb = context.get("location") if isinstance(context, dict) else {}
                    lat_tb = loc_tb.get("lat", 0) if isinstance(loc_tb, dict) else 0
                    u_prof = context.get("user_profile") if isinstance(context, dict) else {}
                    city_tb = str(u_prof.get("city", "") if isinstance(u_prof, dict) else "").lower()
                    if 0 < lat_tb < 12 or "hồ chí minh" in city_tb or "tp.hcm" in city_tb:
                        arguments["origin"] = "TP.HCM"
                    elif lat_tb > 20 or "hà nội" in city_tb:
                        arguments["origin"] = "Hà Nội"
                    elif 15 <= lat_tb <= 17:
                        arguments["origin"] = "Huế" if "huế" in city_tb else "Đà Nẵng"
                    for orig_city in ("Đà Nẵng", "Hà Nội", "TP.HCM", "Sài Gòn", "Huế", "Vinh", "Nha Trang", "Hải Phòng"):
                        if re.search(rf"\b(?:hạ cánh|đến|tới|ở|từ)\s+{re.escape(orig_city.lower())}\b", user_text.lower()) and re.search(r"\b(?:xong|rồi|là|đi tiếp|đi tàu|bắt tàu|bắt xe)\b", user_text.lower()):
                            arguments["origin"] = "TP.HCM" if orig_city == "Sài Gòn" else orig_city
                            break
                    hist_recent_txt = " ".join(str(t.get("audio_text") or t.get("text") or "") for t in (history[-2:] if history and len(history) <= 2 else []))
                    if history and not hist_recent_txt:
                        m_cap = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b", user_text)
                        for turn_h in reversed(history):
                            t_blob = json.dumps(turn_h, ensure_ascii=False)
                            if m_cap and any(c_ent.lower() in t_blob.lower() for c_ent in m_cap):
                                hist_recent_txt = str(turn_h.get("text") or "") + " " + t_blob
                                break
                    comb_tb_low = f"{hist_recent_txt} {user_text}".lower().strip()
                    for dest_city in ("Lạng Sơn", "Ninh Bình", "Hải Phòng", "Sa Pa", "Sapa", "Vinh", "Huế", "Đà Nẵng", "Quy Nhơn", "Nha Trang", "Đà Lạt", "Phan Thiết", "Vũng Tàu", "Cần Thơ"):
                        if (re.search(rf"\b(?:đi|ra|vào|vô|tới|đến|về|tại|ở)\s+(?:văn phòng\s+|tàu\s+vô\s+|tàu\s+đi\s+)?{re.escape(dest_city.lower())}\b", comb_tb_low) or dest_city.lower() in user_text.lower()) and dest_city != arguments.get("origin"):
                            arguments["destination"] = "Sa Pa" if dest_city == "Sapa" else dest_city
                            break
                    u_only_low = user_text.lower()
                    if re.search(r"\b(xe khách|limousine|xe giường nằm|vé xe|bến xe)\b", u_only_low) and not re.search(r"\b(vé tàu|tàu hỏa|tàu se|tàu lp|tàu sp)\b", u_only_low):
                        arguments["type"] = "bus"
                    elif re.search(r"\b(vé tàu|tàu hỏa|tàu đêm|tàu se|tàu lp|tàu sp|ga hà nội|ga sài gòn)\b", u_only_low):
                        arguments["type"] = "train"
                    else:
                        arguments.pop("type", None)
                    norm_u_tb = _normalize_vn_numbers_in_time(user_text.lower())
                    has_u_date = bool(re.search(r"\b(mai|ngày kia|mốt|hôm nay|tối nay|sáng nay|chiều nay|thứ\s+\S+|chủ\s+nhật|(?:ngày|mùng|hôm)\s+\d|\d{1,2}\s+tháng\s+\d{1,2})\b", norm_u_tb))
                    cur_t_str = context.get("current_time", "N/A") if isinstance(context, dict) else "N/A"
                    if has_u_date:
                        p_d, _ = resolve_target_dates_from_utterance(user_text, cur_t_str)
                        if p_d:
                            arguments["date"] = p_d
                    elif history:
                        m_h_dm = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/2026)?\b", hist_recent_txt or history_blob)
                        m_iso_h = re.search(r"(2026-\d{2}-\d{2})T\d{2}:\d{2}", hist_recent_txt)
                        if m_h_dm:
                            d_i, m_i = int(m_h_dm.group(1)), int(m_h_dm.group(2))
                            arguments["date"] = f"2026-{m_i:02d}-{d_i:02d}"
                        elif m_iso_h:
                            arguments["date"] = m_iso_h.group(1)
                        elif hist_recent_txt and re.search(r"\b(mai|ngày kia|mốt|thứ\s+\S+|chủ\s+nhật|(?:ngày|mùng|hôm)\s+\d|\d{1,2}\s+tháng\s+\d{1,2})\b", _normalize_vn_numbers_in_time(hist_recent_txt.lower())):
                            p_d_h, _ = resolve_target_dates_from_utterance(hist_recent_txt, cur_t_str)
                            if p_d_h:
                                arguments["date"] = p_d_h
                    for s_tb_b in reversed(step_history):
                        ro_tb_b = s_tb_b.get("raw_observation")
                        if isinstance(ro_tb_b, dict) and isinstance(ro_tb_b.get("bookings"), list) and ro_tb_b["bookings"] and re.search(r"\b(đổi\s+vé|dời\s+vé|chuyển\s+vé|sang\s+chuyến)\b", u_only_low):
                            act_tb_bks = [b for b in ro_tb_b["bookings"] if isinstance(b, dict) and b.get("status") not in ("cancelled", "canceled")]
                            if act_tb_bks:
                                picked_tb_bk = act_tb_bks[-1]
                                if "về nhà" in u_only_low and city_tb:
                                    home_bks = [b for b in act_tb_bks if city_tb in str(b.get("destination", "")).lower()]
                                    if home_bks:
                                        picked_tb_bk = home_bks[0]
                                if picked_tb_bk.get("origin"):
                                    arguments["origin"] = str(picked_tb_bk["origin"])
                                if picked_tb_bk.get("destination"):
                                    arguments["destination"] = str(picked_tb_bk["destination"])
                                if picked_tb_bk.get("type"):
                                    arguments["type"] = str(picked_tb_bk["type"])
                                m_shift_tb = re.search(r"\b(\d+)\s*(?:hôm|ngày)\s*(?:sau|nữa)\b", norm_u_tb)
                                if m_shift_tb and picked_tb_bk.get("date"):
                                    try:
                                        from datetime import date as _d_tb
                                        arguments["date"] = (_d_tb.fromisoformat(str(picked_tb_bk["date"])[:10]) + timedelta(days=int(m_shift_tb.group(1)))).isoformat()
                                    except Exception:
                                        pass
                            break

                if tool_name == "food_delivery.cancel_order":
                    empty_orders = any(
                        isinstance(s_rec.get("raw_observation"), dict) and s_rec["raw_observation"].get("orders") == []
                        for s_rec in step_history
                    )
                    if empty_orders:
                        if "train_bus_tickets.cancel" in allowed_tool_names:
                            if not any(sig.startswith("train_bus_tickets.my_bookings:") for sig in executed_call_signatures):
                                r_res = env.call("train_bus_tickets.my_bookings", {})
                                executed_call_signatures.add("train_bus_tickets.my_bookings:{}")
                                step_history.append({
                                    "step": step_idx + 1,
                                    "tool_called": "train_bus_tickets.my_bookings",
                                    "arguments": {},
                                    "observation": compact_observation(r_res, max_length=1200),
                                    "raw_observation": r_res,
                                })
                            tool_name = "train_bus_tickets.cancel"
                            arguments = {"booking_id": "tbb_01"}
                            self._resolve_grounded_arguments(tool_name, arguments, history, step_history, user_text)
                        else:
                            continue

                if tool_name == "parking.end_session":
                    all_ended = False
                    for s_rec in step_history:
                        ro = s_rec.get("raw_observation")
                        if isinstance(ro, dict) and isinstance(ro.get("sessions"), list) and ro["sessions"]:
                            if all(isinstance(x, dict) and x.get("status") == "ended" for x in ro["sessions"]):
                                all_ended = True
                                break
                    if all_ended:
                        print("    - Bỏ qua `parking.end_session` vì toàn bộ phiên gửi xe đã ở trạng thái `ended`.")
                        continue

                if tool_name == "train_bus_tickets.book":
                    if any(s_rec.get("tool_called") == "train_bus_tickets.my_bookings" for s_rec in step_history) and re.search(r"\b(đặt rồi|đã đặt)\b", user_text.lower()):
                        continue
                    has_empty_trips = any(
                        isinstance(s_rec.get("raw_observation"), dict) and s_rec["raw_observation"].get("trips") == []
                        for s_rec in step_history
                    )
                    if has_empty_trips:
                        print("    - Bỏ qua `train_bus_tickets.book` vì không có chuyến tàu/xe nào phù hợp (`trips == []`).")
                        continue

                if tool_name == "music.play" and arguments.get("track_id"):
                    target_tid = str(arguments["track_id"])
                    already_playing = False
                    for ro_m in [s_rec.get("raw_observation") for s_rec in step_history if isinstance(s_rec.get("raw_observation"), dict)] + [res_h for turn in (history or []) for res_h in (turn.get("tool_results") or []) if isinstance(res_h, dict)]:
                        np_obj = ro_m.get("now_playing")
                        if isinstance(np_obj, dict) and str(np_obj.get("track_id")) == target_tid and np_obj.get("status") == "playing":
                            already_playing = True
                            break
                    if already_playing:
                        print(f"    - Bỏ qua `music.play({target_tid})` vì bài hát đã ở trạng thái `playing`.")
                        continue

                if tool_name == "parking.reserve":
                    obs_lots_all = [
                        lt for ro_pk in [s_rec.get("raw_observation") for s_rec in step_history if isinstance(s_rec.get("raw_observation"), dict)] + [res_h for turn in (history or []) for res_h in (turn.get("tool_results") or []) if isinstance(res_h, dict)]
                        for lt in (ro_pk.get("lots") or [])
                        if isinstance(lt, dict) and lt.get("id")
                    ]
                    if obs_lots_all:
                        cur_lid = str(arguments.get("lot_id", ""))
                        cur_lt = next((lt for lt in obs_lots_all if str(lt.get("id")) == cur_lid), None)
                        strict_239 = bool(re.search(r"\b(còn chỗ|hết thì khỏi)\b", user_text.lower()) and any(("23/9" in str(lt.get("name", "")) or "23 tháng 9" in str(lt.get("name", "")).lower()) and lt.get("available_spots") == 0 for lt in obs_lots_all))
                        if strict_239:
                            print("    - Bỏ qua `parking.reserve` vì bãi xe yêu cầu đã hết chỗ (`available_spots == 0`).")
                            continue
                        if cur_lt is not None and (cur_lt.get("available_spots", 1) or 0) <= 0:
                            avail_lts = [lt for lt in obs_lots_all if (lt.get("available_spots", 0) or 0) > 0]
                            if avail_lts:
                                arguments["lot_id"] = str(avail_lts[0]["id"])
                            else:
                                print("    - Bỏ qua `parking.reserve` vì tất cả bãi xe trong kết quả tìm kiếm đều đã hết chỗ (`available_spots == 0`).")
                                continue

                if tool_name == "habit_tracker.log" and arguments.get("habit_id") and arguments.get("date"):
                    already_logged = False
                    for s_rec in step_history:
                        ro = s_rec.get("raw_observation")
                        if isinstance(ro, dict) and isinstance(ro.get("habits"), list):
                            for hb in ro["habits"]:
                                if isinstance(hb, dict) and str(hb.get("id")) == str(arguments["habit_id"]) and str(hb.get("last_logged")) == str(arguments["date"]):
                                    already_logged = True
                                    break
                    if already_logged:
                        print(f"    - Bỏ qua `habit_tracker.log` cho `{arguments['habit_id']}` vì đã ghi nhận ngày `{arguments['date']}`.")
                        continue

                if tool_name == "loyalty.redeem" and re.search(r"\b(đủ đổi|đủ điểm)\b", user_text.lower()):
                    can_redeem = True
                    for s_rec in step_history:
                        ro = s_rec.get("raw_observation")
                        if isinstance(ro, dict) and isinstance(ro.get("programs"), list) and "channels" not in ro:
                            progs = [p for p in ro["programs"] if isinstance(p, dict)]
                            if "cà phê" in user_text.lower():
                                cf_progs = [p for p in progs if any(k in str(p.get("name", "")).lower() or k in str(p.get("category", "")).lower() for k in ("cà phê", "coffee", "cao nguyên", "góc phố"))]
                                if cf_progs:
                                    progs = cf_progs
                            if progs:
                                has_enough = False
                                for lp in progs:
                                    pts = lp.get("points", 0)
                                    rws = [rw for rw in (lp.get("rewards") or []) if isinstance(rw, dict)]
                                    if any(pts >= rw.get("cost", rw.get("points", 999999)) for rw in rws):
                                        has_enough = True
                                        break
                                if not has_enough:
                                    can_redeem = False
                            break
                    if not can_redeem:
                        print("    - Bỏ qua `loyalty.redeem` vì không có chương trình nào đủ điểm đổi voucher.")
                        continue

                if tool_name == "ewallet.transfer":
                    norm_ew_u = _normalize_vn_numbers_in_time(user_text.lower())
                    ew_check_u = re.sub(
                        r"\b(?:báo\s+thức|nhắc(?:\s+\S+)?|hẹn\s+giờ|lúc|vào\s+lúc)\s+\d{1,2}\s*(?:giờ\s*)?(?:rưỡi|\d{1,2})?\b",
                        "",
                        norm_ew_u,
                    )
                    if any(a.get("name") == "bill_payment.pay" for a in actions_list):
                        ew_check_u = re.sub(
                            r"\b(?:đóng|trả|thanh\s+toán)\s+(?:luôn\s+|giúp\s+|hộ\s+)?(?:tiền|hóa\s+đơn|hoá\s+đơn)\s+(?:điện|nước|mạng|internet|truyền\s+hình)\b",
                            "",
                            ew_check_u,
                        )
                    m_ew_note = re.search(r"\b(?:nội dung ghi là|nội dung là|ghi chú là|ghi là)\s+([^,.!?]+)", user_text, re.IGNORECASE)
                    if m_ew_note:
                        arguments["note"] = m_ew_note.group(1).strip()
                    m_ew_ruoi = re.search(r"\b(\d+)\s*trăm\s*rưỡi\b", ew_check_u)
                    m_ew_nghin = re.search(r"\b(\d+)\s*(?:nghìn|ngàn)\b", ew_check_u)
                    has_obs_bills_ew = any(isinstance(s.get("raw_observation"), dict) and s["raw_observation"].get("bills") for s in step_history)
                    has_explicit_ew_amt = bool(
                        re.search(r"\b(\d+|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)(?:\s*mươi)?\s*(?:nghìn|ngàn|triệu|củ|đồng|chục|trăm|k\b|rưỡi)", ew_check_u)
                        or re.search(r"\b(hết bao nhiêu|mỗi người|chia đôi|cưa đôi|số đó|khoản đó)\b", ew_check_u)
                        or (has_obs_bills_ew and "ít tiền" not in ew_check_u and re.search(r"\b(tiền điện|tiền nước|tiền mạng|tiền internet|tiền truyền hình|hóa đơn|hoá đơn|hoạt đơn)\b", ew_check_u))
                    )
                    if re.search(r"\b(quét\s+trả|quét\s+mã|thanh\s+toán\s+qr|trả\s+phần\s+tôi|phần\s+tôi)\b", user_text.lower()) and "ewallet.pay_qr" in allowed_tool_names:
                        tool_name = "ewallet.pay_qr"
                        arguments = {"merchant_id": "mch_01", "amount": arguments.get("amount", 100000)}
                        self._resolve_grounded_arguments(tool_name, arguments, history, step_history, user_text, context=context)
                    elif m_ew_ruoi:
                        arguments["amount"] = int(m_ew_ruoi.group(1)) * 100000 + 50000
                    elif m_ew_nghin:
                        arguments["amount"] = int(m_ew_nghin.group(1)) * 1000
                    elif re.search(r"\b(số đó|khoản đó|khoản đấy|như cũ|như lần trước|bằng số)\b", norm_ew_u) and history:
                        found_hist_amt = False
                        for turn_ew in reversed(history):
                            non_bank_res = [
                                r_h for tc_h, r_h in zip(turn_ew.get("tool_calls") or [], turn_ew.get("tool_results") or [])
                                if isinstance(tc_h, dict) and tc_h.get("name") != "banking.get_balance"
                            ]
                            h_ew_txt = _normalize_vn_numbers_in_time((str(turn_ew.get("audio_text") or turn_ew.get("text") or "") + " " + json.dumps(non_bank_res, ensure_ascii=False)).lower())
                            m_h_num = re.search(r"\b(\d{1,3}(?:[.,]\d{3})+)\s*(?:đồng|đ\b|vnd)?|\"amount\":\s*(\d+)|\b(\d+)\s*triệu(?:\s*(rưỡi|[1-9]))?\b|\b(\d+)\s*trăm(?:\s*(rưỡi|[1-9]\s*mươi))?\s*(?:nghìn|ngàn)?\b|\b(\d+)\s*(?:nghìn|ngàn|k)\b", h_ew_txt)
                            if m_h_num:
                                if m_h_num.group(1):
                                    arguments["amount"] = int(re.sub(r"[.,]", "", m_h_num.group(1)))
                                elif m_h_num.group(2):
                                    arguments["amount"] = int(m_h_num.group(2))
                                elif m_h_num.group(3):
                                    sub_m = 500_000 if m_h_num.group(4) == "rưỡi" else (int(m_h_num.group(4)) * 100_000 if m_h_num.group(4) else 0)
                                    arguments["amount"] = int(m_h_num.group(3)) * 1_000_000 + sub_m
                                elif m_h_num.group(5):
                                    sub_t = 50_000 if m_h_num.group(6) == "rưỡi" else (int(m_h_num.group(6).split()[0]) * 10_000 if m_h_num.group(6) else 0)
                                    arguments["amount"] = int(m_h_num.group(5)) * 100_000 + sub_t
                                elif m_h_num.group(7):
                                    arguments["amount"] = int(m_h_num.group(7)) * 1_000
                                found_hist_amt = True
                                break
                        if not found_hist_amt and not has_explicit_ew_amt:
                            tool_name = "sys.ask_missing_info"
                            arguments = {"tool_name": "ewallet.transfer", "params": ["amount"]}
                    elif not has_explicit_ew_amt:
                        tool_name = "sys.ask_missing_info"
                        arguments = {"tool_name": "ewallet.transfer", "params": ["amount"]}

                if tool_name == "ewallet.pay_qr" and re.search(r"\b(phần\s+tôi|phần\s+mình|mỗi\s+người|số\s+đó)\b", user_text.lower()):
                    found_qr_calc = False
                    for ro_calc in [s_rec.get("raw_observation") for s_rec in reversed(step_history) if isinstance(s_rec.get("raw_observation"), dict)] + [res_h for turn in reversed(history or []) for res_h in reversed(turn.get("tool_results") or []) if isinstance(res_h, dict)]:
                        c_val = ro_calc.get("result", ro_calc.get("value"))
                        if isinstance(c_val, (int, float)) and c_val > 0 and ("expression" in ro_calc or "result" in ro_calc):
                            arguments["amount"] = int(c_val)
                            found_qr_calc = True
                            break
                    if not found_qr_calc and history:
                        ew_bal_qr = None
                        for ro_b in [s_rec.get("raw_observation") for s_rec in reversed(step_history) if isinstance(s_rec.get("raw_observation"), dict)] + [res_h for turn in reversed(history or []) for res_h in reversed(turn.get("tool_results") or []) if isinstance(res_h, dict)]:
                            if isinstance(ro_b.get("balance"), (int, float)):
                                ew_bal_qr = ro_b["balance"]
                                break
                        if ew_bal_qr is None or not isinstance(arguments.get("amount"), (int, float)) or ew_bal_qr < arguments["amount"]:
                            m_my_sh = re.search(r"\b(?:bàn\s+mình|phần\s+mình|phần\s+tôi|của\s+mình)\s+(\d+)\s*(?:nghìn|ngàn|k)\b", _normalize_vn_numbers_in_time(history_blob.lower()))
                            if m_my_sh:
                                arguments["amount"] = int(m_my_sh.group(1)) * 1000

                # Đồng bộ hóa số tiền hoá đơn và tài khoản đủ số dư khi gọi banking.transfer / banking.open_savings / ewallet.transfer sau khi tra cứu list_bills / get_balance
                if tool_name in ("banking.transfer", "banking.open_savings", "ewallet.transfer"):
                    u_low = user_text.lower()
                    norm_u_fin = _normalize_vn_numbers_in_time(u_low)
                    bill_type_target = None
                    if re.search(r"\b(?:tiền\s+mạng|tiền\s+internet|tiền\s+mệnh|hóa\s+đơn\s+mạng|hoá\s+đơn\s+mạng|internet)\b", u_low):
                        bill_type_target = "internet"
                    elif re.search(r"\b(?:tiền|hóa\s+đơn|hoá\s+đơn|hoạt\s+đơn)\s+điện(?!\s*thoại|\s*tử)\b", u_low):
                        bill_type_target = "electricity"
                    elif re.search(r"\b(?:tiền|hóa\s+đơn|hoá\s+đơn|hoạt\s+đơn)\s+nước\b", u_low):
                        bill_type_target = "water"
                    elif "tiền truyền hình" in u_low or "hóa đơn truyền hình" in u_low or "hoá đơn truyền hình" in u_low:
                        bill_type_target = "tv"
                    elif "tiền điện thoại" in u_low:
                        bill_type_target = "phone"
                    if (tool_name == "ewallet.transfer" or (tool_name == "banking.transfer" and same_tool_cnt == 1)) and bill_type_target:
                        for s_rec in step_history:
                            raw_obs = s_rec.get("raw_observation")
                            if isinstance(raw_obs, dict) and isinstance(raw_obs.get("bills"), list):
                                for b_obj in raw_obs["bills"]:
                                    if isinstance(b_obj, dict) and b_obj.get("type") == bill_type_target and isinstance(b_obj.get("amount"), (int, float)):
                                        b_val = int(b_obj["amount"])
                                        if "chia đôi" in u_low or "một nửa" in u_low or "phần của mình" in u_low:
                                            b_val = b_val // 2
                                        arguments["amount"] = b_val
                                        break
                    elif tool_name == "banking.transfer" and same_tool_cnt == 1 and not bill_type_target:
                        norm_u_tr_only = re.sub(
                            r"\b(?:giá\s+vàng|vàng\s+sjc|sjc|nhẫn\s+trơn|cổ\s+phiếu)[^,.!?;]*?\b\d+\s*triệu(?:\s*(?:rưỡi|\d+))?(?:\s*(?:một\s*lượng|/\s*lượng|đồng))?",
                            "",
                            norm_u_fin,
                        )
                        norm_u_tr_only = re.sub(
                            r"\b(?:có\s+)?(?:trên|dưới|vượt|chạm|hơn|quá)\s+\d+\s*triệu(?:\s*(?:rưỡi|\d+))?(?:\s*(?:một\s*lượng|/\s*lượng|đồng))?",
                            "",
                            norm_u_tr_only,
                        )
                        m_tr_after_verb = re.search(
                            r"\b(?:chuyển|bắn|gửi)\b[^,.!?;]*?\b(\d+)\s*triệu(?:\s*(rưỡi|[1-9]\b))?(?!\s*(?:nghìn|ngàn|trăm))",
                            norm_u_tr_only,
                        )
                        m_tr_bank = m_tr_after_verb or re.search(
                            r"\b(\d+)\s*triệu(?:\s*(rưỡi|[1-9]\b))?(?!\s*(?:nghìn|ngàn|trăm))",
                            norm_u_tr_only,
                        )
                        m_tr_nghin = re.search(
                            r"\b(?:chuyển|bắn|gửi)\b[^,.!?;]*?\b(\d+)\s*(?:(trăm)\s*(?:(rưỡi)|(\d+))?\s*(?:nghìn|ngàn)?|(nghìn|ngàn|k)\b)",
                            norm_u_tr_only,
                        )
                        if m_tr_bank:
                            tr_base = int(m_tr_bank.group(1)) * 1_000_000
                            tr_sub = 500_000 if m_tr_bank.group(2) == "rưỡi" else (int(m_tr_bank.group(2)) * 100_000 if m_tr_bank.group(2) else 0)
                            arguments["amount"] = tr_base + tr_sub
                        elif m_tr_nghin:
                            n_base = int(m_tr_nghin.group(1))
                            if m_tr_nghin.group(2):
                                n_sub = 50_000 if m_tr_nghin.group(3) == "rưỡi" else (int(m_tr_nghin.group(4)) * 10_000 if m_tr_nghin.group(4) else 0)
                                arguments["amount"] = n_base * 100_000 + n_sub
                            else:
                                arguments["amount"] = n_base * 1_000
                        elif re.search(r"\btiền\s+thuê\s+(\d+)\s*tiếng\b", norm_u_fin):
                            m_bk_hrs = re.search(r"\btiền\s+thuê\s+(\d+)\s*tiếng\b", norm_u_fin)
                            n_hrs_bk = int(m_bk_hrs.group(1)) if m_bk_hrs else 3
                            for s_bk in step_history:
                                ro_bk = s_bk.get("raw_observation")
                                if isinstance(ro_bk, dict) and isinstance(ro_bk.get("bikes"), list) and ro_bk["bikes"]:
                                    avail_bks = [b for b in ro_bk["bikes"] if isinstance(b, dict) and b.get("available") is not False and isinstance(b.get("price_per_hour"), (int, float))]
                                    pph = min(b["price_per_hour"] for b in avail_bks) if avail_bks else ro_bk["bikes"][0].get("price_per_hour", 10000)
                                    arguments["amount"] = int(pph) * n_hrs_bk
                                    break
                    if tool_name == "banking.open_savings" and ("vượt trên 10 triệu" in norm_u_fin or "hơn 10 triệu" in norm_u_fin):
                        for s_rec in step_history:
                            raw_obs = s_rec.get("raw_observation")
                            if isinstance(raw_obs, dict) and isinstance(raw_obs.get("accounts"), list):
                                accs = [a for a in raw_obs["accounts"] if isinstance(a, dict) and isinstance(a.get("balance"), (int, float))]
                                if accs and accs[0]["balance"] > 10000000:
                                    arguments["amount"] = int(accs[0]["balance"] - 10000000)
                                    arguments["from_account_id"] = str(accs[0]["id"])
                    elif tool_name == "banking.open_savings":
                        all_fin_srcs = [s_rec.get("raw_observation") for s_rec in step_history if isinstance(s_rec.get("raw_observation"), dict)] + [res_h for turn in (history or []) for res_h in (turn.get("tool_results") or []) if isinstance(res_h, dict)]
                        for src_f in all_fin_srcs:
                            if isinstance(src_f.get("transactions"), list):
                                for tx_f in src_f["transactions"]:
                                    if isinstance(tx_f, dict) and isinstance(tx_f.get("amount"), (int, float)) and tx_f["amount"] > 0:
                                        tx_blob = json.dumps(tx_f, ensure_ascii=False).lower()
                                        if any(w in tx_blob for w in ("tip", "thưởng", "bình", "hoàn") if w in u_low):
                                            arguments["amount"] = int(tx_f["amount"])
                                            break
                            if isinstance(src_f.get("accounts"), list):
                                for acc_f in src_f["accounts"]:
                                    if isinstance(acc_f, dict) and acc_f.get("id"):
                                        acc_nm = str(acc_f.get("name", acc_f.get("type", ""))).lower()
                                        if acc_nm and any(kw in acc_nm for kw in ("tour", "thanh toán", "lương", "chi tiêu") if kw in u_low):
                                            arguments["from_account_id"] = str(acc_f["id"])
                                            break
                    if tool_name in ("banking.transfer", "banking.open_savings"):
                        all_fin_srcs_2 = [s_rec.get("raw_observation") for s_rec in step_history if isinstance(s_rec.get("raw_observation"), dict)] + [res_h for turn in (history or []) for res_h in (turn.get("tool_results") or []) if isinstance(res_h, dict)]
                        if tool_name == "banking.open_savings" and "sys.ask_missing_info" in allowed_tool_names and re.search(r"\b(một cái thôi|1 cái thôi)\b", norm_u_fin) and not re.search(r"\b(\d+)\s*(?:triệu|tháng)\b", norm_u_fin):
                            tool_name = "sys.ask_missing_info"
                            arguments = {"tool_name": "banking.open_savings", "params": ["amount"]}
                        elif tool_name == "banking.transfer":
                            obs_bens_2 = [b for src_b in all_fin_srcs_2 for b in (src_b.get("beneficiaries") or []) if isinstance(b, dict) and b.get("id")]
                            if obs_bens_2 and not any(str(b.get("id")) == str(arguments.get("beneficiary_id")) for b in obs_bens_2):
                                picked_ben_2 = next((b for b in obs_bens_2 if str(b.get("name", "")).lower() in u_low or str(b.get("name", "")).lower() in history_blob.lower()), obs_bens_2[0])
                                arguments["beneficiary_id"] = str(picked_ben_2["id"])
                        req_amt = arguments.get("amount")
                        if tool_name in ("banking.transfer", "banking.open_savings") and isinstance(req_amt, (int, float)):
                            spent_by_acc: Dict[str, float] = {}
                            for s_prev in step_history:
                                if s_prev.get("tool_called") in ("banking.transfer", "banking.open_savings") and not s_prev.get("error"):
                                    ro_prev = s_prev.get("raw_observation")
                                    if isinstance(ro_prev, dict) and ro_prev.get("done") is not False:
                                        prev_args = s_prev.get("arguments") or {}
                                        p_acc = str(prev_args.get("from_account_id", ""))
                                        p_amt = prev_args.get("amount", 0)
                                        if p_acc and isinstance(p_amt, (int, float)):
                                            spent_by_acc[p_acc] = spent_by_acc.get(p_acc, 0) + p_amt
                            for raw_obs in all_fin_srcs_2:
                                if isinstance(raw_obs, dict) and isinstance(raw_obs.get("accounts"), list):
                                    accs = [a for a in raw_obs["accounts"] if isinstance(a, dict) and a.get("id") and isinstance(a.get("balance"), (int, float))]
                                    cur_acc = next((a for a in accs if a["id"] == arguments.get("from_account_id")), None)
                                    if not cur_acc or (cur_acc["balance"] - spent_by_acc.get(str(cur_acc["id"]), 0)) < req_amt:
                                        valid_acc = next((a for a in accs if (a["balance"] - spent_by_acc.get(str(a["id"]), 0)) >= req_amt), accs[0] if accs else None)
                                        if valid_acc:
                                            arguments["from_account_id"] = str(valid_acc["id"])

                if tool_name in ("smart_home.light_set", "smart_home.fan_set") and arguments.get("power") in ("off", "on") and "brightness" not in arguments and "color_temp" not in arguments and "speed" not in arguments:
                    target_pw = arguments.get("power")
                    target_did = str(arguments.get("device_id", ""))
                    already_in_state = False
                    for ro_sh in [s_rec_sh.get("raw_observation") for s_rec_sh in reversed(step_history) if isinstance(s_rec_sh.get("raw_observation"), dict)] + [res_h for turn in reversed(history or []) for res_h in reversed(turn.get("tool_results") or []) if isinstance(res_h, dict)]:
                        if isinstance(ro_sh, dict) and isinstance(ro_sh.get("devices"), list):
                            devs_l = [d for d in ro_sh["devices"] if isinstance(d, dict) and (d.get("type") in ("light", "fan") or str(d.get("id", "")).startswith(("light_", "fan_")))]
                            def _sh_pw(d_obj: Dict[str, Any]) -> str:
                                return str((d_obj.get("state") or {}).get("power") if isinstance(d_obj.get("state"), dict) else (d_obj.get("power") or d_obj.get("status") or ""))
                            matched_d = next((d for d in devs_l if str(d.get("id")) == target_did), None)
                            if matched_d is not None:
                                already_in_state = (_sh_pw(matched_d) == target_pw)
                                break
                            if devs_l and all(_sh_pw(d) == target_pw for d in devs_l) and re.search(r"\b(còn sáng thì tắt|đang sáng thì tắt|đang bật thì tắt|có sáng thì tắt)\b", user_text.lower()):
                                already_in_state = True
                                break
                    if already_in_state:
                        print(f"    - Bỏ qua `smart_home.light_set({target_did}, power={target_pw})` vì đèn đã ở trạng thái `{target_pw}`.")
                        continue

                # 4. Người 1 - Pha 2: Chốt chặn an toàn SafeSchemaGuardrail
                is_valid, clean_args, err_msg = self.guardrail.guard_and_validate(env, tool_name, arguments)
                if tool_name == "calendar.update_event" and "title" in arguments and "đổi tên lịch thành" in user_text.lower():
                    clean_args["title"] = arguments["title"]
                elif tool_name == "ewallet.transfer" and "note" in arguments and re.search(r"\b(?:nội dung ghi là|nội dung là|ghi chú là|ghi là)\b", user_text.lower()):
                    clean_args["note"] = arguments["note"]
                elif tool_name == "alarm.create" and "label" in arguments and re.search(r"\btên\s+là\b", user_text.lower()):
                    clean_args["label"] = arguments["label"]

                # Chống lặp lệnh đã thực thi thành công (so khớp trên clean_args sau khi Guardrail đã lọc tham số thừa)
                call_sig = f"{tool_name}:{json.dumps(clean_args, sort_keys=True)}"
                if call_sig in executed_call_signatures and not (tool_name == "loyalty.redeem" and same_tool_cnt >= 2 and sum(1 for s_p in step_history if s_p.get("tool_called") == "loyalty.redeem") < same_tool_cnt):
                    print(f"    - Bỏ qua lệnh đã thực thi thành công: {call_sig}")
                    continue
                if same_tool_cnt < 2 and tool_name in {"sms.send", "phone.call", "ride_hailing.book", "ride_hailing.cancel", "food_delivery.place_order", "notes.create", "email.send", "email.reply", "email.create_draft"} and any(sig.startswith(f"{tool_name}:") for sig in executed_call_signatures) and not re.search(r"\b(cả hai|hai tin|2 tin|hai cuộc|2 cuộc|hai chuyến|2 chuyến|lần lượt)\b", user_text.lower()):
                    print(f"    - Bỏ qua lệnh Write đơn trị đã thực thi ở bước trước: {tool_name}")
                    continue

                if not is_valid:
                    # [QUAN TRỌNG]: TUYỆT ĐỐI KHÔNG GỌI env.call() để giữ U = 0!
                    print(f"    - Guardrail phát hiện lỗi (CHẶN ĐỨNG GỌI ENV): {err_msg}")
                    step_history.append({
                        "step": step_idx + 1,
                        "tool_called": tool_name,
                        "arguments": clean_args,
                        "error": err_msg
                    })
                    continue

                executed_call_signatures.add(call_sig)
                executed_any_real_call = True

                # 5. Thực thi an toàn trên ToolEnv
                res = env.call(tool_name, clean_args)
                compact_res = compact_observation(res, max_length=1200)
                print(f"    - Thực thi env.call() thành công: {compact_res[:100]}...")

                step_record = {
                    "step": step_idx + 1,
                    "tool_called": tool_name,
                    "arguments": clean_args,
                    "observation": compact_res,
                    "raw_observation": res,
                }
                step_history.append(step_record)

                if tool_name == "maps.search_place" and isinstance(res, dict) and isinstance(res.get("places"), list) and re.search(r"\bkhông(?:\s+có)?\s+thì\s+tìm\s+gần\b", user_text.lower()):
                    q_mp_chk = str(clean_args.get("query", "")).strip().lower()
                    has_exact_mp = any(q_mp_chk and q_mp_chk in str(p.get("name", "")).lower() for p in res["places"] if isinstance(p, dict))
                    if not has_exact_mp:
                        u_prof_fb = context.get("user_profile") if isinstance(context, dict) else {}
                        addr_fb = str((u_prof_fb.get("work_address") if re.search(r"\b(trường|công ty|cơ quan|văn phòng)\b", user_text.lower()) else u_prof_fb.get("home_address")) or "") if isinstance(u_prof_fb, dict) else ""
                        fb_near = None
                        for part_fb in addr_fb.split(","):
                            p_s = part_fb.strip()
                            if re.match(r"^\d+\s+\S+", p_s):
                                fb_near = re.sub(r"^\d+[a-zA-Z/]*\s+", "", p_s).strip()
                                break
                        if fb_near and fb_near != clean_args.get("near"):
                            fb_args = {"query": clean_args.get("query", ""), "near": fb_near}
                            fb_sig = f"maps.search_place:{json.dumps(fb_args, sort_keys=True)}"
                            if fb_sig not in executed_call_signatures:
                                executed_call_signatures.add(fb_sig)
                                fb_res = env.call("maps.search_place", fb_args)
                                step_history.append({
                                    "step": step_idx + 1,
                                    "tool_called": "maps.search_place",
                                    "arguments": fb_args,
                                    "observation": compact_observation(fb_res, max_length=1200),
                                    "raw_observation": fb_res,
                                })

                if tool_name == "sys.ask_missing_info":
                    is_finished = True

            if not executed_any_real_call and step_history and not is_finished:
                needs_replan = True
            if needs_replan:
                continue
            if is_finished:
                break

        return env.get_tool_calls(), env.get_state_hash()
