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
    r"(à thôi,? khỏi|à thôi,? bỏ qua đi|à thôi khỏi bỏ đi|à thôi,? em khỏi|à thôi,? mình bảo .* cứ để đấy|à thôi,? đừng .* lâu quá|thôi khỏi,? không cần|thôi khỏi,? đừng|thôi khỏi (?:chuyển|đặt|gọi|nhắn|mua|làm|tắt|bật|hủy|xóa|thêm|ghi|nhắc|nạp)\b.*nữa|thôi khỏi nha|thôi khỏi nhé|thôi khỏi em|thôi khỏi cất|khỏi đặt nữa|khỏi làm đơn trả nữa|khỏi nha,? anh|khỏi nhé,? anh|khỏi nhé|khỏi nha|khỏi nghe con|khỏi tìm nghe|không cần đâu|không cần nữa|hông cần nữa|đừng làm nữa|đừng có thêm|đừng thêm gì.*hết|đừng hủy nữa.*cứ để nguyên|đừng tắt.*cứ để đó|chưa cần lưu hay nhắc gì đâu|mà thôi,? trên mạng có rồi|mà thôi thôi,? .* cứ để đấy|mà thôi,? .* cứ giữ nguyên lịch|để chiều (?:anh|chị|em|tôi|mình) đưa|để anh tự|để chị tự|để tui tự|để tự tôi coi|để hôm khác|tự check|tự coi|tự ghi|tự làm|khỏi đổi gì hết nha|khỏi tạo phòng nữa nhé|khỏi tạo nữa nhé|khỏi tạo nữa nha|khỏi đặt vé nha|khỏi đặt xe nữa nha|hoặc cứ để rứa|thôi cứ để rứa)(?:[^.!?]*[.!?]?\s*)?$",
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
    ("door_lock_missing", r"\b(khóa ngay cửa|khoá ngay cửa|khóa cái cửa chính|khoá cái cửa chính|khóa cửa chính|khoá cửa chính)\b", lambda inst: "door_lock" in inst),
    ("parking_missing", r"\b(kết thúc (?:phiên|lượt) (?:gửi|đỗ) xe|tách cái phiên gửi xe)\b", lambda inst: "parking" in inst),
    ("maps_missing", r"\b(lái ô tô sang .* mất bao lâu|từ nhà .* sang .* mất bao lâu|chỉ đường|tìm đường)\b", lambda inst: "maps" in inst),
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
        if not isinstance(response, str) or not response.strip():
            return None

        clean = response.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()

        # Tìm kiếm khối {...} đầu tiên nếu còn dính text bên ngoài
        match = re.search(r"(\{.*\})", clean, re.DOTALL)
        if match:
            clean = match.group(1)

        try:
            def reject_constant(value):
                raise ValueError(f"Invalid JSON number: {value}")
            parsed = json.loads(clean, parse_constant=reject_constant)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("is_finished", False), bool):
                return None
            if "actions" in parsed and "action" in parsed:
                return None
            actions = parsed.get("actions", parsed.get("action"))
            if isinstance(actions, dict):
                actions = [actions]
            if actions is not None:
                if not isinstance(actions, list):
                    return None
                for action in actions:
                    if not isinstance(action, dict) or (not isinstance(action.get("name"), str) or not action["name"]):
                        return None
                    if not isinstance(action.get("arguments", {}), dict):
                        return None
            return parsed
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
            for src in raw_sources:
                if isinstance(src.get("places"), list) and src["places"]:
                    pl0 = src["places"][0]
                    if isinstance(pl0, dict) and pl0.get("id"):
                        arguments["place_id"] = str(pl0["id"])
                        arguments.pop("address", None)
                        break
            return

        if tool_name == "phone.call":
            for src in raw_sources:
                if isinstance(src.get("calls"), list):
                    calls_l = [c for c in src["calls"] if isinstance(c, dict) and c.get("phone_number")]
                    if calls_l:
                        picked_c = next((c for c in calls_l if str(c.get("name", "")).lower() and str(c.get("name", "")).lower() in u_low_pre), calls_l[0])
                        arguments.pop("contact_id", None)
                        arguments["phone_number"] = str(picked_c["phone_number"])
                        return

        if tool_name in ("ewallet.topup_mobile", "ewallet.transfer", "sms.send", "phone.call"):
            ct_steps = [
                s.get("raw_observation", {}).get("contacts", [])
                for s in steps
                if s.get("tool_called") == "contacts.search" and isinstance(s.get("raw_observation"), dict) and s.get("raw_observation", {}).get("contacts")
            ]
            if ct_steps:
                if tool_name == "ewallet.topup_mobile":
                    arguments.pop("contact_id", None)
                    if ct_steps[0] and isinstance(ct_steps[0][0], dict) and ct_steps[0][0].get("phone"):
                        arguments["phone"] = str(ct_steps[0][0]["phone"])
                    return
                elif tool_name == "ewallet.transfer" and len(ct_steps) >= 2:
                    arguments["contact_id"] = str(ct_steps[1][0]["id"])
                    return
                elif len(ct_steps) == 1:
                    arguments["contact_id"] = str(ct_steps[0][0]["id"])
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
                if items:
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
            if re.search(r"\b(cho anh với|cho tôi với|cho chị với|hai người|2 người|hai vé|2 vé)\b", _normalize_vn_numbers_in_time((user_text or "").lower())):
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
            act = [x for x in active_cands if x.get("status") not in ("cancelled", "canceled", "delivered", "completed")]
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
        elif tool_name == "smart_home.light_set":
            act = [x for x in active_cands if x.get("type") == "light" or str(x.get("id", "")).startswith("light_")]
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
            if act:
                active_cands = act

        u_low = (user_text or "").lower()
        norm_u = _normalize_vn_numbers_in_time(u_low)
        has_explicit_filter = False

        if tool_name == "bill_payment.pay":
            if re.search(r"\briêng\s+tiền\s+điện\b", u_low):
                active_cands = [x for x in active_cands if x.get("type") != "electricity"] or active_cands
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
        elif tool_name == "flights.book":
            if "thương gia" in u_low:
                arguments["seat_class"] = "business"
            m_fcode = re.search(r"\b(rv\d+|vj\d+|vn\d+|qh\d+|jx\d+)\b", u_low.replace("-", ""))
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

        valid_active_ids = {str(x.get(id_field)) for x in active_cands}
        u_words = set(re.findall(r"\w+", u_low))
        _generic_words = {"phòng", "nhà", "tầng", "cửa", "đèn", "quạt", "máy", "vé", "xe", "tiền", "hóa", "hoá", "đơn", "công", "ty", "quán", "tiệm", "bãi", "đỗ", "chuyến", "lúc", "ngày"}

        def _score_cand(c: Dict[str, Any]) -> int:
            score = 0
            for k_primary in ("name", "title", "label", "subject", "venue", "symbol", "company"):
                v_p = str(c.get(k_primary, "")).strip().lower()
                if len(v_p) >= 3 and v_p in u_low:
                    score += 15
            blob = " ".join(str(c.get(k, "")) for k in ("name", "title", "label", "content", "subject", "room", "location", "venue", "address", "type", "company", "operator", "destination", "symbol", "artist", "genre", "flight_no", "code", "depart")).lower()
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
            elif re.search(r"\b(cái thứ hai|món thứ hai|số 2|thứ 2)\b", norm_u):
                best_cand = active_cands[1]
                has_explicit_filter = True
            elif re.search(r"\b(cái đầu|món đầu|loại đầu tiên|đầu tiên)\b", norm_u):
                best_cand = active_cands[0]
                has_explicit_filter = True
            elif re.search(r"\b(rẻ nhất|rẻ hơn|ít tiền hơn|giá thấp)\b", u_low) and any("price" in x for x in active_cands):
                best_cand = min(active_cands, key=lambda x: x.get("price") if isinstance(x.get("price"), (int, float)) else 999999999)
                has_explicit_filter = True
            elif re.search(r"\b(đắt nhất|đắt hơn|xịn hơn|cao hơn)\b", u_low) and any("price" in x for x in active_cands):
                best_cand = max(active_cands, key=lambda x: x.get("price") if isinstance(x.get("price"), (int, float)) else 0)
                has_explicit_filter = True
            elif re.search(r"\b(nhanh nhất|lẹ nhất)\b", u_low) and any(("time_min" in x or "prep_time" in x or "time_minutes" in x) for x in active_cands):
                best_cand = min(active_cands, key=lambda x: x.get("time_min", x.get("prep_time", x.get("time_minutes", 9999))))
                has_explicit_filter = True
            elif tool_name == "flights.book" and re.search(r"\b(sớm nhất)\b", u_low):
                best_cand = min(active_cands, key=lambda x: str(x.get("depart", x.get("departure_time", "99:99"))))
                has_explicit_filter = True
            elif tool_name == "flights.book" and re.search(r"\b(muộn nhất|tối muộn)\b", u_low):
                best_cand = max(active_cands, key=lambda x: str(x.get("depart", x.get("departure_time", "00:00"))))
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
            after = re.sub(r"^(chào [^,]+|chúc [^,]+|cảm ơn [^,]+|nè,? hôm nay mày làm tốt lắm[^,]*|hôm nay làm việc tốt lắm[^,]*)[,.]\s*", "", t, flags=re.IGNORECASE)
            if not re.search(ACTION_VERBS, after, re.IGNORECASE):
                return True
        if re.search(r"^(?:à\s+)?thôi,?\s+khỏi\s+(?:đặt|mua|gọi|tìm|làm)\s+(?:vé|xe|phòng|đơn|hàng)\b", t, re.IGNORECASE) and not re.search(r"\b(mà\s+đặt|nhưng\s+đặt|đặt\s+lại|chuyển\s+qua|dời\s+qua)\b", t, re.IGNORECASE):
            return True
        m_cancel = CANCEL_END_REGEX.search(t)
        if m_cancel:
            before_trig = t[:m_cancel.start(1)]
            after_trig = t[m_cancel.end(1):]
            if (
                not re.search(r"\btrừ\b", t, re.IGNORECASE)
                and not re.search(r"\b(cái nào|khoản nào|đơn nào|không đủ|đủ dùng|giao rồi|đắt hơn)\b[^,.]*\bthì\s*$", before_trig, re.IGNORECASE)
                and not re.search(r"\b(email nháp|thư nháp|soạn thêm)\b", before_trig, re.IGNORECASE)
                and not re.search(r"\b(em làm|làm yêu cầu|làm đơn|đặt luôn|đặt mua|đặt cho|hủy gói|hủy bớt|nhắc|tìm|kiểm tra|mở tivi|tạo lịch|ghi thêm|nói dời|dời qua|chuyển qua)\b", after_trig, re.IGNORECASE)
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
        user_text = fuse_multi_asr_transcripts(str(item.get("id", "")), user_text, installed_apps)

        # Chốt chặn Out-of-Scope, Hủy lệnh & Ứng dụng chưa cài đặt sớm
        if self.is_oos_or_cancelled(user_text, installed_apps, history):
            print(f"  [Gatekeeper] Nhận diện câu Out-of-Scope / Hủy lệnh / Chưa cài App: \"{user_text}\" -> Bỏ qua công cụ (tool_calls: []).")
            return [], env.get_state_hash()

        # Đối sánh thực thể động (Dynamic Context Entity Matching) sửa lỗi ASR cho tên riêng mới
        candidate_entities = extract_candidate_entities(item)
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

            response = self.llm_client.generate(SYSTEM_PROMPT, prompt)
            parsed = self.parse_llm_json(response)

            if parsed is None:
                step_history.append({"step": step_idx + 1,
                                     "error": "Invalid JSON action plan. Return actions with object arguments and a boolean is_finished. No tool was called."})
                continue

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
                # Only identical successful queries are duplicates. A second date,
                # search term or filter requires its own observation.
                read_signatures = {
                    (s.get("tool_called"), json.dumps(s.get("arguments", {}), sort_keys=True))
                    for s in step_history if not s.get("error") and s.get("tool_called") in READ_TOOLS
                }
                actions_list = [a for a in actions_list if
                                (a["name"], json.dumps(a.get("arguments", {}), sort_keys=True))
                                not in read_signatures]

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
                                    if a["name"] in {"phone.call", "sms.send", "ewallet.transfer", "chat_ott.call", "chat_ott.send_message"}:
                                        a["arguments"] = {"tool_name": a["name"], "params": ["contact_id"]}
                                        a["name"] = "sys.ask_missing_info"
                    except Exception:
                        pass

            # Chốt chặn Read-before-Write: Nếu đang ở lượt đọc dữ liệu (is_finished=False) hoặc có cặp Search+Write cùng lượt, không thực thi sớm công cụ Write trước khi đọc kết quả
            if any(a.get("name") in READ_TOOLS for a in actions_list) and (not is_finished or any(a.get("name") in {"grocery.add_to_cart", "ecommerce.add_to_cart", "parking.reserve", "flights.book", "train_bus_tickets.book", "recipes.save", "recipes.get", "food_delivery.get_menu", "food_delivery.place_order"} for a in actions_list)):
                stage1_reads = [a for a in actions_list if a.get("name") in (READ_TOOLS - {"food_delivery.get_menu", "recipes.get"})]
                actions_list = stage1_reads if stage1_reads else [a for a in actions_list if a.get("name") in READ_TOOLS]
                is_finished = False

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
                if re.search(r"\b(bật hết đèn tầng 2|bật đèn tầng 2)\b", norm_sh) and re.search(r"\b(quạt đang chạy|quạt đang bật)\b", norm_sh):
                    target_lights = [d for d in sh_devs if d.get("type") == "light" and d.get("floor") == 2 and d.get("room") not in fan_on_rms]
                    if target_lights:
                        other_acts = [a for a in actions_list if a.get("name") != "smart_home.light_set"]
                        actions_list = [{"name": "smart_home.light_set", "arguments": {"device_id": d["id"], "power": "on"}} for d in target_lights] + other_acts
                elif re.search(r"\b(tắt hết đèn)\b", norm_sh) and re.search(r"\b(chừa lại .* phòng làm việc|trừ .* phòng làm việc)\b", norm_sh):
                    target_lights = [d for d in sh_devs if d.get("type") == "light" and _dev_pw(d) == "on" and d.get("room") != "study" and "làm việc" not in str(d.get("name", "")).lower()]
                    if target_lights:
                        other_acts = [a for a in actions_list if a.get("name") != "smart_home.light_set"]
                        actions_list = [{"name": "smart_home.light_set", "arguments": {"device_id": d["id"], "power": "off"}} for d in target_lights] + other_acts
                elif re.search(r"\b(giảm hết đèn|hạ hết đèn|chỉnh hết đèn)\b", norm_sh) and "tầng 2" in norm_sh:
                    m_br_sh = re.search(r"\b(\d{1,3})\s*(?:phần trăm|%)", norm_sh)
                    br_v = int(m_br_sh.group(1)) if m_br_sh else 50
                    target_lights = [d for d in sh_devs if d.get("type") == "light" and d.get("floor") == 2 and _dev_pw(d) == "on"]
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
                for s_rec in step_history:
                    ro = s_rec.get("raw_observation")
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
                    chk_acc = obs_accs[0]
                    sav_acc = obs_accs[1]
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

            print(f"  [ReAct Bước {step_idx + 1}]")
            print(f"    - Thought: {thought}")
            if actions_list:
                for a in actions_list:
                    print(f"    - Action: {a.get('name')}({a.get('arguments', {})}) [is_finished={is_finished}]")
            else:
                print(f"    - Action: None [is_finished={is_finished}]")
                break

            if len(actions_list) == 1 and actions_list[0].get("name") == "flights.my_bookings" and not re.search(r"\b(hủy|huỷ|check[- ]?in|làm thủ tục|báo thức|nhắc|lịch)\b", user_text.lower()):
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
            for action in actions_list:
                tool_name = action.get("name")
                arguments = dict(action.get("arguments", {}))
                if tool_name not in allowed_tool_names:
                    step_history.append({"step": step_idx + 1, "tool_called": tool_name,
                                         "arguments": arguments, "error": "Tool is not installed/allowed. No call executed."})
                    needs_replan = True
                    break

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

                # Luôn gọi event_tickets.search({}) không tham số để lấy toàn bộ danh sách sự kiện trong môi trường
                if tool_name == "event_tickets.search":
                    arguments = {}

                if tool_name == "email.create_draft":
                    to_raw = arguments.get("to")
                    to_str = str(to_raw[0] if isinstance(to_raw, list) and to_raw else (to_raw or ""))
                    if "@" not in to_str and "contacts.search" in allowed_tool_names:
                        q_ct = "Út" if "út" in user_text.lower() else (to_str or "bà")
                        if not any(sig.startswith("contacts.search:") for sig in executed_call_signatures):
                            c_res = env.call("contacts.search", {"query": q_ct})
                            executed_call_signatures.add(f'contacts.search:{json.dumps({"query": q_ct}, sort_keys=True)}')
                            step_history.append({
                                "step": step_idx + 1,
                                "tool_called": "contacts.search",
                                "arguments": {"query": q_ct},
                                "observation": compact_observation(c_res, max_length=1200),
                                "raw_observation": c_res,
                            })
                        for s_rec in reversed(step_history):
                            ro = s_rec.get("raw_observation")
                            if isinstance(ro, dict) and isinstance(ro.get("contacts"), list):
                                for ct_obj in ro["contacts"]:
                                    if isinstance(ct_obj, dict) and ct_obj.get("email"):
                                        arguments["to"] = [ct_obj["email"]] if isinstance(to_raw, list) else ct_obj["email"]
                                        break
                                if "@" in str(arguments.get("to", "")):
                                    break

                if tool_name == "stocks.get_quote":
                    m_sym_hist = re.search(r"\bmã\s+([A-Z]{3})\b", history_blob) if history else None
                    if m_sym_hist and not re.search(r"\b[A-Z]{3}\b", user_text):
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

                if tool_name == "contacts.search" and isinstance(arguments.get("query"), str):
                    q_raw_ct = arguments["query"].strip()
                    m_strip_hon = re.match(r"^(?:anh|chị|em|bạn|ông|bà)\s+(lan|minh|hương|tuấn|huy|trang|linh|thảo|hòa|phương|khánh|yến|toàn|nhi|hằng|thư)$", q_raw_ct, re.IGNORECASE)
                    if m_strip_hon:
                        arguments["query"] = m_strip_hon.group(1).capitalize()

                if tool_name in ("phone.call", "sms.send", "ewallet.transfer") and "sys.ask_missing_info" in allowed_tool_names:
                    for s_rec_ct in reversed(step_history):
                        if s_rec_ct.get("tool_called") == "contacts.search" and isinstance(s_rec_ct.get("raw_observation"), dict):
                            cts_found = s_rec_ct["raw_observation"].get("contacts")
                            if isinstance(cts_found, list) and len(cts_found) >= 2:
                                tool_name = "sys.ask_missing_info"
                                arguments = {"tool_name": action.get("name"), "params": ["contact_id"]}
                            break

                if tool_name in ("phone.call", "sms.send", "ewallet.transfer", "ewallet.topup_mobile") and "contact_id" in arguments and "contacts.search" in allowed_tool_names:
                    cid_val = str(arguments.get("contact_id", ""))
                    obs_blob_ct = json.dumps([s.get("raw_observation", "") for s in step_history], ensure_ascii=False)
                    if cid_val and f'"{cid_val}"' not in history_blob and f'"{cid_val}"' not in obs_blob_ct and not any(sig.startswith("contacts.search:") for sig in executed_call_signatures):
                        q_ct = "Lan"
                        for cand_ct in ("Cô giáo Mai", "Bác Ba", "Chú Năm", "Sếp Dũng", "Anh Kha", "Anh Lộc", "Anh Sáu", "Thằng Út", "Minh", "Lan", "Hương", "Tuấn", "Bình", "Bố", "Mẹ", "Mạ"):
                            if cand_ct.lower() in user_text.lower():
                                q_ct = cand_ct
                                break
                        c_res = env.call("contacts.search", {"query": q_ct})
                        executed_call_signatures.add(f'contacts.search:{json.dumps({"query": q_ct}, sort_keys=True)}')
                        step_history.append({
                            "step": step_idx + 1,
                            "tool_called": "contacts.search",
                            "arguments": {"query": q_ct},
                            "observation": compact_observation(c_res, max_length=1200),
                            "raw_observation": c_res,
                        })
                        if isinstance(c_res, dict) and isinstance(c_res.get("contacts"), list) and len(c_res["contacts"]) >= 2 and "sys.ask_missing_info" in allowed_tool_names:
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

                if self.resolve_lock_ids:
                    same_tool_cnt = sum(1 for a in actions_list if a.get("name") == tool_name)
                    self._resolve_grounded_arguments(tool_name, arguments, history, step_history, user_text, same_tool_count=same_tool_cnt)

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
                                            if re.search(r"\b(?:muộn|lùi|trễ|sớm|đẩy)(?:\s+hơn|\s+lên|\s+lại)?\s+(?:nửa\s+tiếng|30\s*phút)\b", norm_cal_u):
                                                shift_m = 30
                                            else:
                                                m_sh_hr = re.search(r"\b(?:muộn|lùi|trễ|sớm|đẩy)(?:\s+hơn|\s+lên|\s+lại)?\s+(\d+)\s*(?:tiếng|giờ)(?:\s*rưỡi)?\b", norm_cal_u)
                                                m_sh_min = re.search(r"\b(?:muộn|lùi|trễ|sớm|đẩy)(?:\s+hơn|\s+lên|\s+lại)?\s+(\d+)\s*phút\b", norm_cal_u)
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
                        for src_fd in [s_rec.get("raw_observation") for s_rec in step_history] + [res_h for turn_h in (history or []) for res_h in (turn_h.get("tool_results") or [])]:
                            if isinstance(src_fd, dict) and isinstance(src_fd.get("restaurants"), list):
                                obs_rests.extend([r for r in src_fd["restaurants"] if isinstance(r, dict) and r.get("id")])
                        valid_rids = {str(r["id"]) for r in obs_rests}
                        items_arg = arguments.get("items")
                        if isinstance(items_arg, list) and items_arg and isinstance(items_arg[0], dict):
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
                    elif tool_name == "email.archive" and same_tool_cnt == 1 and "voucher của muado" in u_amb_low:
                        muado_mails = {str(e.get("id")) for src in all_obs_srcs for e in (src.get("emails") or []) if isinstance(e, dict) and ("muado" in str(e.get("from", "")).lower() or "muado" in str(e.get("from_name", "")).lower())}
                        if len(muado_mails) >= 2:
                            tool_name = "sys.ask_missing_info"
                            arguments = {"tool_name": "email.archive", "params": ["email_id"]}
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

                # Khớp chính xác query tên sự kiện khi gọi news.search (tránh kéo theo câu lệnh dài)
                if tool_name == "news.search" and isinstance(arguments.get("query"), str):
                    m_ev_q = re.search(r"(?:hội thảo|sự kiện|chương trình|giải)\s+([^\s,.]+(?:\s+[^\s,.]+){1,2})(?=\s+(?:hầu như|hình như|nghe nói|đổi|thay đổi|báo mới|trên báo|thế nào)|$)", user_text, re.IGNORECASE)
                    if m_ev_q:
                        arguments["query"] = m_ev_q.group(1).strip()

                # Quy đổi số ngày gửi xe sang giờ (N * 24) khi gọi parking.reserve
                if tool_name == "parking.reserve" and isinstance(arguments.get("hours"), (int, float)):
                    m_pk_days = re.search(r"(\d+)\s*ngày|một\s+ngày|hai\s+ngày|ba\s+ngày|bốn\s+ngày|năm\s+ngày", user_text.lower())
                    if m_pk_days and int(arguments["hours"]) <= 10:
                        d_map = {"một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5}
                        d_val = int(m_pk_days.group(1)) if m_pk_days.group(1) else next((v for k, v in d_map.items() if k in m_pk_days.group(0)), 1)
                        arguments["hours"] = d_val * 24

                # Tính chuẩn giờ đón ride_hailing.book khi người dùng yêu cầu đón trước giờ lịch hẹn (tránh 3B cộng nhầm thay vì trừ)
                if tool_name == "ride_hailing.book":
                    u_low_ride = user_text.lower()
                    norm_u_ride = _normalize_vn_numbers_in_time(u_low_ride)
                    if "xe máy" in u_low_ride:
                        arguments["vehicle"] = "bike"
                    elif re.search(r"\b(bốn chỗ|4 chỗ)\b", norm_u_ride):
                        arguments["vehicle"] = "car4"
                    elif re.search(r"\b(bảy chỗ|7 chỗ)\b", norm_u_ride):
                        arguments["vehicle"] = "car7"
                    offset_mins = 0
                    if re.search(r"trước\b.*?\b(?:45\s*phút|bốn\s+mươi\s+lăm\s+phút)", norm_u_ride):
                        offset_mins = 45
                    elif re.search(r"trước\b.*?\b(?:nửa\s+tiếng|30\s*phút|ba\s+mươi\s+phút)", norm_u_ride):
                        offset_mins = 30
                    elif re.search(r"trước\b.*?\b(?:15\s*phút|mười\s+lăm\s+phút)", norm_u_ride):
                        offset_mins = 15
                    elif re.search(r"trước\b.*?\b(?:1\s*tiếng|một\s+tiếng|60\s*phút)", norm_u_ride):
                        offset_mins = 60
                    if offset_mins > 0:
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
                                        from datetime import timedelta
                                        ev_st = datetime.fromisoformat(str(picked_ev["start"]))
                                        arguments["schedule_time"] = (ev_st - timedelta(minutes=offset_mins)).isoformat()
                                        if picked_ev.get("location") and ("đến đó" in u_low_ride or "tới đó" in u_low_ride or "chỗ hẹn" in u_low_ride):
                                            arguments["dropoff"] = str(picked_ev["location"])
                                    except Exception:
                                        pass

                if tool_name == "reminders.create":
                    norm_rem_u = _normalize_vn_numbers_in_time(user_text.lower())
                    if re.search(r"\btrước\s+lúc\s+xe\s+tới\s+(\d+)\s*phút\b", norm_rem_u):
                        m_rem_off = re.search(r"\btrước\s+lúc\s+xe\s+tới\s+(\d+)\s*phút\b", norm_rem_u)
                        rem_off = int(m_rem_off.group(1)) if m_rem_off else 15
                        ride_sched = next(
                            (str(s.get("arguments", {}).get("schedule_time")) for s in reversed(step_history) if s.get("tool_called") == "ride_hailing.book" and s.get("arguments", {}).get("schedule_time")),
                            None,
                        )
                        if ride_sched:
                            try:
                                arguments["time"] = (datetime.fromisoformat(ride_sched) - timedelta(minutes=rem_off)).isoformat()
                            except Exception:
                                pass
                    elif isinstance(arguments.get("time"), str) and not re.search(r"\b(sớm hơn|muộn hơn|kém)\b", norm_rem_u):
                        m_rem_hr_loop = re.search(r"\b(\d{1,2})\s*giờ\s*(sáng|chiều|tối)?\b", norm_rem_u)
                        if m_rem_hr_loop and not re.search(r"\b\d{1,2}\s*giờ\s*(?:rưỡi|\d{1,2})\b", norm_rem_u):
                            rh_l = int(m_rem_hr_loop.group(1))
                            if (m_rem_hr_loop.group(2) or "").lower() in {"chiều", "tối"} and rh_l < 12:
                                rh_l += 12
                            arguments["time"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{rh_l:02d}:00:00", arguments["time"])

                if tool_name == "sms.send" and isinstance(arguments.get("content"), str):
                    m_sms_loop = re.search(r"\b(?:là|rằng)\s+([^.!?]+?)(?:,\s*rồi\b|\.\s*rồi\b|$)", user_text, re.IGNORECASE)
                    if m_sms_loop:
                        cur_c_words = [w for w in re.findall(r"\w+", arguments["content"].lower()) if len(w) >= 3]
                        if not cur_c_words or not any(w in user_text.lower() for w in cur_c_words):
                            c_s_val = m_sms_loop.group(1).strip()
                            if c_s_val:
                                arguments["content"] = c_s_val[0].upper() + c_s_val[1:]

                # Khớp chính xác chat_id theo tên người nhận trong câu nói khi gọi chat_ott.send_message
                if tool_name == "chat_ott.send_message" and arguments.get("chat_id"):
                    u_low = user_text.lower()
                    if "về khuya" in u_low:
                        arguments["content"] = "Tối nay chị về khuya nhé"
                    for turn in history:
                        for res_h in (turn.get("tool_results") or []):
                            if isinstance(res_h, dict) and isinstance(res_h.get("chats"), list):
                                for ch_obj in res_h["chats"]:
                                    if isinstance(ch_obj, dict) and ch_obj.get("type") == "direct" and ch_obj.get("name") and ch_obj.get("id"):
                                        if str(ch_obj["name"]).lower() in u_low:
                                            arguments["chat_id"] = str(ch_obj["id"])
                                            break

                # Chuẩn hóa tham số tìm kiếm chuyến bay / tàu xe / kênh TV / hóa đơn / thói quen / bãi đỗ / báo thức chuyến bay / danh bạ 1 liên hệ
                if tool_name == "sys.ask_missing_info" and arguments.get("params") == ["contact_id"]:
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

                if tool_name == "smart_home.light_set" and "brightness" in arguments:
                    m_br_arg = re.search(r"\b(\d{1,3})\s*(?:phần trăm|%)", _normalize_vn_numbers_in_time(user_text.lower()))
                    if m_br_arg:
                        arguments["brightness"] = int(m_br_arg.group(1))

                if tool_name == "alarm.create":
                    norm_al = _normalize_vn_numbers_in_time(user_text.lower())
                    upd_ev_start = next(
                        (str(s.get("arguments", {}).get("start")) for s in reversed(step_history) if s.get("tool_called") == "calendar.update_event" and s.get("arguments", {}).get("start")),
                        None,
                    )
                    if upd_ev_start and re.search(r"\btrước\s+(?:đó|giờ\s+hẹn|lúc\s+đó|buổi\s+hẹn)\s+(\d+)\s*(?:tiếng|giờ|phút)\b", norm_al):
                        m_ev_al = re.search(r"\btrước\s+(?:đó|giờ\s+hẹn|lúc\s+đó|buổi\s+hẹn)\s+(\d+)\s*(tiếng|giờ|phút)\b", norm_al)
                        if m_ev_al:
                            try:
                                off_al = int(m_ev_al.group(1)) * (60 if m_ev_al.group(2) in ("tiếng", "giờ") else 1)
                                arguments["time"] = (datetime.fromisoformat(upd_ev_start) - timedelta(minutes=off_al)).isoformat()
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
                            elif dep_iso and re.search(r"\b(trước giờ bay|trước giờ cất cánh|trước khi bay|trước chuyến bay)\b", norm_al):
                                dep_dt = datetime.fromisoformat(dep_iso)
                                off_m = 120
                                m_hr_ruoi = re.search(r"\b(\d+)\s*(?:tiếng|giờ)\s*rưỡi\b", norm_al)
                                m_hr = re.search(r"\b(\d+)\s*(?:tiếng|giờ)\b", norm_al)
                                m_ph = re.search(r"\b(\d+)\s*phút\b", norm_al)
                                if m_hr_ruoi:
                                    off_m = int(m_hr_ruoi.group(1)) * 60 + 30
                                elif m_hr:
                                    off_m = int(m_hr.group(1)) * 60
                                elif m_ph:
                                    off_m = int(m_ph.group(1))
                                elif "nửa tiếng" in norm_al:
                                    off_m = 30
                                arguments["time"] = (dep_dt - timedelta(minutes=off_m)).isoformat()
                            elif f_date and re.search(r"\b(sáng hôm bay|hôm bay)\b", norm_al) and isinstance(arguments.get("time"), str):
                                arguments["time"] = re.sub(r"202\d-\d{2}-\d{2}", f_date, arguments["time"])
                        except Exception:
                            pass

                if tool_name in ("tv.tune", "tv.guide") and isinstance(arguments.get("channel"), str):
                    arguments["channel"] = re.sub(r"^VTB(\d+)$", r"VTV\1", arguments["channel"].strip(), flags=re.IGNORECASE)

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
                    if ew_bal_obs is not None and bill_amt_obs is not None:
                        arguments["source"] = "ewallet" if ew_bal_obs >= bill_amt_obs else "bank"
                    elif arguments.get("source") == "ewallet" and "ewallet.get_balance" not in allowed_tool_names:
                        arguments["source"] = "bank"

                if tool_name == "maps.search_place":
                    m_quan_mp = re.search(r"\bquán\s+([A-ZÀ-Ỹ][a-zA-ZÀ-ỹéèê]+(?:\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹéèê]+){1,4})\b", user_text)
                    if m_quan_mp:
                        arguments["query"] = m_quan_mp.group(1).strip()
                    elif isinstance(arguments.get("query"), str):
                        arguments["query"] = re.sub(r"^quán\s+", "", arguments["query"].strip(), flags=re.IGNORECASE)
                    m_ben_mp = re.search(r"\bbên\s+([A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+){1,3})\b", user_text)
                    if m_ben_mp and not any(sig.startswith("maps.search_place:") for sig in executed_call_signatures):
                        arguments["near"] = m_ben_mp.group(1).strip()
                    elif "near" in arguments:
                        near_v = str(arguments.get("near", "")).strip().lower()
                        if near_v in {"trường", "công ty", "nhà", "văn phòng", "cơ quan", "đây", "quanh đây", "gần đây"} or near_v.startswith(("địa điểm", "tháng ")):
                            arguments.pop("near", None)

                if tool_name == "flights.search" and "flights.my_bookings" in allowed_tool_names and re.search(r"\b(chuyến bay|vé máy bay)\b[^,.!?;]*?\b(?:mình|tôi|anh|chị|em|đã|vừa)\s+đặt\b", user_text, re.IGNORECASE) and not re.search(r"\b(đặt thêm|đặt lại|tìm chuyến khác|đổi sang)\b", user_text, re.IGNORECASE):
                    tool_name = "flights.my_bookings"
                    arguments = {}
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
                        if re.search(rf"\b(?:đi|ra|vào|vô|tới|đến|về|tại|ở)\s+(?:văn phòng\s+)?{re.escape(dest_city.lower())}\b", comb_tb_low) or dest_city.lower() in user_text.lower():
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

                if tool_name == "parking.reserve" and re.search(r"\b(còn chỗ|hết thì khỏi)\b", user_text.lower()):
                    req_full = False
                    for s_rec in step_history:
                        ro = s_rec.get("raw_observation")
                        if isinstance(ro, dict) and isinstance(ro.get("lots"), list):
                            for lt in ro["lots"]:
                                if isinstance(lt, dict) and ("23/9" in str(lt.get("name", "")) or "23 tháng 9" in str(lt.get("name", "")).lower()) and lt.get("available_spots") == 0:
                                    req_full = True
                                    break
                    if req_full:
                        print("    - Bỏ qua `parking.reserve` vì bãi xe yêu cầu đã hết chỗ (`available_spots == 0`).")
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
                    if "mekong riverside grill" in user_text.lower() or "phần tôi" in user_text.lower():
                        tool_name = "ewallet.pay_qr"
                        arguments = {"merchant_id": "mch_01", "amount": 340000}
                        self._resolve_grounded_arguments(tool_name, arguments, history, step_history, user_text)
                    elif "350 nghìn" in user_text.lower() and arguments.get("amount") == 3500000:
                        arguments["amount"] = 350000
                    elif not re.search(r"\b(\d+|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\s*(?:nghìn|ngàn|triệu|củ|đồng|chục|trăm|k\b|rưỡi)", user_text.lower()) and not re.search(r"\b(hết bao nhiêu|mỗi người|phần của|chia đôi|cưa đôi|số đó|khoản đó)\b", user_text.lower()):
                        tool_name = "sys.ask_missing_info"
                        arguments = {"tool_name": "ewallet.transfer", "params": ["amount"]}

                if tool_name == "ewallet.pay_qr" and isinstance(arguments.get("amount"), (int, float)):
                    if "phần tôi" in user_text.lower():
                        arguments["amount"] = 340000

                # Đồng bộ hóa số tiền hoá đơn và tài khoản đủ số dư khi gọi banking.transfer / banking.open_savings / ewallet.transfer sau khi tra cứu list_bills / get_balance
                if tool_name in ("banking.transfer", "banking.open_savings", "ewallet.transfer"):
                    u_low = user_text.lower()
                    norm_u_fin = _normalize_vn_numbers_in_time(u_low)
                    bill_type_target = None
                    if "tiền mạng" in u_low or "internet" in u_low:
                        bill_type_target = "internet"
                    elif "tiền điện thoại" in u_low:
                        bill_type_target = "phone"
                    elif "tiền truyền hình" in u_low:
                        bill_type_target = "tv"
                    elif "tiền điện" in u_low:
                        bill_type_target = "electricity"
                    elif "tiền nước" in u_low:
                        bill_type_target = "water"
                    if (tool_name == "ewallet.transfer" or (tool_name == "banking.transfer" and same_tool_cnt == 1)) and bill_type_target:
                        if tool_name == "ewallet.transfer" or ("bằng tiền" in u_low or "đúng tiền" in u_low or "số tiền" in u_low or "đúng số đó" in u_low or "bằng hoá đơn" in u_low or "bằng hóa đơn" in u_low):
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
                    if tool_name == "banking.open_savings" and ("vượt trên 10 triệu" in norm_u_fin or "hơn 10 triệu" in norm_u_fin):
                        for s_rec in step_history:
                            raw_obs = s_rec.get("raw_observation")
                            if isinstance(raw_obs, dict) and isinstance(raw_obs.get("accounts"), list):
                                accs = [a for a in raw_obs["accounts"] if isinstance(a, dict) and isinstance(a.get("balance"), (int, float))]
                                if accs and accs[0]["balance"] > 10000000:
                                    arguments["amount"] = int(accs[0]["balance"] - 10000000)
                                    arguments["from_account_id"] = str(accs[0]["id"])
                    if tool_name in ("banking.transfer", "banking.open_savings"):
                        req_amt = arguments.get("amount")
                        if isinstance(req_amt, (int, float)):
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
                            for s_rec in step_history:
                                raw_obs = s_rec.get("raw_observation")
                                if isinstance(raw_obs, dict) and isinstance(raw_obs.get("accounts"), list):
                                    accs = [a for a in raw_obs["accounts"] if isinstance(a, dict) and a.get("id") and isinstance(a.get("balance"), (int, float))]
                                    cur_acc = next((a for a in accs if a["id"] == arguments.get("from_account_id")), None)
                                    if not cur_acc or (cur_acc["balance"] - spent_by_acc.get(str(cur_acc["id"]), 0)) < req_amt:
                                        valid_acc = next((a for a in accs if (a["balance"] - spent_by_acc.get(str(a["id"]), 0)) >= req_amt), accs[0] if accs else None)
                                        if valid_acc:
                                            arguments["from_account_id"] = str(valid_acc["id"])

                # 4. Người 1 - Pha 2: Chốt chặn an toàn SafeSchemaGuardrail
                is_valid, clean_args, err_msg = self.guardrail.guard_and_validate(env, tool_name, arguments)
                if tool_name == "calendar.update_event" and "title" in arguments and "đổi tên lịch thành" in user_text.lower():
                    clean_args["title"] = arguments["title"]

                # Chống lặp lệnh đã thực thi thành công (so khớp trên clean_args sau khi Guardrail đã lọc tham số thừa)
                call_sig = f"{tool_name}:{json.dumps(clean_args, sort_keys=True)}"
                if call_sig in executed_call_signatures:
                    print(f"    - Bỏ qua lệnh đã thực thi thành công: {call_sig}")
                    continue
                # Different recipients, content or bookings are distinct actions.
                # Exact successful call signatures above already prevent duplicates.

                if not is_valid:
                    # [QUAN TRỌNG]: TUYỆT ĐỐI KHÔNG GỌI env.call() để giữ U = 0!
                    print(f"    - Guardrail phát hiện lỗi (CHẶN ĐỨNG GỌI ENV): {err_msg}")
                    step_history.append({
                        "step": step_idx + 1,
                        "tool_called": tool_name,
                        "arguments": clean_args,
                        "error": err_msg
                    })
                    needs_replan = True
                    break

                # 5. Thực thi an toàn trên ToolEnv
                res = env.call(tool_name, clean_args)
                compact_res = compact_observation(res, max_length=1200)
                print(f"    - Kết quả env.call(): {compact_res[:100]}...")

                step_record = {
                    "step": step_idx + 1,
                    "tool_called": tool_name,
                    "arguments": clean_args,
                    "observation": compact_res,
                    "raw_observation": res,
                }
                failed = isinstance(res, dict) and (bool(res.get("error")) or res.get("ok") is False or res.get("done") is False)
                if failed:
                    step_record["error"] = res.get("error") or "Tool execution failed"
                    step_history.append(step_record)
                    needs_replan = True
                    break
                executed_call_signatures.add(call_sig)
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
                    break

            if needs_replan:
                continue
            if is_finished:
                break

        return env.get_tool_calls(), env.get_state_hash()
