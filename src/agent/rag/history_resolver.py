"""
history_resolver.py — Bóc tách thực thể và tóm tắt ngữ cảnh từ lịch sử hội thoại (ZAC 2026 LifeMate).
Phụ trách: Thành viên 2 (Context & History Memory Resolver).

Mục tiêu cốt lõi:
- Giải quyết hiện tượng đa lượt (Multi-turn Coreference Resolution):
  Người dùng nói "cái cửa đấy", "Kevin Walsh dặn", "đơn hôm nọ", "hội thảo lúc nãy".
- Khai thác tối đa `assistant.text` và `assistant.tool_results` (vốn có sẵn dạng Text/JSON trong cả train lẫn test).
- Tóm tắt cô đọng (dưới 200 tokens) để không làm phình to Prompt của LLM.
"""

import json
from typing import List, Dict, Any, Optional


class HistoryResolver:
    """
    Bộ giải quyết ngữ cảnh đa lượt:
    Trích xuất các thực thể quan trọng (tên người, thiết bị, cảnh báo, tin nhắn gần nhất)
    từ danh sách các lượt hội thoại trước đó trong `history`.
    """

    def __init__(self, max_history_turns: int = 10):
        self.max_history_turns = max_history_turns

    def extract_active_memory(self, history: Optional[List[Dict[str, Any]]]) -> str:
        """
        Duyệt qua history và tạo chuỗi tóm tắt ngữ cảnh súc tích cho LLM Prompt.
        """
        if not history:
            return "Không có lịch sử hội thoại trước đó."

        memory_items: List[str] = []

        # Chỉ lấy tối đa N lượt gần nhất để tránh tràn context
        recent_history = history[-self.max_history_turns * 2:]

        turn_idx = 1
        for item in recent_history:
            role = item.get("role")

            if role == "user":
                # Lượt của người dùng
                user_text = item.get("audio_text")
                if user_text:
                    memory_items.append(f"- Người dùng (lượt cũ): \"{user_text}\"")

            elif role == "assistant":
                # Lượt của trợ lý: Chứa thông tin cực kỳ quý giá
                assistant_text = item.get("text", "")
                if assistant_text:
                    memory_items.append(f"- Trợ lý đã phản hồi: \"{assistant_text}\"")

                # Bóc tách thực thể từ tool_results của lượt trước
                tool_results = item.get("tool_results", [])
                extracted_entities = self._extract_entities_from_results(tool_results)
                if extracted_entities:
                    memory_items.extend(extracted_entities)

                turn_idx += 1

        if not memory_items:
            return "Không có thông tin thực thể đặc biệt từ lịch sử."

        return "\n".join(memory_items)

    def _extract_entities_from_results(self, tool_results: List[Dict[str, Any]]) -> List[str]:
        """
        Bóc tách các trường thực thể quan trọng từ kết quả tool của assistant.
        """
        entities = []

        for res in tool_results:
            if not isinstance(res, dict) or not res.get("ok", False):
                continue

            # 1. Cảnh báo an ninh / cảm biến (home_security.list_alerts)
            if "alerts" in res:
                alerts = res["alerts"]
                for alt in alerts[:3]:
                    loc = alt.get("location", "")
                    msg = alt.get("message", "")
                    if loc or msg:
                        entities.append(f"  * Cảnh báo gần nhất: [{loc}] {msg}")

            # 2. Tin nhắn OTT / Zalo / SMS (chat_ott.read_chat, sms.read)
            if "messages" in res:
                messages = res["messages"]
                for msg in messages[-2:]:
                    sender = msg.get("from", "")
                    content = msg.get("content", "")
                    if sender and content:
                        entities.append(f"  * Tin nhắn từ {sender}: \"{content}\"")

            # 3. Lịch tivi / Kênh truyền hình (tv.guide)
            if "programs" in res:
                programs = res["programs"]
                for p in programs[:3]:
                    ch = p.get("channel", "")
                    title = p.get("title", "")
                    start = p.get("start", "")
                    if ch and title:
                        entities.append(f"  * Lịch TV: Kênh {ch} - \"{title}\" (bắt đầu {start})")

            # 4. Lịch sự kiện (calendar.list_events)
            if "events" in res:
                events = res["events"]
                for ev in events[:3]:
                    ev_id = ev.get("id", "")
                    title = ev.get("title", "")
                    start = ev.get("start", "")
                    loc = ev.get("location", "")
                    entities.append(f"  * Sự kiện lịch: [{ev_id}] \"{title}\" ({start} tại {loc})")

            # 5. Khóa cửa (door_lock.list_locks)
            if "locks" in res:
                locks = res["locks"]
                for lk in locks:
                    if not lk.get("locked", True):  # Cửa đang mở
                        entities.append(f"  * Thiết bị khoá: ID '{lk.get('id')}' ({lk.get('name')} - {lk.get('location')}) đang mở")

            # 6. Danh bạ (contacts.search)
            if "contacts" in res:
                contacts = res["contacts"]
                for ct in contacts:
                    if isinstance(ct, dict):
                        entities.append(f"  * Liên hệ: ID '{ct.get('id')}' - Tên: {ct.get('name')} - Quan hệ: {ct.get('relation', '')} - SĐT: {ct.get('phone')} - Email: {ct.get('email', '')}")

        return entities
