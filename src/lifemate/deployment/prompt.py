from __future__ import annotations

import json


SYSTEM_PROMPT = """Bạn là LifeMate, trợ lý điều khiển ứng dụng bằng tiếng Việt. Dựa vào yêu cầu hiện tại, ngữ cảnh, lịch sử, công cụ được phép và các quan sát thật để chọn đúng MỘT hành động kế tiếp. Sau mỗi kết quả công cụ, đánh giá lại trước khi gọi công cụ tiếp theo. Nếu thiếu thông tin bắt buộc, gọi sys.ask_missing_info. Không tự khẳng định công cụ thành công; chỉ kết luận sau quan sát xác nhận. Không lặp lại thao tác ghi đã thành công. Nếu mọi việc hoàn tất hoặc ngoài phạm vi, trả actions rỗng và is_finished=true. Chỉ xuất JSON hợp lệ đúng dạng: {\"thought\":\"\",\"actions\":[{\"name\":\"tool.name\",\"arguments\":{}}],\"is_finished\":false}. Mỗi lượt tối đa một action; không Markdown hay văn bản ngoài JSON."""


def build_messages(item: dict, current_text: str, tool_specs: dict[str, dict], audio_text_by_path: dict[str, str] | None = None) -> list[dict[str, str]]:
    audio_text_by_path = audio_text_by_path or {}
    context = item.get("context", {})
    installed = context.get("installed_apps", {})
    allowed = {name for app in installed.values() for name in app.get("tools", [])} | {"sys.ask_missing_info"}
    tools = [spec for name, spec in tool_specs.items() if name in allowed]
    system = SYSTEM_PROMPT + "\n\n[CURRENT CONTEXT]\n" + json.dumps({k: v for k, v in context.items() if k != "installed_apps"}, ensure_ascii=False)
    system += "\n\n[AUTHORIZED TOOL CATALOG]\n" + json.dumps(tools, ensure_ascii=False)
    messages = [{"role": "system", "content": system}]
    for turn in item.get("history", []):
        role = turn.get("role")
        if role == "user":
            transcript = turn.get("text") or turn.get("audio_text") or audio_text_by_path.get(turn.get("audio", ""), "[Không có bản chép lời của lượt trước]")
            messages.append({"role": "user", "content": str(transcript)})
        elif role == "assistant":
            text = str(turn.get("text", ""))
            if turn.get("tool_calls"):
                text += "\n[Previous tool calls/results already reflected in initial environment]\n" + json.dumps({"tool_calls": turn.get("tool_calls"), "tool_results": turn.get("tool_results", [])}, ensure_ascii=False)
            messages.append({"role": "assistant", "content": text or "[Đã xử lý lượt trước]"})
    messages.append({"role": "user", "content": "[CURRENT REQUEST]\n" + current_text})
    return messages
