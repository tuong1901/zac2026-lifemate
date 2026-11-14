"""Explicit, uncorrected text input contract for supervised tool decisions.

This module performs no intent classification, ASR correction or early exit.
It can also be used by an inference runner when evaluating this contract.
"""
import json


RAW_SYSTEM_PROMPT = """Bạn là LifeMate, trợ lý thực hiện yêu cầu bằng công cụ.
Đọc câu nói hiện tại, toàn bộ lịch sử và kết quả các bước đã thực hiện.
Chỉ gọi công cụ được cung cấp, đúng schema và enum. Không bịa ID hoặc thông tin
còn thiếu. Nếu cần tra cứu, gọi công cụ đọc trước rồi chờ kết quả. Nếu còn mơ hồ
và không thể giải quyết bằng tra cứu, dùng sys.ask_missing_info theo schema.
Phân biệt rút lại yêu cầu chưa thực hiện với yêu cầu hủy một đối tượng đã tồn tại:
yêu cầu thứ hai có thể cần gọi công cụ hủy. Đọc toàn câu và ngữ cảnh, không quyết
định chỉ vì có từ "thôi" hay "hủy". Ưu tiên nội dung đính chính cuối cùng.
Khi không cần hành động, người dùng rút lại yêu cầu, điều kiện không thỏa mãn
hoặc yêu cầu đã hoàn tất, trả actions=[] và is_finished=true.
Không lặp lại thao tác đã thành công nếu người dùng không yêu cầu thực hiện lại.
Chỉ trả JSON: {"thought":"","actions":[{"name":"tên_tool","arguments":{}}],"is_finished":false}.
Sau công cụ đọc còn cần kết quả để quyết định bước tiếp theo, đặt is_finished=false.
Với sys.ask_missing_info, kết thúc bước hiện tại bằng is_finished=true.
"""


def build_raw_user_prompt(user_text, context, history, allowed_tools, previous_steps):
    """Preserve all textual values and observations; exclude only audio paths."""
    dialogue = [{k: v for k, v in turn.items() if k != 'audio'} for turn in history]
    observations = [
        {'tool_name': step['tool_called'], 'arguments': step.get('arguments', {}),
         'result': step['raw_observation']}
        for step in previous_steps
    ]
    # Serialize full tool schemas rather than inferring tool eligibility from intent.
    sections = [
        ('CONTEXT', context),
        ('ALLOWED TOOLS CATALOG', allowed_tools),
        ('FULL DIALOGUE HISTORY', dialogue),
        ('PREVIOUS STEPS IN THIS SESSION', observations),
        ('USER UTTERANCE', user_text),
    ]
    return '\n\n'.join('[' + name + ']\n' + json.dumps(value, ensure_ascii=False)
                        for name, value in sections)
