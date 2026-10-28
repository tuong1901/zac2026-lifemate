import json
from typing import List, Dict, Any, Tuple
from src.agent.prompt import SYSTEM_PROMPT, build_user_prompt
from src.rules.catalog_filter import CatalogFilter

class ReActAgent:
    """
    Tác nhân ReAct: Tương tác trực tiếp với ToolEnv của zaic_tools.
    Gọi tool -> Xem kết quả -> Lập luận bước tiếp theo -> Dừng lại khi xong.
    """
    def __init__(self, llm_client, catalog_filter: CatalogFilter, max_steps: int = 6):
        self.llm_client = llm_client
        self.catalog_filter = catalog_filter
        self.max_steps = max_steps

    def run(self, env, item: dict, user_text: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        env: ToolEnv instance từ zaic_tools
        item: Mẫu dữ liệu từ train.jsonl hoặc public_test.jsonl
        user_text: Câu lệnh sau khi ASR gỡ băng (hoặc lấy từ audio_text)
        """
        context = item.get("context", {})
        installed_apps = context.get("installed_apps", {})
        history = item.get("history", [])

        # 1. Lọc chỉ lấy các tool được phép cho mẫu này
        allowed_tools = self.catalog_filter.get_allowed_tools(installed_apps)
        tools_desc = self.catalog_filter.format_tools_for_prompt(allowed_tools)

        step_history = []

        for step_idx in range(self.max_steps):
            prompt = build_user_prompt(
                user_utterance=user_text,
                context=context,
                history=history,
                tools_description=tools_desc,
                previous_steps=step_history
            )

            # Gọi LLM sinh bước tiếp theo
            response = self.llm_client.generate(SYSTEM_PROMPT, prompt)
            
            try:
                # Trích xuất JSON từ phản hồi của LLM
                clean_json = response.strip()
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_json:
                    clean_json = clean_json.split("```")[1].split("```")[0].strip()
                    
                parsed = json.loads(clean_json)
            except Exception as e:
                # Fallback nếu format bị vỡ
                break

            action = parsed.get("action")
            is_finished = parsed.get("is_finished", False)

            # Nếu không có hành động hoặc LLM báo hoàn thành
            if not action or action.get("name") is None:
                break

            tool_name = action["name"]
            arguments = action.get("arguments", {})

            # 2. Thực thi tool trên môi trường ảo ToolEnv
            res = env.call(tool_name, arguments)
            
            step_history.append({
                "step": step_idx + 1,
                "tool_called": tool_name,
                "arguments": arguments,
                "result": res
            })

            # Nếu đã gọi công cụ sys.ask_missing_info thì kết thúc luôn
            if tool_name == "sys.ask_missing_info" or is_finished:
                break

        return env.get_tool_calls(), env.get_state_hash()
