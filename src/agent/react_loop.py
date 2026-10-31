"""
react_loop.py — Vòng lặp điều phối ReAct Agent (Nấc thang 1 hoàn chỉnh).
Tích hợp:
- BM25 Exemplar Retriever (Thành viên 1 - Pha 1)
- Safe Schema Guardrail (Thành viên 1 - Pha 2)
- History Memory Resolver (Thành viên 2)
- Observation Compactor & Chống lặp vô hạn
"""

import json
import re
from typing import List, Dict, Any, Tuple, Optional

from src.agent.prompt import SYSTEM_PROMPT, build_user_prompt, compact_observation
from src.agent.rag.exemplar_retriever import BM25ExemplarRetriever
from src.agent.rag.history_resolver import HistoryResolver
from src.rules.catalog_filter import CatalogFilter
from src.rules.schema_guardrail import SafeSchemaGuardrail


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
        max_steps: int = 7
    ):
        self.llm_client = llm_client
        self.catalog_filter = catalog_filter
        self.exemplar_retriever = exemplar_retriever
        self.history_resolver = history_resolver or HistoryResolver()
        self.guardrail = guardrail or SafeSchemaGuardrail()
        self.max_steps = max_steps

    def parse_llm_json(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Trích xuất và phân tích JSON từ chuỗi sinh của LLM.
        Hỗ trợ dọn dẹp markdown ```json ... ``` và sửa lỗi cú pháp nhẹ.
        """
        if not response:
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
            return json.loads(clean)
        except Exception:
            return None

    def run(self, env: Any, item: Dict[str, Any], user_text: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Thực thi trọn vẹn một mẫu yêu cầu từ người dùng.
        
        Args:
            env: Phiên làm việc ToolEnv của mẫu hiện tại
            item: Dữ liệu mẫu (context, history, v.v.)
            user_text: Khẩu lệnh của người dùng sau ASR hoặc audio_text
        
        Returns:
            (tool_calls, state_hash): Đúng định dạng yêu cầu của BTC
        """
        context = item.get("context", {})
        installed_apps = context.get("installed_apps", {})
        history = item.get("history", [])

        # 1. Lọc danh sách công cụ được phép
        allowed_tools = self.catalog_filter.get_allowed_tools(installed_apps)
        tools_desc = self.catalog_filter.format_tools_for_prompt(allowed_tools)

        # 2. Người 2: Bóc tách thông tin từ History hội thoại
        history_entities = self.history_resolver.extract_active_memory(history)

        # 3. Người 1: Truy xuất Top-2 ví dụ mẫu bằng BM25
        few_shot_str = "Không có ví dụ mẫu."
        if self.exemplar_retriever:
            exemplars = self.exemplar_retriever.retrieve(
                query_text=user_text,
                installed_apps=installed_apps,
                top_k=1
            )
            few_shot_str = self.exemplar_retriever.format_exemplars_for_prompt(exemplars)

        step_history: List[Dict[str, Any]] = []
        executed_call_signatures = set()
        failed_call_signatures = set()
        repeated_call_signatures = set()

        # Vòng lặp ReAct từng bước
        for step_idx in range(self.max_steps):
            # Ghép User Prompt đầy đủ ngữ cảnh
            prompt = build_user_prompt(
                user_utterance=user_text,
                context=context,
                history_entities=history_entities,
                tools_description=tools_desc,
                few_shot_examples=few_shot_str,
                previous_steps=step_history
            )

            # Gọi LLM sinh suy luận bước tiếp theo
            response = self.llm_client.generate(SYSTEM_PROMPT, prompt)
            parsed = self.parse_llm_json(response)

            if not parsed:
                print(f"  [ReAct Bước {step_idx + 1}] Không phân tích được JSON từ LLM: {response[:100]}...")
                break

            thought = parsed.get("thought", "")
            is_finished = parsed.get("is_finished", False)

            # Hỗ trợ cả danh sách "actions" lẫn single "action"
            actions_raw = parsed.get("actions")
            if actions_raw is None:
                action_single = parsed.get("action")
                actions_raw = [action_single] if action_single else []
            elif isinstance(actions_raw, dict):
                actions_raw = [actions_raw]

            # Lọc các action hợp lệ (có trường name)
            actions_list = [a for a in actions_raw if isinstance(a, dict) and a.get("name")]

            print(f"  [ReAct Bước {step_idx + 1}]")
            print(f"    - Thought: {thought}")
            if actions_list:
                for a in actions_list:
                    print(f"    - Action: {a.get('name')}({a.get('arguments', {})}) [is_finished={is_finished}]")
            else:
                print(f"    - Action: None [is_finished={is_finished}]")
                break

            # Kiểm tra nếu toàn bộ actions trong lượt này đều đã thực thi xong từ trước -> Hoàn thành
            all_duplicate = all(
                f"{a.get('name')}:{json.dumps(a.get('arguments', {}), sort_keys=True)}" in executed_call_signatures
                for a in actions_list
            )
            if all_duplicate:
                print(f"    - Tất cả các lệnh trong lượt này đều đã được thực thi từ trước: Hoàn tất trọn vẹn.")
                break

            for action in actions_list:
                tool_name = action.get("name")
                arguments = action.get("arguments", {})

                # Chống lặp lệnh đã thực thi thành công
                call_sig = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
                if call_sig in executed_call_signatures:
                    print(f"    - Bỏ qua lệnh đã thực thi thành công: {call_sig}")
                    continue

                # 4. Người 1 - Pha 2: Chốt chặn an toàn SafeSchemaGuardrail
                is_valid, clean_args, err_msg = self.guardrail.guard_and_validate(env, tool_name, arguments)

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

                # 5. Thực thi an toàn trên ToolEnv
                res = env.call(tool_name, clean_args)
                compact_res = compact_observation(res, max_length=3000)
                print(f"    - Thực thi env.call() thành công: {compact_res[:100]}...")

                step_history.append({
                    "step": step_idx + 1,
                    "tool_called": tool_name,
                    "arguments": clean_args,
                    "observation": compact_res
                })

                if tool_name == "sys.ask_missing_info":
                    is_finished = True
                    break

            # Nếu là tool kết thúc cuộc gọi, sys.ask_missing_info, hoặc LLM báo đã hoàn thành -> Dừng ngay
            if is_finished:
                break

        return env.get_tool_calls(), env.get_state_hash()
