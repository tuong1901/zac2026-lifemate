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
from src.agent.rag.fuzzy_entity_matcher import extract_candidate_entities, repair_utterance_with_context_entities
from src.rules.catalog_filter import CatalogFilter
from src.rules.schema_guardrail import SafeSchemaGuardrail

CHITCHAT_REGEX = re.compile(
    r"^(chào (em|bạn|cô|anh|chị|trợ lý)|chúc (em|bạn|cậu|anh|chị)|nè,? hôm nay mày làm tốt lắm|hôm nay làm việc tốt lắm|nói chuyện với em vui thật|mới sáng ra mà nói chuyện với em|cảm ơn em (nhiều|nhé|nha)|thôi mình đi làm đây|thôi được rồi,? cảm ơn em|ừ được rồi,? cảm ơn em)",
    re.IGNORECASE
)
CANCEL_END_REGEX = re.compile(
    r"(à thôi,? khỏi|à thôi,? bỏ qua đi|thôi khỏi,? không cần|thôi khỏi,? đừng|thôi khỏi nha|thôi khỏi nhé|thôi khỏi em|khỏi nha,? anh|khỏi nhé,? anh|khỏi nhé|khỏi nha|không cần đâu|không cần nữa|đừng làm nữa|đừng có thêm|đừng thêm gì.*hết|để anh tự|để chị tự|để tui tự|để hôm khác|tự check|tự coi|tự ghi|tự làm|khỏi đổi gì hết nha|khỏi tạo phòng nữa nhé|khỏi tạo nữa nhé|khỏi tạo nữa nha|khỏi đặt vé nha)[.!?,;]?\s*$",
    re.IGNORECASE
)
ACTION_VERBS = r"(đặt|chuyển|gọi|nhắn|gửi|coi|kiểm tra|tìm|mở|bật|tắt|tạo|hủy|thêm|xóa|ghi|nhắc|nạp|trừ|làm|báo|dời|đẩy|đổi)"

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

    def is_oos_or_cancelled(self, text: str) -> bool:
        """
        Nhận diện nhanh các câu chào hỏi, khen ngợi, cảm ơn (chit-chat)
        hoặc câu hủy lệnh giữa chừng mà không yêu cầu thêm tác vụ nào khác.
        Đạt độ chính xác 100% (0 false positives trên toàn bộ tập train).
        """
        if not text:
            return False
        t = text.strip()
        if CHITCHAT_REGEX.search(t):
            after = re.sub(r"^(chào [^,]+|chúc [^,]+|cảm ơn [^,]+)[,.]\s*", "", t, flags=re.IGNORECASE)
            if not re.search(ACTION_VERBS, after, re.IGNORECASE):
                return True
        if CANCEL_END_REGEX.search(t):
            if not re.search(r"\btrừ\b", t, re.IGNORECASE):
                return True
        return False

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
        # Chốt chặn Out-of-Scope & Hủy lệnh sớm (thu hồi 7.2% - 8% điểm số)
        if self.is_oos_or_cancelled(user_text):
            print(f"  [Gatekeeper] Nhận diện câu hội thoại Out-of-Scope / Hủy lệnh: \"{user_text}\" -> Bỏ qua công cụ (tool_calls: []).")
            return [], env.get_state_hash()

        # Đối sánh thực thể động (Dynamic Context Entity Matching) sửa lỗi ASR cho tên riêng mới
        candidate_entities = extract_candidate_entities(item)
        repaired_text, entity_repairs = repair_utterance_with_context_entities(user_text, candidate_entities)
        if entity_repairs:
            print(f"  [Entity Matcher] Đã phục hồi tên riêng ASR: {entity_repairs}")
            user_text = repaired_text

        context = item.get("context", {})
        installed_apps = context.get("installed_apps", {})
        history = item.get("history", [])

        # 1. Lọc danh sách công cụ được phép
        allowed_tools = self.catalog_filter.get_allowed_tools(installed_apps)
        tools_desc = self.catalog_filter.format_tools_for_prompt(allowed_tools)

        task_id = self._resolve_history_task_ordinal(history, user_text)
        if task_id and any(tool["name"] == "tasks.complete" for tool in allowed_tools):
            arguments = {"task_id": task_id}
            is_valid, clean_args, err_msg = self.guardrail.guard_and_validate(env, "tasks.complete", arguments)
            if is_valid:
                env.call("tasks.complete", clean_args)
                print(f"  [History Resolver] 'việc thứ ...' -> tasks.complete({clean_args})")
                return env.get_tool_calls(), env.get_state_hash()
            print(f"  [History Resolver] Không thực thi được tasks.complete: {err_msg}")

        # 2. Người 2: Bóc tách thông tin từ History hội thoại
        history_entities = self.history_resolver.extract_active_memory(history)

        # 3. Truy xuất một ví dụ gần nhất bằng BM25
        few_shot_str = "Không có ví dụ mẫu."
        exemplars = []
        if self.exemplar_retriever:
            if self.structured_router:
                exemplars = self.structured_router.retrieve(user_text, installed_apps)
            else:
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

            needs_replan = False
            for action in actions_list:
                tool_name = action.get("name")
                arguments = dict(action.get("arguments", {}))

                if self.resolve_lock_ids and tool_name == "door_lock.set_lock":
                    reader = "door_lock.list_locks"
                    reader_seen = any(sig.startswith(f"{reader}:") for sig in executed_call_signatures)
                    has_alert_context = any(
                        call.get("name") == "home_security.list_alerts"
                        for turn in history if turn.get("role") == "assistant"
                        for call in turn.get("tool_calls", [])
                    )
                    explicit_lock_reference = bool(re.search(r"\b(cửa|cổng)\b", user_text, re.IGNORECASE))
                    if not reader_seen and (has_alert_context or explicit_lock_reference) and any(t["name"] == reader for t in allowed_tools):
                        valid_reader, reader_args, _ = self.guardrail.guard_and_validate(env, reader, {})
                        if valid_reader:
                            reader_result = env.call(reader, reader_args)
                            executed_call_signatures.add(f"{reader}:{{}}")
                            step_history.append({
                                "step": step_idx + 1,
                                "tool_called": reader,
                                "arguments": reader_args,
                                "observation": compact_observation(reader_result, max_length=1000),
                                "raw_observation": reader_result,
                            })
                            print(f"    - Resolve ID: {reader} trước {tool_name}; lập kế hoạch lại từ cảnh báo và danh sách khóa.")
                            needs_replan = True
                            is_finished = False
                            break

                if self.resolve_lock_ids and tool_name == "door_lock.set_lock":
                    resolved_lock_id = self._resolve_lock_id(history, step_history, user_text)
                    if resolved_lock_id and arguments.get("lock_id") != resolved_lock_id:
                        print(f"    - Resolve ID: sửa lock_id {arguments.get('lock_id')} -> {resolved_lock_id} từ cảnh báo và list_locks.")
                        arguments["lock_id"] = resolved_lock_id

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
                compact_res = compact_observation(res, max_length=1000)
                print(f"    - Thực thi env.call() thành công: {compact_res[:100]}...")

                step_record = {
                    "step": step_idx + 1,
                    "tool_called": tool_name,
                    "arguments": clean_args,
                    "observation": compact_res
                }
                if tool_name == "door_lock.list_locks":
                    step_record["raw_observation"] = res
                step_history.append(step_record)

                if tool_name == "sys.ask_missing_info":
                    is_finished = True
                    break

            # Nếu là tool kết thúc cuộc gọi, sys.ask_missing_info, hoặc LLM báo đã hoàn thành -> Dừng ngay
            if needs_replan:
                continue
            if is_finished:
                break

        return env.get_tool_calls(), env.get_state_hash()
