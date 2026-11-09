"""
src/agent/v2/two_stage_agent.py — Bộ điều phối Agent Hai Giai Đoạn (Two-Stage ReAct Agent).

Kiến trúc:
- Stage 1 (Router & Compressor): Đọc toàn bộ context + audio transcript và danh sách app tóm gọn.
  -> Xuất ra: selected_apps + condensed_history.
  -> Nếu là Chit-chat / Hủy lệnh: Thoát ngay trong ~0.7s mà không cần vào Stage 2.
  -> Áp dụng Mạng lưới An toàn (Companion Network): Đảm bảo 100% recall (có thể lấy thừa nhưng không được thiếu).
- Stage 2 (Targeted ReAct Planner): Chỉ nạp vào prompt các tool thuộc các app đã chọn ở Stage 1
  -> Giảm 75% lượng token prompt.
  -> Tăng tốc độ giải mã của mô hình LLM.
  -> Tránh hoàn toàn việc mô hình bị nhầm lẫn giữa các công cụ khác miền.
"""

import os
import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Set

from src.agent.execution import (
    GuardedToolEnv,
    RunBudget,
    RunLimitReached,
    ActionFeedback,
    result_failed,
)
from src.agent.prompt import (
    clean_asr_transcript,
    fuse_multi_asr_transcripts,
    _get_multi_asr_variants,
    compact_observation,
    READ_TOOLS,
)
from src.agent.rag.fuzzy_entity_matcher import (
    extract_candidate_entities,
    repair_utterance_with_context_entities,
)
from .prompt_v2 import (
    APP_DOMAIN_CATALOG,
    STAGE1_SYSTEM_PROMPT,
    STAGE2_SYSTEM_PROMPT,
    build_stage1_router_prompt,
    build_stage2_planner_prompt,
    expand_selected_apps_with_safety_net,
)

# Nhận diện sớm câu thoại Chit-chat hoặc Hủy lệnh
CHITCHAT_PATTERNS = re.compile(
    r"^(chào (em|bạn|cô|anh|chị|trợ lý|buổi sáng)|chúc (em|bạn|cậu|anh|chị)|nè,? hôm nay mày làm tốt lắm|"
    r"hôm nay làm việc tốt lắm|nói chuyện với em vui thật|cảm ơn em (nhiều|nhé|nha)|thôi mình đi làm đây|"
    r"thôi được rồi,? cảm ơn em|ừ được rồi,? cảm ơn em)",
    re.IGNORECASE,
)

CANCEL_PATTERNS = re.compile(
    r"(à thôi,? khỏi|à thôi,? bỏ qua đi|thôi khỏi,? không cần|thôi khỏi,? đừng|"
    r"thôi khỏi (?:chuyển|đặt|gọi|nhắn|mua|làm|tắt|bật|hủy|xóa|thêm|ghi|nhắc|nạp)\b.*nữa|"
    r"thôi khỏi nha|thôi khỏi nhé|khỏi đặt nữa|không cần đâu|không cần nữa|đừng làm nữa|"
    r"để anh tự|để chị tự|để tui tự|để hôm khác|tự check|tự làm)(?:[^.!?]*[.!?]?\s*)?$",
    re.IGNORECASE,
)


class TwoStageReActAgent:
    """
    Agent 2 Giai đoạn tối ưu hóa thời gian và độ chính xác cho ZAC 2026.
    """

    def __init__(
        self,
        llm_client: Any,
        catalog_filter: Any,
        guardrail: Optional[Any] = None,
        max_steps: int = 4,
        time_budget: float = 30.0,
        verbose: bool = True,
    ):
        self.llm_client = llm_client
        self.catalog_filter = catalog_filter
        self.guardrail = guardrail
        self.max_steps = max_steps
        self.time_budget = time_budget
        self.verbose = verbose
        self.last_run_stats: Dict[str, Any] = {}

    def _extract_json_from_llm(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Trích xuất và parse JSON an toàn từ phản hồi của LLM.
        Hỗ trợ markdown fenced code blocks ```json ... ``` hoặc khối JSON thô.
        """
        if not text:
            return None
        t = text.strip()

        # 1. Thử match khối markdown ```json ... ```
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, re.IGNORECASE)
        if code_block:
            raw_json = code_block.group(1).strip()
            try:
                return json.loads(raw_json)
            except Exception:
                pass

        # 2. Thử tìm cặp ngoặc nhọn đầu tiên và cuối cùng {...}
        first_brace = t.find("{")
        last_brace = t.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            sub = t[first_brace : last_brace + 1]
            try:
                return json.loads(sub)
            except Exception:
                pass

        # 3. Thử parse trực tiếp
        try:
            return json.loads(t)
        except Exception:
            return None

    def _fallback_keyword_apps(
        self, user_text: str, installed_apps: Dict[str, Any]
    ) -> List[str]:
        """
        Fallback chọn app bằng đối sánh từ khóa nếu Stage 1 LLM gặp sự cố parse.
        """
        u_lower = user_text.lower()
        matched = set()
        installed_set = (
            set(installed_apps.keys())
            if isinstance(installed_apps, dict)
            else set(installed_apps)
        )

        keyword_rules = [
            ("alarm", ["báo thức", "đặt báo", "hẹn giờ"]),
            ("weather", ["thời tiết", "mưa", "nắng", "nhiệt độ", "dự báo"]),
            ("gold_price", ["giá vàng", "vàng sjc", "vàng nhẫn"]),
            ("fuel_price", ["giá xăng", "giá dầu", "xăng dầu", "ron 95", "ron95"]),
            ("lottery", ["xổ số", "vé số", "dò vé", "kết quả"]),
            ("smart_home", ["đèn", "quạt", "điều hòa", "máy lạnh", "rèm", "ổ cắm", "phòng khách", "phòng ngủ", "tầng"]),
            ("door_lock", ["khoá cửa", "mở cửa", "mã khách", "chốt cửa"]),
            ("robot_vacuum", ["robot", "hút bụi", "lau nhà"]),
            ("appliances", ["máy giặt", "nồi cơm", "lò vi sóng", "lọc khí"]),
            ("bill_payment", ["tiền điện", "tiền nước", "tiền mạng", "internet", "truyền hình", "hóa đơn", "autopay"]),
            ("banking", ["ngân hàng", "tài khoản", "số dư", "sao kê", "chuyển khoản", "tiết kiệm"]),
            ("ewallet", ["ví", "nạp ví", "chuyển ví", "nạp điện thoại", "zalopay", "momo"]),
            ("food_delivery", ["đặt món", "quán ăn", "giao về", "suất", "bún bò", "cơm", "pizza"]),
            ("ride_hailing", ["đặt xe", "chở tới", "xe 4 chỗ", "xe máy", "ô tô"]),
            ("flights", ["vé máy bay", "chuyến bay", "check-in", "checkin", "hàng không"]),
            ("calendar", ["lịch", "sự kiện", "cuộc hẹn", "họp"]),
            ("reminders", ["nhắc", "nhắc nhở", "hẹn nhắc"]),
            ("tasks", ["việc cần làm", "công việc", "to-do", "hạn chót"]),
            ("contacts", ["danh bạ", "số điện thoại", "liên hệ"]),
            ("phone", ["gọi điện", "gọi cho", "gọi"]),
            ("sms", ["nhắn tin", "sms", "tin nhắn"]),
            ("chat_ott", ["zalo", "nhắn zalo", "nhóm chat"]),
            ("notes", ["ghi chú", "lưu vào ghi chú", "note"]),
        ]

        for app_name, keywords in keyword_rules:
            if app_name in installed_set and any(kw in u_lower for kw in keywords):
                matched.add(app_name)

        if not matched:
            # Nếu không tìm thấy từ khóa đặc trưng, an toàn nhất là lấy toàn bộ installed_apps
            return list(installed_set)
        return list(matched)

    def run(
        self, env: Any, item: Dict[str, Any], user_text: str
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Điều phối toàn bộ quá trình thực thi 2 Giai đoạn trên một mẫu kiểm thử.
        """
        t_start = time.perf_counter()
        budget = RunBudget.start(self.time_budget, self.max_steps * 2)

        # Lấy thông tin ngữ cảnh
        context = item.get("context", {})
        installed_apps = context.get("installed_apps", {})
        history = item.get("history", [])
        sample_id = str(item.get("id", ""))
        current_time_str = context.get("current_time", "")

        # 0. Chuẩn hóa & Hậu xử lý ASR tiếng Việt
        user_text = fuse_multi_asr_transcripts(sample_id, user_text, installed_apps)
        user_text = clean_asr_transcript(user_text)

        candidate_entities = extract_candidate_entities(item)
        repaired_text, entity_repairs = repair_utterance_with_context_entities(
            user_text, candidate_entities
        )
        if entity_repairs:
            user_text = repaired_text

        # 1. Kiểm tra nhanh Chit-chat / Hủy lệnh bằng Regex (Fast Exit Rule 1)
        if CHITCHAT_PATTERNS.search(user_text) or CANCEL_PATTERNS.search(user_text):
            if self.verbose:
                print(
                    f"  [Fast Exit] Phát hiện câu Chit-chat / Hủy lệnh: \"{user_text}\" -> Thoát ngay không gọi tool."
                )
            t_total = time.perf_counter() - t_start
            self.last_run_stats = {
                "sample_id": sample_id,
                "stage1_time": t_total,
                "stage2_time": 0.0,
                "total_time": t_total,
                "selected_apps": [],
                "expanded_apps": ["sys"],
                "num_tools_stage2": 0,
                "steps_count": 0,
            }
            return env.get_tool_calls(), env.get_state_hash()

        # ======================================================================
        # GIAI ĐOẠN 1: ROUTER & HISTORY COMPRESSOR (STAGE 1)
        # ======================================================================
        t_s1_start = time.perf_counter()
        stage1_prompt = build_stage1_router_prompt(
            user_text=user_text,
            history=history,
            installed_apps=installed_apps,
            current_time_str=current_time_str,
        )

        selected_apps: List[str] = []
        condensed_history: str = "Không có."

        try:
            budget.before_call()
            stage1_response = self.llm_client.generate(
                prompt=stage1_prompt,
                system_prompt=STAGE1_SYSTEM_PROMPT,
                max_tokens=256,
                temperature=0.0,
            )
            stage1_json = self._extract_json_from_llm(stage1_response)

            if isinstance(stage1_json, dict):
                selected_apps = stage1_json.get("selected_apps", [])
                condensed_history = stage1_json.get(
                    "condensed_history", "Không có."
                )
            else:
                # LLM không trả về JSON hợp lệ -> Dùng fallback keyword
                selected_apps = self._fallback_keyword_apps(user_text, installed_apps)
        except Exception as e:
            if self.verbose:
                print(f"  [Stage 1 Exception] {e} -> Dùng fallback keyword")
            selected_apps = self._fallback_keyword_apps(user_text, installed_apps)

        t_s1_end = time.perf_counter()
        t_stage1 = t_s1_end - t_s1_start

        # Fast Exit Rule 2: Stage 1 kết luận không cần gọi app nào (và không có từ khóa hành động)
        if not selected_apps:
            if self.verbose:
                print(
                    f"  [Stage 1] Không nhận diện thấy app cần gọi -> Thoát sớm sau {t_stage1:.2f}s."
                )
            t_total = time.perf_counter() - t_start
            self.last_run_stats = {
                "sample_id": sample_id,
                "stage1_time": t_stage1,
                "stage2_time": 0.0,
                "total_time": t_total,
                "selected_apps": [],
                "expanded_apps": ["sys"],
                "num_tools_stage2": 0,
                "steps_count": 0,
            }
            return env.get_tool_calls(), env.get_state_hash()

        # Mở rộng tập app với Mạng Lưới Đồng Hành (Companion Network) đảm bảo Safe Recall
        expanded_apps = expand_selected_apps_with_safety_net(
            selected_apps, installed_apps
        )

        if self.verbose:
            print(
                f"  [Stage 1 Hoàn tất trong {t_stage1:.2f}s] Apps đề xuất: {selected_apps} -> Mở rộng an toàn: {expanded_apps}"
            )
            print(f"  [Stage 1 Tóm tắt lịch sử] {condensed_history}")

        # ======================================================================
        # GIAI ĐOẠN 2: TARGETED REACT PLANNER & EXECUTOR (STAGE 2)
        # ======================================================================
        t_s2_start = time.perf_counter()

        # Lọc danh sách công cụ: CHỈ GIỮ CÁC CÔNG CỤ THUỘC expanded_apps
        target_installed_apps = {
            app_id: installed_apps.get(app_id, {})
            for app_id in expanded_apps
            if app_id in installed_apps
        }
        allowed_tools = self.catalog_filter.get_allowed_tools(target_installed_apps)
        allowed_tools_desc = self.catalog_filter.format_tools_for_prompt(allowed_tools)
        read_tools = {
            tool["name"]
            for tool in allowed_tools
            if tool.get("kind") == "read"
            or (not tool.get("kind") and tool["name"] in READ_TOOLS)
        }

        guarded = GuardedToolEnv(
            env=env,
            allowed_tools=allowed_tools,
            guardrail=self.guardrail,
            budget=budget,
            read_tools=read_tools,
        )

        observations_history = []
        steps_count = 0

        try:
            for step in range(1, self.max_steps + 1):
                steps_count = step
                # Ghép lịch sử quan sát các bước trước
                obs_text = "\n".join(observations_history) if observations_history else ""

                stage2_prompt = build_stage2_planner_prompt(
                    user_text=user_text,
                    condensed_history=condensed_history,
                    allowed_tools_desc=allowed_tools_desc,
                    current_time_str=current_time_str,
                    observations_history=obs_text,
                )

                budget.before_call()
                stage2_response = self.llm_client.generate(
                    prompt=stage2_prompt,
                    system_prompt=STAGE2_SYSTEM_PROMPT,
                    max_tokens=256,
                    temperature=0.0,
                )

                step_data = self._extract_json_from_llm(stage2_response)
                if not step_data or not isinstance(step_data, dict):
                    if self.verbose:
                        print(f"  [Stage 2 Bước {step}] Không parse được JSON -> Dừng.")
                    break

                thought = step_data.get("thought", "")
                action = step_data.get("action")

                if self.verbose:
                    action_repr = json.dumps(action, ensure_ascii=False) if action else "None"
                    print(f"  [Stage 2 Bước {step}] Thought: {thought}")
                    print(f"    -> Action: {action_repr}")

                # Nếu LLM xác định không cần gọi thêm công cụ -> Dừng ReAct
                if not action or action is None or not isinstance(action, dict):
                    break

                tool_name = action.get("name")
                tool_args = action.get("arguments", {})

                if not tool_name:
                    break

                # Thực thi hành động an toàn qua GuardedToolEnv
                try:
                    res = guarded.call(tool_name, tool_args)
                    res_compact = compact_observation(res, max_length=1200)
                    observations_history.append(
                        f"Bước {step}: Gọi `{tool_name}` với tham số {json.dumps(tool_args, ensure_ascii=False)} => Kết quả: {res_compact}"
                    )
                    if self.verbose:
                        print(f"    => Kết quả env: {res_compact[:180]}...")
                except ActionFeedback as fb:
                    observations_history.append(
                        f"Bước {step}: Gọi `{tool_name}` bị từ chối => Phản hồi: {fb.message}"
                    )
                    if self.verbose:
                        print(f"    => Bị guardrail chặn: {fb.message}")

                # Nếu là sys.ask_missing_info thì chờ người dùng -> Kết thúc lượt
                if tool_name == "sys.ask_missing_info":
                    break

        except RunLimitReached as limit_err:
            if self.verbose:
                print(f"  [Stage 2 Giới hạn thời gian / cuộc gọi] {limit_err}")

        t_s2_end = time.perf_counter()
        t_stage2 = t_s2_end - t_s2_start
        t_total = time.perf_counter() - t_start

        self.last_run_stats = {
            "sample_id": sample_id,
            "stage1_time": t_stage1,
            "stage2_time": t_stage2,
            "total_time": t_total,
            "selected_apps": selected_apps,
            "expanded_apps": expanded_apps,
            "num_tools_stage2": len(allowed_tools),
            "steps_count": steps_count,
        }

        return guarded.get_tool_calls(), guarded.get_state_hash()
