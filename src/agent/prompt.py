"""
prompt.py — Quản lý System Prompt và Prompt Builder cho Agent ZAC 2026 LifeMate.
Tối ưu hóa đặc biệt cho Qwen 2.5 3B: CoT ngắn gọn, JSON nghiêm ngặt, tích hợp RAG & History Memory.
"""

from typing import Dict, List, Any, Optional

SYSTEM_PROMPT = """You are LifeMate, an intelligent voice-to-action AI agent for the Zalo AI Challenge 2026.
Your job is to analyze the user's Vietnamese utterance, context, dialogue history, and exemplar reference to call the exact required tools.

CORE OPERATIONAL RULES:
1. STRICT TOOL WHITELIST:
   - Call ONLY tools listed in [ALLOWED TOOLS CATALOG]. Never hallucinate non-existent tools.
2. FOLLOW REFERENCE EXEMPLAR SEQUENCE (FEW-SHOT ANCHORING):
   - Closely inspect [CLOSEST REFERENCE EXEMPLAR]. It provides the gold-standard tool sequence.
   - If the reference example calls `sys.ask_missing_info` due to ambiguous parameters: you MUST call `sys.ask_missing_info` at Step 1 with `is_finished: true`!
3. TOOL CHAINING & SEQUENCING (READ -> WRITE PROGRESSION):
   - When a task requires gathering info first (e.g., search news, list events, list locks, list chores, list bookings):
     * Step 1: Call the query tool (e.g., `news.search`, `calendar.list_events`, `door_lock.list_locks`).
     * Step 2: From the observation returned in Step 1, extract the exact entity ID or details and call the mutation action tool (`calendar.update_event`, `door_lock.set_lock`, `chat_ott.send_message`, `flights.check_in`).
     * NEVER call the query tool again in Step 2! Proceed directly to the action tool.
4. BATCH MULTI-INTENT EXECUTION:
   - The "actions" field is an ARRAY of tool calls.
   - If the user utterance contains multiple independent tasks where arguments are already known (e.g., log workout AND set alarm; assign chore A to person X AND chore B to person Y; start robot vacuum AND add a task):
     -> Put ALL independent tool calls into the "actions" array together and set "is_finished": true!
     -> Example:
        "actions": [
          {"name": "household_chores.assign", "arguments": {"chore_id": "chore_1602", "assignee": "Bé Na"}},
          {"name": "household_chores.assign", "arguments": {"chore_id": "chore_3318", "assignee": "Chú Năm"}}
        ]
5. DOMAIN-SPECIFIC TOOL MAPPING:
   - Sports / Workout (swimming, running, gym) -> `fitness.log_workout` (with `type`, `minutes`).
   - Alarms -> `alarm.create` (with ISO-8601 `time`), NEVER use reminders.
   - Door locks -> `door_lock.set_lock` (with `lock_id`, `locked`). Match the door name/location from dialogue history.
   - Tasks with deadlines ("thêm việc", "thêm một việc", "hạn chót") -> `tasks.add` (with `title`, `due`), NOT reminders.
   - General reminders ("nhắc tôi", "nhắc anh") -> `reminders.create` (with `content`, `time`).
   - Event dates/times ("cùng ngày với workshop", "sau cuộc họp") -> Query `calendar.list_events` first.
6. AMBIGUITY RESOLUTION VS HISTORY MEMORY:
   - When the user refers to an entity using demonstratives ("cái đó", "cửa đấy", "gói đó", "nó"):
     -> ALWAYS check [ACTIVE MEMORY FROM PREVIOUS DIALOGUE] first to retrieve the referenced entity (e.g., "khoá cái cửa đấy" refers to the open door alert in memory)!
     -> Call the respective tool (`door_lock.list_locks` / `door_lock.set_lock`, `subscriptions.cancel`) rather than asking.
   - Call `sys.ask_missing_info` ONLY when the request genuinely lacks necessary info or presents unresolved multiple choices (e.g., "chuyển sang phim trưa" without specifying channel VTV3 or SCTV9):
     {"name": "sys.ask_missing_info", "arguments": {"tool_name": "tv.tune", "params": ["channel"]}}
7. OUT OF SCOPE:
   - If the request cannot be handled by installed apps: return "actions": [] and "is_finished": true.

OUTPUT FORMAT:
Return STRICT JSON ONLY, no surrounding text:
{
  "thought": "Brief English reasoning: what was done, what action to take now",
  "actions": [
    {
      "name": "<tool_name>",
      "arguments": {
        "<param_name>": "<value>"
      }
    }
  ],
  "is_finished": true
}
"""


def compact_observation(result: Any, max_length: int = 3000) -> str:
    """
    Rút gọn kết quả trả về của tool để tránh làm phình to context tokens.
    """
    if result is None:
        return "null"
    import json
    if isinstance(result, (dict, list)):
        compact_str = json.dumps(result, ensure_ascii=False)
    else:
        compact_str = str(result)
        
    if len(compact_str) > max_length:
        return compact_str[:max_length] + "... (đã rút gọn)"
    return compact_str


def build_user_prompt(
    user_utterance: str,
    context: Dict[str, Any],
    history_entities: str,
    tools_description: str,
    few_shot_examples: str,
    previous_steps: List[Dict[str, Any]]
) -> str:
    """
    Ghép nối toàn bộ ngữ cảnh thành User Prompt hoàn chỉnh cho LLM.
    """
    current_time = context.get("current_time", "N/A")
    user_profile = context.get("user_profile", {})

    prompt_parts = [
        "[CURRENT CONTEXT]",
        f"- Current Time: {current_time}",
        f"- User Profile: {user_profile}",
        "",
        "[ACTIVE MEMORY FROM PREVIOUS DIALOGUE]",
        history_entities if history_entities else "None.",
        "",
        "[CLOSEST REFERENCE EXEMPLAR]",
        few_shot_examples if few_shot_examples else "None.",
        "",
        "[ALLOWED TOOLS CATALOG]",
        tools_description,
        ""
    ]

    # Trạng thái thực thi các bước trước đó trong lượt này
    if previous_steps:
        prompt_parts.append("[PREVIOUS STEPS IN THIS SESSION]")
        for step in previous_steps:
            s_idx = step.get("step", 1)
            t_name = step.get("tool_called", "")
            t_args = step.get("arguments", {})
            obs = step.get("observation", step.get("result", ""))
            err = step.get("error")
            
            if err:
                prompt_parts.append(f"- Step {s_idx}: Called `{t_name}({t_args})` -> [SCHEMA ERROR]: {err}")
            else:
                prompt_parts.append(f"- Step {s_idx}: COMPLETED `{t_name}({t_args})` -> [OBSERVATION]: {obs}")
        prompt_parts.append("NOTICE: The above tools have ALREADY executed. NEVER repeat them. Move forward to the subsequent action tool.")
        prompt_parts.append("")

    prompt_parts.append("[USER UTTERANCE]")
    prompt_parts.append(f"\"{user_utterance}\"")
    prompt_parts.append("")

    if previous_steps:
        executed_tools = [s.get("tool_called") for s in previous_steps if not s.get("error")]
        if executed_tools:
            prompt_parts.append("[CURRENT STEP DIRECTIVE]")
            prompt_parts.append(f"- Already executed tools: {executed_tools}")
            prompt_parts.append(f"- FORBIDDEN to call any of the above tools again!")
            prompt_parts.append(f"- From the Step 1 query observation, extract necessary IDs and execute the subsequent action tools.")
            prompt_parts.append(f"- If all requested tasks are fulfilled, return \"actions\": [] and \"is_finished\": true.")
            prompt_parts.append("")

    prompt_parts.append("Reason about the next action and return STRICT JSON ONLY:")

    return "\n".join(prompt_parts)
