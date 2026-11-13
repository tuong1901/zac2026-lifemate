def categorize(*, call_match: bool, state_match: bool, status: str, calls: list[dict]) -> str:
    if status == "max_steps":
        return "timeout_or_loop_limit"
    if not call_match and not calls:
        return "intent_or_completion_decision"
    if not call_match:
        return "tool_selection_order_or_arguments"
    if not state_match:
        return "tool_execution_or_final_state"
    return "incorrect_completion_decision"
