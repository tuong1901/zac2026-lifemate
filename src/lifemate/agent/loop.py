from __future__ import annotations

import json
import time
from typing import Any, Callable

from .controller import ToolController
from lifemate.models.policy import PolicyContextOverflow


def run_episode(*, sample_id: str, initial_messages: list[dict[str, str]], controller: ToolController,
                policy: Callable[[list[dict[str, str]]], str], max_steps: int = 12,
                log_sink: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    transcript = list(initial_messages)
    controller.state.waiting_for_user = False
    if controller.state.status == "waiting_for_user":
        controller.state.status = "in_progress"
    started = time.perf_counter()
    failures = 0
    status = "max_steps"
    for index in range(max_steps):
        step_started = time.perf_counter()
        raw, output, observations, error = None, None, [], None
        stop_after_error = False
        try:
            raw = policy(transcript)
            output = controller.parse(raw)
            observations = controller.execute(output)
        except PolicyContextOverflow as exc:
            failures += 1
            error = f"{type(exc).__name__}: {exc}"
            status = "context_overflow"
            stop_after_error = True
        except Exception as exc:  # includes policy/parser/schema/tool errors; loop remains bounded
            failures += 1
            error = f"{type(exc).__name__}: {exc}"
        record = {"sample_id": sample_id, "step_index": index, "model_output": raw,
                  "parsed_action": output.get("actions") if output else None,
                  "validation_result": {"ok": error is None, "error": error}, "tool_result": observations,
                  "workflow_state": controller.state.to_dict(), "execution_time_seconds": time.perf_counter()-step_started,
                  "error_message": error}
        if log_sink:
            log_sink(record)
        if error:
            transcript.append({"role": "user", "content": f"Lỗi bộ điều khiển/tool đã được xác minh: {error}. Hãy sửa hoặc hỏi thêm thông tin; không khẳng định thao tác thành công."})
            if stop_after_error:
                break
            continue
        if observations:
            transcript.append({"role": "assistant", "content": raw})
            transcript.append({"role": "user", "content": "[TOOL OBSERVATION]\n" + json.dumps(observations[0], ensure_ascii=False)})
        if controller.state.waiting_for_user:
            status = "waiting_for_user"
            break
        if output["is_finished"]:
            status = "finished"
            break
    return {"sample_id": sample_id, "status": status, "steps": index+1, "failure_count": failures,
            "tool_calls": controller.env.get_tool_calls(), "state_hash": controller.env.get_state_hash(),
            "workflow_state": controller.state.to_dict(), "transcript": transcript,
            "latency_seconds": time.perf_counter()-started}
