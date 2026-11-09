"""
src/agent/v2 — Kiến trúc Agent Hai Giai Đoạn (Two-Stage Agent Architecture) cho ZAC 2026.
Giai đoạn 1 (Stage 1): Phân loại Ý định, Nhận diện Ứng dụng Mục tiêu & Rút gọn Lịch sử Hội thoại.
Giai đoạn 2 (Stage 2): Lập Kế hoạch Chi tiết và Thực thi ReAct trên Tập Công cụ Tối Giản.
"""

from .prompt_v2 import (
    APP_DOMAIN_CATALOG,
    COMPANION_APPS_MAPPING,
    build_stage1_router_prompt,
    build_stage2_planner_prompt,
    expand_selected_apps_with_safety_net,
)
from .two_stage_agent import TwoStageReActAgent

__all__ = [
    "APP_DOMAIN_CATALOG",
    "COMPANION_APPS_MAPPING",
    "build_stage1_router_prompt",
    "build_stage2_planner_prompt",
    "expand_selected_apps_with_safety_net",
    "TwoStageReActAgent",
]

