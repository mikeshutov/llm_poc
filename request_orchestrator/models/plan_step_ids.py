from __future__ import annotations


def format_plan_step_id(iteration_number: int, step_id: str) -> str:
    return f"P{max(1, iteration_number)}{step_id}"
