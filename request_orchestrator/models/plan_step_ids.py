from __future__ import annotations


def format_plan_step_id(iteration_number: int, step_id: str) -> str:
    return f"P{max(1, iteration_number)}{step_id}"


def namespace_step_id(agent_name: str, step_id: str) -> str:
    normalized_agent_name = agent_name.strip()
    normalized_step_id = step_id.strip()
    if not normalized_agent_name or not normalized_step_id:
        return normalized_step_id
    if normalized_step_id.startswith(f"{normalized_agent_name}:"):
        return normalized_step_id
    return f"{normalized_agent_name}:{normalized_step_id}"
