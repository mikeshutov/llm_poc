from __future__ import annotations

import json
from typing import Any

from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.constants import EVALUATOR_PROMPT_KIND
from request_orchestrator.models.agent_prompt import AgentPrompt, PlanEvidenceStep
from request_orchestrator.shared.evaluator.prompts.evaluator_schema_prompt import EVALUATOR_SCHEMA


def _serialize_plan_with_evidence(plan_with_evidence: list[PlanEvidenceStep]) -> str:
    return json.dumps([step.model_dump() for step in plan_with_evidence], indent=2, ensure_ascii=True, default=str)


def _build_instruction(state: AgentState) -> str:
    return (
        "You are an evaluator between planning and synthesis. "
        "Check whether the current evidence is sufficient to satisfy the goal. "
        "If the evidence is sufficient, mark satisfied=true. "
        "If the evidence is not sufficient, identify the missing information and refine the goal for the next planning pass."
    )


def build_evaluator_prompt(*, state: AgentState, plan_with_evidence: list[PlanEvidenceStep]) -> str:
    parts: list[str] = [
        _build_instruction(state),
        "Goal:",
        (state.request_analysis.goal or state.task).strip(),
        "Latest User Prompt:",
        state.task.strip(),
        "User Profile (JSON):",
        json.dumps(state.user_profile.to_prompt_dict(), indent=2, ensure_ascii=True),
        "Executed Evidence (JSON):",
        _serialize_plan_with_evidence(plan_with_evidence),
        f"Response Schema: {EVALUATOR_SCHEMA}",
    ]
    return "\n\n".join(part for part in parts if part)
