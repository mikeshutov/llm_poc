from __future__ import annotations

from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.constants import EVALUATOR_PROMPT_KIND
from request_orchestrator.models.agent_prompt import AgentPrompt, PlanEvidenceStep
from request_orchestrator.shared.evaluator.prompts.evaluator_schema_prompt import EVALUATOR_SCHEMA


def _build_instruction(state: AgentState) -> str:
    return (
        "You are an evaluator between planning and synthesis. "
        "Decide whether the current evidence is enough to answer well, whether another meaningful action remains, "
        "or whether the search should terminate without another planning pass. "
        "Return SATISFIED, RETRYABLE, or TERMINAL accordingly."
    )


def build_evaluator_prompt(state: AgentState, plan_with_evidence: list[PlanEvidenceStep]) -> AgentPrompt:
    return AgentPrompt(
        prompt_kind=EVALUATOR_PROMPT_KIND,
        instruction=_build_instruction(state),
        user_profile=state.user_profile,
        task=(state.request_analysis.goal or state.task).strip(),
        latest_user_prompt=state.task,
        plan_with_evidence=plan_with_evidence,
        schema=EVALUATOR_SCHEMA,
    )
