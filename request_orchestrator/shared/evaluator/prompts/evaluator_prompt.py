from __future__ import annotations

from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.constants import EVALUATOR_PROMPT_KIND
from request_orchestrator.models.agent_prompt import AgentPrompt, EvidenceStep
from request_orchestrator.shared.evaluator.prompts.evaluator_schema_prompt import EVALUATOR_SCHEMA


def _build_instruction(state: AgentState) -> str:
    return (
        "You are an evaluator between planning and synthesis. "
        "Decide whether the current evidence is enough to answer well, whether another meaningful action remains, "
        "or whether the search should terminate without another planning pass. "
        "Return exactly one JSON object matching the provided schema, with the `status` field set to SATISFIED, RETRYABLE, or TERMINAL."
    )


def build_evaluator_prompt(state: AgentState, evidence: list[EvidenceStep]) -> AgentPrompt:
    prompt = AgentPrompt(
        prompt_kind=EVALUATOR_PROMPT_KIND,
        instruction=_build_instruction(state),
        user_profile=state.execution_context.user_profile,
        task=state.task.strip(),
        latest_user_prompt=state.task,
        evidence=evidence,
        schema=EVALUATOR_SCHEMA,
    )
    prompt.include_user_profile()
    prompt.include_evidence()
    prompt.include_latest_user_prompt()
    prompt.include_schema_as_response_label()
    prompt.include_task(heading="Goal:")
    return prompt
