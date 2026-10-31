from __future__ import annotations

from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.main_state import MainState
from agent_runtime.models.agent_prompt import (
    EVIDENCE_VIEW_EVALUATOR,
    AgentPrompt,
    EvidenceStep,
    PromptSectionKeys,
)
from agent_runtime.evaluation.prompts.evaluator_schema_prompt import EVALUATOR_SCHEMA


def _build_instruction(state: AgentState) -> str:
    return (
        "You are an evaluator between planning and synthesis. "
        "Decide whether the current evidence is sufficient to answer the user's stated question or goal,"
        "Do not expand the user's scope or require optional information that was not requested unless absolutely necessary. "
        "Return exactly one JSON object matching the provided schema, with the `status` field set to SATISFIED, PARTIAL_BUT_SUFFICIENT, RETRYABLE, or TERMINAL. "
        "When evidence is sufficient, identify only the evidence IDs that support the answer in relevant_evidence."
    )


def build_evaluator_prompt(state: AgentState | MainState, evidence: list[EvidenceStep]) -> AgentPrompt:
    prompt = AgentPrompt(
        instruction=_build_instruction(state),
        user_profile=state.execution_context.user_profile,
        task=(state.inputs.task if isinstance(state, AgentState) else state.task).strip(),
        evidence=evidence,
        evidence_view=EVIDENCE_VIEW_EVALUATOR,
        schema=EVALUATOR_SCHEMA,
    )
    prompt.include_section(PromptSectionKeys.USER_PROFILE)
    prompt.include_section(PromptSectionKeys.EVIDENCE)
    prompt.include_section(PromptSectionKeys.SCHEMA)
    prompt.include_section(PromptSectionKeys.TASK)
    return prompt
