from __future__ import annotations

from langsmith import traceable

from common.parsing import strip_code_fences
from conversation.models.conversation_model_config import MAIN_AGENT_MODEL_SCOPE, SYNTHESIS_STAGE
from conversation.repository.repo_factory import get_conversation_repo
from request_orchestrator.constants import SYNTHESIS_PROMPT_STEP
from request_orchestrator.models.agent_prompt import PlanEvidenceStep
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.synthesized_result import SynthesisResult
from request_orchestrator.shared.prompts.render_agent_prompt import render_agent_prompt
from request_orchestrator.shared.synthesis.prompts.solver_prompt import build_solver_prompt


@traceable(name="Synthesis Node")
def run_synthesis(state: AgentState) -> AgentState:
    if not state.iteration_trace and not state.goal_reached:
        state.result = AgentResult(answer=[])
        state.goal_reached = True
        return state

    plan_with_evidence: list[PlanEvidenceStep] = []
    for iteration in state.iteration_trace:
        if iteration.plan is None:
            continue

        for step in iteration.plan.steps:
            plan_with_evidence.append(
                PlanEvidenceStep(
                    step_id=step.id,
                    plan=step.plan,
                    tool=step.tool,
                    args=step.args,
                    evidence=iteration.results.get(step.id, ""),
                )
            )

    prompt = build_solver_prompt(plan_with_evidence=plan_with_evidence, state=state)
    prompt_text = render_agent_prompt(prompt)
    raw = state.build_llm_for_stage(
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=SYNTHESIS_STAGE,
    ).invoke(prompt_text).content
    raw = strip_code_fences(raw)

    try:
        synthesis_result = SynthesisResult.model_validate_json(raw)
    except Exception as e:
        state.result = AgentResult(
            answer=[f"Synthesis produced invalid JSON result: {e}\nRaw:\n{raw}"]
        )
        state.goal_reached = True
        return state

    had_tool_results = any(bool(iteration.results) for iteration in state.iteration_trace)
    tool_summary = synthesis_result.tool_summary.model_dump() if had_tool_results else {}

    state.result = AgentResult.from_state(
        state=state,
        answer=synthesis_result.result,
        follow_up=synthesis_result.follow_up,
        clarifying_question=synthesis_result.clarifying_question,
        roundtrip_summary=synthesis_result.roundtrip_summary,
        tool_summary=tool_summary,
    )
    state.goal_reached = True

    if state.roundtrip_id:
        get_conversation_repo().create_roundtrip_prompt(
            state.roundtrip_id,
            agent=state.agent_profile.name,
            prompt_step=SYNTHESIS_PROMPT_STEP,
            prompt=prompt_text,
        )

    return state
