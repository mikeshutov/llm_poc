from __future__ import annotations

from langsmith import traceable

from agent.agentstate.model import AgentState
from agent.models.agent_result import AgentResult
from agent.models.synthesized_result import SynthesisResult
from agent.prompt_constants import MAIN_AGENT_NAME, SYNTHESIS_PROMPT_STEP
from agent.prompts.agent_prompt import PlanEvidenceStep
from agent.synthesis.prompts.solver_prompt import build_solver_prompt
from common.parsing import strip_code_fences
from conversation.repository.repo_factory import get_conversation_repo


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
    raw = state.llm.invoke(prompt.to_string()).content
    raw = strip_code_fences(raw)

    try:
        synthesis_result = SynthesisResult.model_validate_json(raw)
    except Exception as e:
        state.result = AgentResult(
            answer=[f"Synthesis produced invalid JSON result: {e}\nRaw:\n{raw}"]
        )
        state.goal_reached = True
        return state

    state.result = AgentResult.from_state(
        state=state,
        answer=synthesis_result.result,
        follow_up=synthesis_result.follow_up,
        clarifying_question=synthesis_result.clarifying_question,
        roundtrip_summary=synthesis_result.roundtrip_summary,
        tool_summary=synthesis_result.tool_summary.model_dump(),
    )
    state.goal_reached = True

    if state.roundtrip_id:
        get_conversation_repo().create_roundtrip_prompt(
            state.roundtrip_id,
            agent=MAIN_AGENT_NAME,
            prompt_step=SYNTHESIS_PROMPT_STEP,
            prompt=prompt.to_string(),
        )

    return state
