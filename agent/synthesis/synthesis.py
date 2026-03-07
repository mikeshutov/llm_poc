from __future__ import annotations

from dotenv import load_dotenv
from langsmith import traceable

from agent.agentstate.model import AgentState
from agent.models.agent_result import AgentResult
from agent.models.synthesized_result import SynthesisResult
from agent.synthesis.prompts.solver_prompt import build_solver_prompt
load_dotenv()


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]
            s = s.lstrip()
            if s.startswith("json"):
                s = s[4:].lstrip()
    return s.strip()


@traceable(name="Synthesis Node")
def run_synthesis(state: AgentState) -> AgentState:
    if not state.iteration_trace:
        state.result = AgentResult(answer=[])
        state.goal_reached = True
        return state

    lines: list[str] = []
    for iteration in state.iteration_trace:
        if iteration.plan is None:
            continue

        for step in iteration.plan.steps:
            evidence = iteration.results.get(step.id, "")
            # Keep it deterministic + readable for the solver
            lines.append(
                "STEP\n"
                f"id: {step.id}\n"
                f"plan: {step.plan}\n"
                f"tool: {step.tool}\n"
                f"args: {step.args}\n"
                f"evidence: {evidence}\n"
            )
    plan_block = "\n\n".join(lines).strip()

    # If you add final_answer/clarifying_question to Plan, you could short-circuit here
    # e.g. if last_plan.final_answer: state.result = last_plan.final_answer; ...

    prompt = build_solver_prompt(plan_block=plan_block, task=state.task)
    raw = state.llm.invoke(prompt).content
    raw = _strip_code_fences(raw)

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
    )
    state.goal_reached = True
    return state
