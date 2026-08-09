from __future__ import annotations

from langsmith import traceable

from request_orchestrator.models.agent_state import AgentState, IterationState
from request_orchestrator.models import AgentResult, Plan
from request_orchestrator.shared.prompts.render_agent_prompt import render_agent_prompt
from request_orchestrator.shared.planner.prompts.planner_prompt import build_planner_prompt
from request_orchestrator.constants import PLANNER_PROMPT_STEP
from common.parsing import strip_code_fences
from conversation.models.conversation_model_config import PLANNER_STAGE
from conversation.repository.repo_factory import get_conversation_repo
from tool.repository.plan_repository import PlanRepository
from tool.tools import tools
from rendering.debug import PLAN_KIND

tool_list = "\n".join(
    f'- {tool.name}: {getattr(tool, "description", "")}'.strip()
    for tool in tools
)


def _invoke_planner(agent_state: AgentState, prompt_text: str) -> Plan:
    raw = strip_code_fences(agent_state.build_llm_for_stage(stage=PLANNER_STAGE).invoke(prompt_text).content)
    return Plan.model_validate_json(raw)


@traceable(name="Planner Node")
def run_planner(agent_state: AgentState) -> AgentState:
    it_state = IterationState.new()

    prompt = build_planner_prompt(state=agent_state)
    prompt_text = render_agent_prompt(prompt)
    had_prior_tool_results = any(bool(iteration.results) for iteration in agent_state.iteration_trace)

    try:
        plan = _invoke_planner(agent_state, prompt_text)
        if (
            agent_state.request_analysis.requires_tools
            and not had_prior_tool_results
            and (len(plan.steps) == 0 or plan.final_answer)
        ):
            retry_prompt = (
                f"{prompt_text}\n\n"
                "Additional requirement:\n"
                "- Request analysis already determined that tool use is required.\n"
                "- No tool results have been gathered yet.\n"
                "- Return at least one tool step.\n"
                "- Set final_answer to null on this pass.\n"
            )
            plan = _invoke_planner(agent_state, retry_prompt)
    except Exception as e:
        agent_state.goal_reached = True
        agent_state.result = AgentResult(
            answer=f"Planner produced invalid JSON plan: {e}"
        )
        return agent_state

    if agent_state.roundtrip_id:
        plan.db_id = PlanRepository().save_plan(agent_state.roundtrip_id, plan)

    it_state.plan = plan
    agent_state.add_iteration(it_state)

    if len(plan.steps) == 0 or plan.final_answer:
        agent_state.goal_reached = True

    agent_state.log_status(
        agent_name=agent_state.agent_profile.name,
        kind=PLAN_KIND,
        data={
            "step_plans": [step.plan for step in plan.steps],
            "final_answer": plan.final_answer,
        },
    )

    if agent_state.roundtrip_id:
        get_conversation_repo().create_roundtrip_prompt(
            agent_state.roundtrip_id,
            agent=agent_state.agent_profile.name,
            prompt_step=PLANNER_PROMPT_STEP,
            prompt=prompt_text,
        )

    return agent_state
