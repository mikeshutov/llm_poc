from __future__ import annotations

from langsmith import traceable

from rendering.debug import build_plan_status_message, emit_status_message
from agent.agentstate.model import AgentState, IterationState
from agent.models import AgentResult, Plan
from agent.planner.prompts.planner_prompt import build_planner_prompt
from agent.prompt_constants import MAIN_AGENT_NAME, PLANNER_PROMPT_STEP
from common.parsing import strip_code_fences
from conversation.repository.repo_factory import get_conversation_repo
from tool.repository.plan_repository import PlanRepository
from tool.tools import tools

tool_list = "\n".join(
    f'- {tool.name}: {getattr(tool, "description", "")}'.strip()
    for tool in tools
)

@traceable(name="Planner Node")
def run_planner(agent_state: AgentState) -> AgentState:
    it_state = IterationState.new()

    prompt = build_planner_prompt(state=agent_state)
    raw = strip_code_fences(agent_state.llm.invoke(prompt.to_string()).content)
    try:
        plan = Plan.model_validate_json(raw)
    except Exception as e:
        agent_state.goal_reached = True
        agent_state.result = AgentResult(
            answer=f"Planner produced invalid JSON plan: {e}\nRaw:\n{raw}"
        )
        return agent_state

    if agent_state.roundtrip_id:
        plan.db_id = PlanRepository().save_plan(agent_state.roundtrip_id, plan)

    it_state.plan = plan
    agent_state.add_iteration(it_state)

    # Naive approach. We are just going to check for no steps produced but this could be anything.
    if ((len(plan.steps) == 0) or plan.final_answer) :
        agent_state.goal_reached = True

    emit_status_message(
        build_plan_status_message(
            [step.plan for step in plan.steps],
            final_answer=plan.final_answer,
        )
    )

    if agent_state.roundtrip_id:
        get_conversation_repo().create_roundtrip_prompt(
            agent_state.roundtrip_id,
            agent=MAIN_AGENT_NAME,
            prompt_step=PLANNER_PROMPT_STEP,
            prompt=prompt.to_string(),
        )

    return agent_state
