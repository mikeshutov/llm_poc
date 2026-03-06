from __future__ import annotations

from dotenv import load_dotenv
from langsmith import traceable

from rendering.debug import build_plan_status_message, emit_status_message
from agent.agentstate.model import AgentState, IterationState
from agent.models import AgentResult, Plan
from agent.planner.prompts.planner_prompt import build_planner_prompt
from agent.tool.tools import tools
load_dotenv()

tool_list = "\n".join(
    f'- {tool.name}: {getattr(tool, "description", "")}'.strip()
    for tool in tools
)

# handle special cases wher ethe payload has additional fences
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

@traceable(name="Planner Node")
def run_planner(agent_state: AgentState) -> AgentState:
    it_state = IterationState.new()

    prompt = build_planner_prompt(
        tool_list=tool_list,
        task=agent_state.task,
        previous_calls=agent_state.iteration_trace,
    )
    raw = agent_state.llm.invoke(prompt).content
    raw = _strip_code_fences(raw)

    try:
        plan = Plan.model_validate_json(raw)
    except Exception as e:
        agent_state.goal_reached = True
        agent_state.result = AgentResult(
            answer=f"Planner produced invalid JSON plan: {e}\nRaw:\n{raw}"
        )
        return agent_state

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

    return agent_state
