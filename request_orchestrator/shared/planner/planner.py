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
from llm.usage import record_llm_call, serialize_llm_call_record
from tool.repository.plan_repository import PlanRepository
from rendering.debug import PLAN_KIND

def _serialize_llm_call_for_log(llm_call) -> dict | None:
    if llm_call is None:
        return None
    return serialize_llm_call_record(llm_call)


def _invoke_planner(agent_state: AgentState, prompt_text: str) -> tuple[Plan, object | None]:
    llm = agent_state.build_llm_for_stage(stage=PLANNER_STAGE)
    response = llm.invoke(prompt_text)
    agent_scope = agent_state.resolve_agent_scope()
    llm_call = record_llm_call(
        raw_response=response,
        model_name=agent_state.resolve_model_for_stage(agent=agent_scope, stage=PLANNER_STAGE),
        conversation_id=agent_state.conversation_id,
        roundtrip_id=agent_state.roundtrip_id,
        user_id=agent_state.user_profile.user_id,
        agent=agent_scope,
        stage=PLANNER_STAGE,
        callsite="shared_planner.run_planner",
        input_object={
            "prompt": prompt_text,
        },
        output_object={
            "raw_content": response.content,
        },
    )
    raw = strip_code_fences(response.content)
    return Plan.model_validate_json(raw), llm_call


@traceable(name="Planner Node")
def run_planner(agent_state: AgentState) -> AgentState:
    it_state = IterationState.new()

    prompt = build_planner_prompt(state=agent_state)
    prompt_text = render_agent_prompt(prompt)
    had_prior_tool_results = any(bool(iteration.results) for iteration in agent_state.iteration_trace)
    llm_calls: list[dict[str, object]] = []

    try:
        plan, llm_call = _invoke_planner(agent_state, prompt_text)
        serialized = _serialize_llm_call_for_log(llm_call)
        if serialized is not None:
            llm_calls.append(serialized)
        if (
            agent_state.request_analysis.requires_tools
            and not had_prior_tool_results
            and len(plan.steps) == 0
        ):
            retry_prompt = (
                f"{prompt_text}\n\n"
                "Additional requirement:\n"
                "- Request analysis already determined that tool use is required.\n"
                "- No tool results have been gathered yet.\n"
                "- Return at least one tool step.\n"
            )
            plan, llm_call = _invoke_planner(agent_state, retry_prompt)
            serialized = _serialize_llm_call_for_log(llm_call)
            if serialized is not None:
                llm_calls.append(serialized)
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

    if len(plan.steps) == 0:
        agent_state.goal_reached = True

    agent_state.log_status(
        agent_name=agent_state.agent_profile.name,
        kind=PLAN_KIND,
        data={
            "step_plans": [step.plan for step in plan.steps],
            "llm_usage": llm_calls,
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

