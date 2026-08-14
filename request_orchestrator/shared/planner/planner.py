from __future__ import annotations

from time import perf_counter

from langsmith import traceable

from request_orchestrator.models.agent_state import AgentState, IterationState
from request_orchestrator.models import AgentResult, Plan, PlanningResult
from common.logging import log_roundtrip_prompt
from request_orchestrator.shared.planner.prompts.planner_prompt import build_planner_prompt
from request_orchestrator.constants import PLANNER_PROMPT_STEP
from common.data import repair_common_json_issues, strip_code_fences
from conversation.models.conversation_model_config import PLANNER_STAGE
from llm.usage import record_llm_call, serialize_llm_call_record
from tool.repository.plan_repository import PlanRepository
from rendering.debug import PLAN_KIND

REQUIRED_CAPABILITY_UNAVAILABLE_REASON = "required capability unavailable."


def _serialize_llm_call_for_log(llm_call) -> dict | None:
    if llm_call is None:
        return None
    return serialize_llm_call_record(llm_call)


def _invoke_planner(
    agent_state: AgentState,
    prompt_text: str,
    *,
    prompt_input_object: dict[str, object],
) -> tuple[PlanningResult, object | None]:
    llm = agent_state.build_llm_for_stage(stage=PLANNER_STAGE)
    started_at = perf_counter()
    response = llm.invoke(prompt_text)
    latency_ms = int((perf_counter() - started_at) * 1000)
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
        latency_ms=latency_ms,
        owner_agent_name=agent_state.agent_profile.name,
        input_object=prompt_input_object,
        output_object={
            "raw_content": response.content,
        },
    )
    raw = repair_common_json_issues(strip_code_fences(response.content))
    return PlanningResult.model_validate_json(raw), llm_call


@traceable(name="Planner Node")
def run_planner(agent_state: AgentState) -> AgentState:
    it_state = IterationState.new()

    prompt = build_planner_prompt(state=agent_state)
    prompt_text = prompt.prompt_text()
    prompt_input_object = prompt.to_log_input_object()
    llm_calls: list[dict[str, object]] = []
    planning_result: PlanningResult

    try:
        planning_result, llm_call = _invoke_planner(
            agent_state,
            prompt_text,
            prompt_input_object=prompt_input_object,
        )
        serialized = _serialize_llm_call_for_log(llm_call)
        if serialized is not None:
            llm_calls.append(serialized)
    except Exception as e:
        agent_state.goal_reached = True
        agent_state.result = AgentResult(
            answer=[f"Planner produced invalid JSON plan: {e}"]
        )
        return agent_state

    if len(planning_result.steps) == 0 and planning_result.status == "blocked" and not planning_result.reason:
        planning_result = PlanningResult(
            steps=[],
            status="blocked",
            reason=REQUIRED_CAPABILITY_UNAVAILABLE_REASON,
            needs_replan=False,
        )

    plan = Plan(steps=planning_result.steps)

    if agent_state.roundtrip_id:
        plan.db_id = PlanRepository().save_plan(agent_state.roundtrip_id, plan)

    it_state.plan = plan
    it_state.needs_replan = planning_result.needs_replan
    agent_state.add_iteration(it_state)

    if len(plan.steps) == 0:
        agent_state.goal_reached = True

    agent_state.log_status(
        agent_name=agent_state.agent_profile.name,
        kind=PLAN_KIND,
        status=planning_result.status,
        data={
            "step_plans": [step.plan for step in plan.steps],
            "planner_status": planning_result.status,
            "planner_reason": planning_result.reason,
            "needs_replan": planning_result.needs_replan,
            "llm_usage": llm_calls,
        },
    )

    if agent_state.roundtrip_id:
        log_roundtrip_prompt(
            roundtrip_id=agent_state.roundtrip_id,
            agent=agent_state.agent_profile.name,
            prompt_step=PLANNER_PROMPT_STEP,
            prompt=prompt_text,
        )

    return agent_state

