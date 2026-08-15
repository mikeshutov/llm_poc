from __future__ import annotations

from time import perf_counter

from langsmith import traceable

from common.data import sanitize_for_json_storage
from common.logging import create_conversation_event, log_roundtrip_prompt
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models import AgentResult, Plan, PlanningResult
from request_orchestrator.shared.planner.prompts.planner_prompt import build_planner_prompt
from request_orchestrator.shared.llm_factory import build_llm_for_stage, resolve_stage_model_name
from request_orchestrator.constants import PLANNER_PROMPT_KIND
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
    execution_context = agent_state.execution_context
    agent_scope = agent_state.resolve_agent_scope()
    llm = build_llm_for_stage(
        execution_context=execution_context,
        llm=agent_state.llm,
        agent=agent_scope,
        stage=PLANNER_STAGE,
        reuse_llm_for_agent_scope=agent_scope,
    )
    started_at = perf_counter()
    response = llm.invoke(prompt_text)
    latency_ms = int((perf_counter() - started_at) * 1000)
    llm_call = record_llm_call(
        raw_response=response,
        model_name=resolve_stage_model_name(
            execution_context=execution_context,
            agent=agent_scope,
            stage=PLANNER_STAGE,
        ),
        conversation_id=execution_context.conversation_id,
        roundtrip_id=execution_context.roundtrip_id,
        user_id=execution_context.user_profile.user_id,
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
        agent_state.node_states.evaluator.goal_reached = True
        return agent_state

    if len(planning_result.steps) == 0 and planning_result.status == "blocked" and not planning_result.reason:
        planning_result = PlanningResult(
            steps=[],
            status="blocked",
            reason=REQUIRED_CAPABILITY_UNAVAILABLE_REASON,
            needs_replan=False,
        )

    plan = Plan(steps=planning_result.steps)

    if agent_state.execution_context.roundtrip_id:
        plan.db_id = PlanRepository().save_plan(agent_state.execution_context.roundtrip_id, plan)

    agent_state.begin_plan(
        plan,
        needs_replan=planning_result.needs_replan,
    )

    if len(plan.steps) == 0:
        agent_state.node_states.evaluator.goal_reached = True

    payload = {
        "agent_name": agent_state.agent_profile.name,
        "kind": PLAN_KIND,
        "status": planning_result.status,
        "data": sanitize_for_json_storage({
            "step_plans": [step.plan for step in plan.steps],
            "planner_status": planning_result.status,
            "planner_reason": planning_result.reason,
            "needs_replan": planning_result.needs_replan,
            "llm_usage": llm_calls,
        }),
    }
    create_conversation_event(
        conversation_id=agent_state.execution_context.conversation_id,
        roundtrip_id=agent_state.execution_context.roundtrip_id,
        event_type=PLAN_KIND,
        source=agent_state.agent_profile.name,
        agent_name=agent_state.agent_profile.name,
        payload=payload,
    )

    if agent_state.execution_context.roundtrip_id:
        log_roundtrip_prompt(
            roundtrip_id=agent_state.execution_context.roundtrip_id,
            agent=agent_state.agent_profile.name,
            prompt_step=PLANNER_PROMPT_KIND,
            prompt=prompt_text,
        )

    return agent_state

