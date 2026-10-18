from __future__ import annotations

import logging
from time import perf_counter

from langsmith import traceable
from langchain_core.tools import BaseTool

from common.data import sanitize_for_json_storage
from common.logging import create_conversation_event, log_roundtrip_prompt
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.evaluation_result import EVALUATION_STATUS_TERMINAL
from request_orchestrator.models.plan import PlanningResult
from request_orchestrator.models.plan import Plan, PlanStep
from request_orchestrator.shared.planner.prompts.planner_prompt import build_planner_prompt
from llm.chat_models import build_llm_for_stage, resolve_stage_model_name, resolve_stage_provider_name
from llm.conversation_model_config import OPENAI_PROVIDER
from request_orchestrator.constants import PLANNER_PROMPT_KIND
from llm.conversation_model_config import PLANNER_STAGE
from llm.usage import record_llm_call, serialize_llm_call_record
from rendering.debug import PLAN_KIND
from request_orchestrator.shared.planner.tool_wrapper import build_tools
from tool.repository.plan_repository import PlanRepository

logger = logging.getLogger(__name__)


def _serialize_llm_call_for_log(llm_call) -> dict | None:
    if llm_call is None:
        return None
    return serialize_llm_call_record(llm_call)


def invoke_planner(
    agent_state: AgentState,
    prompt_text: str,
    *,
    tools: list[BaseTool],
    prompt_input_object: dict[str, object],
) -> tuple[PlanningResult, object | None]:
    execution_context = agent_state.execution_context
    agent_scope = agent_state.resolve_agent_scope()
    llm = build_llm_for_stage(
        execution_context=execution_context,
        llm=agent_state.llm,
        agent=agent_scope,
        stage=PLANNER_STAGE,
        agent_profile=agent_state.agent_profile,
        reuse_llm_for_agent_scope=agent_scope,
    )
    model_name = resolve_stage_model_name(
        execution_context=execution_context,
        agent=agent_scope,
        stage=PLANNER_STAGE,
        agent_profile=agent_state.agent_profile,
    )
    bind_kwargs = (
        {"reasoning_effort": "none"}
        if resolve_stage_provider_name(
            execution_context=execution_context,
            agent=agent_scope,
            stage=PLANNER_STAGE,
            agent_profile=agent_state.agent_profile,
        ) == OPENAI_PROVIDER and model_name.startswith(("gpt-5", "gpt-6"))
        else {}
    )
    llm = llm.bind_tools(build_tools(tools), **bind_kwargs)
    started_at = perf_counter()
    response = llm.invoke(prompt_text)
    latency_ms = int((perf_counter() - started_at) * 1000)
    llm_call = record_llm_call(
        raw_response=response,
        model_name=resolve_stage_model_name(
            execution_context=execution_context,
            agent=agent_scope,
            stage=PLANNER_STAGE,
            agent_profile=agent_state.agent_profile,
        ),
        provider=resolve_stage_provider_name(
            execution_context=execution_context,
            agent=agent_scope,
            stage=PLANNER_STAGE,
            agent_profile=agent_state.agent_profile,
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
            "tool_calls": getattr(response, "tool_calls", None),
        },
    )
    tool_calls = getattr(response, "tool_calls", None) or []
    steps = []
    available_fields_by_tool = {
        candidate.name: set(getattr(candidate, "available_result_fields", ()))
        for candidate in tools
    }
    for index, call in enumerate(tool_calls):
        name = call.get("name", "")
        args = call.get("args", {}) if isinstance(call.get("args", {}), dict) else {}
        if "parameters" in args:
            tool_arguments = args.get("parameters", {})
            if not isinstance(tool_arguments, dict):
                tool_arguments = {}
        else:
            # Some providers may return the underlying tool arguments even
            # though the bound tool exposes the orchestration envelope. Keep
            # those arguments instead of silently executing the tool with {}.
            tool_arguments = {
                key: value
                for key, value in args.items()
                if key not in {"hypothesis", "metadata", "evaluation_fields"}
            }
        hypothesis = str(args.get("hypothesis", ""))
        evaluation_fields = [
            field.strip()
            for field in args.get("evaluation_fields", [])
            if (
                isinstance(field, str)
                and field.strip()
                and field.strip().lower() not in {"name", "title"}
                and field.strip() in available_fields_by_tool.get(name, set())
            )
        ] if isinstance(args.get("evaluation_fields", []), list) else []
        metadata = args.get("metadata", {}) if isinstance(args.get("metadata", {}), dict) else {}
        if evaluation_fields:
            metadata = {**metadata, "evaluation_fields": evaluation_fields}
        steps.append(PlanStep(
            id=str(call.get("id") or f"E{index + 1}"),
            plan=hypothesis,
            hypothesis=hypothesis,
            metadata=metadata,
            tool=name,
            args=tool_arguments,
        ))
    return PlanningResult(steps=steps), llm_call


@traceable(name="Planner Node")
def run_planner(agent_state: AgentState) -> AgentState:
    prompt, tools = build_planner_prompt(state=agent_state)
    prompt_text = prompt.build()
    prompt_input_object = prompt.to_log_input_object()
    llm_calls: list[dict[str, object]] = []
    planning_result: PlanningResult
    try:
        planning_result, llm_call = invoke_planner(
            agent_state,
            prompt_text,
            tools=tools,
            prompt_input_object=prompt_input_object,
        )
        serialized = _serialize_llm_call_for_log(llm_call)
        if serialized is not None:
            llm_calls.append(serialized)
    except Exception as e:
        logger.exception("Native planner invocation failed: %s", e)
        agent_state.node_states.evaluator.evaluation_status = EVALUATION_STATUS_TERMINAL
        return agent_state

    plan = Plan(steps=planning_result.steps)
    if agent_state.execution_context.roundtrip_id:
        plan.db_id = PlanRepository().save_interactive_plan(
            agent_state.execution_context.roundtrip_id,
            plan,
        )
    planner_state = agent_state.node_states.planner
    planner_state.plan = plan
    planner_state.needs_replan = False
    planner_state.plan_count += 1

    if not plan.steps:
        agent_state.node_states.evaluator.evaluation_status = EVALUATION_STATUS_TERMINAL

    payload = {
        "agent_name": agent_state.agent_profile.name,
        "kind": PLAN_KIND,
        "status": "ready" if plan.steps else "completed",
        "data": sanitize_for_json_storage({
            "steps": plan.steps,
            "llm_usage": llm_calls,
        }),
    }
    create_conversation_event(
        conversation_id=agent_state.execution_context.conversation_id,
        roundtrip_id=agent_state.execution_context.roundtrip_id,
        event_type=PLAN_KIND,
        source=agent_state.agent_profile.name,
        agent_name=agent_state.agent_profile.name,
        agent_version=agent_state.agent_profile.version,
        payload=payload,
    )

    if agent_state.execution_context.roundtrip_id:
        log_roundtrip_prompt(
            roundtrip_id=agent_state.execution_context.roundtrip_id,
            agent=agent_state.agent_profile.name,
            agent_version=agent_state.agent_profile.version,
            prompt_step=PLANNER_PROMPT_KIND,
            prompt=prompt_text,
        )

    return agent_state

