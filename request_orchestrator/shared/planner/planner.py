from __future__ import annotations

from time import perf_counter

from langsmith import traceable
from langchain_core.tools import BaseTool

from common.data import sanitize_for_json_storage
from common.logging import compact_trace_inputs, compact_trace_outputs, record_node_log
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.agent_result import AnswerStatus, ExecutionStatus
from request_orchestrator.constants import NodeName
from request_orchestrator.models.evaluation_result import EVALUATION_STATUS_TERMINAL
from request_orchestrator.models.plan import PlanningResult
from request_orchestrator.models.plan import Plan
from request_orchestrator.shared.planner.prompts.planner_prompt import build_planner_prompt
from llm.chat_models import build_llm_for_stage, resolve_stage_model_name, resolve_stage_provider_name
from llm.conversation_model_config import OPENAI_PROVIDER
from llm.conversation_model_config import PLANNER_STAGE
from llm.usage import record_llm_call
from rendering.debug import PLAN_KIND
from request_orchestrator.shared.planner.tool_call_parser import parse_tool_calls_to_plan_steps
from request_orchestrator.shared.planner.tool_wrapper import build_tools

def invoke_planner(
    agent_state: AgentState,
    prompt_messages: list[dict[str, str]],
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
    response = llm.invoke(prompt_messages)
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
        node_name=NodeName.PLANNER,
        callsite="shared_planner.run_planner",
        latency_ms=latency_ms,
        owner_agent_name=agent_state.agent_profile.name,
        input_object=prompt_input_object,
        output_object={
            "raw_content": response.content,
            "tool_calls": getattr(response, "tool_calls", None),
        },
    )
    steps = parse_tool_calls_to_plan_steps(
        getattr(response, "tool_calls", None) or [],
        tools,
    )
    return PlanningResult(steps=steps), llm_call


@traceable(name="Planner Node", process_inputs=compact_trace_inputs, process_outputs=compact_trace_outputs)
def run_planner(agent_state: AgentState) -> AgentState:
    prompt, tools = build_planner_prompt(state=agent_state)
    prompt_messages = prompt.build_messages()
    prompt_input_object = {"messages": prompt_messages}
    planning_result: PlanningResult
    try:
        planning_result, llm_call = invoke_planner(
            agent_state,
            prompt_messages,
            tools=tools,
            prompt_input_object=prompt_input_object,
        )
    except Exception as e:
        record_node_log(
            roundtrip_id=agent_state.execution_context.roundtrip_id,
            node_name=NodeName.PLANNER,
            stage="invoke",
            event_type="planner",
            level="error",
            message="Planner invocation failed",
            payload={"error": str(e)},
        )
        agent_state.node_states.evaluator.evaluation_status = EVALUATION_STATUS_TERMINAL
        agent_state.result = agent_state.result.copy(
            answer_status=AnswerStatus.FAILED,
            execution_status=ExecutionStatus.FAILED,
        )
        return agent_state

    plan = Plan(steps=planning_result.steps)
    planner_state = agent_state.node_states.planner
    planner_state.plan = plan
    planner_state.needs_replan = False
    planner_state.plan_count += 1

    if not plan.steps:
        agent_state.node_states.evaluator.evaluation_status = EVALUATION_STATUS_TERMINAL

    plan_steps = [step.model_dump(mode="json") for step in plan.steps]
    payload = {
        "agent_name": agent_state.agent_profile.name,
        "kind": PLAN_KIND,
        "status": "ready" if plan.steps else "completed",
        "data": sanitize_for_json_storage({
            "steps": plan_steps,
        }),
    }
    record_node_log(
        conversation_id=agent_state.execution_context.conversation_id,
        roundtrip_id=agent_state.execution_context.roundtrip_id,
        event_type=PLAN_KIND,
        source=agent_state.agent_profile.name,
        agent_name=agent_state.agent_profile.name,
        node_name=NodeName.PLANNER,
        agent_version=agent_state.agent_profile.version,
        payload=payload,
    )

    return agent_state

