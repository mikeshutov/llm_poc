from __future__ import annotations

from time import perf_counter

from langsmith import traceable
from langchain_core.tools import BaseTool

from common.data import sanitize_for_json_storage
from common.logging import compact_trace_inputs, compact_trace_outputs, record_node_log
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.constants import NodeName
from request_orchestrator.models.evaluation_result import EVALUATION_STATUS_TERMINAL
from request_orchestrator.models.plan import PlanningResult
from request_orchestrator.models.plan import Plan, PlanStep
from request_orchestrator.shared.planner.prompts.planner_prompt import build_planner_prompt
from llm.chat_models import build_llm_for_stage, resolve_stage_model_name, resolve_stage_provider_name
from llm.conversation_model_config import OPENAI_PROVIDER
from llm.conversation_model_config import PLANNER_STAGE
from llm.usage import record_llm_call
from rendering.debug import PLAN_KIND
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
                if key not in {"hypothesis", "evaluation_fields"}
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
        metadata: dict[str, object] = {}
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
        return agent_state

    plan = Plan(steps=planning_result.steps)
    planner_state = agent_state.node_states.planner
    planner_state.plan = plan
    planner_state.needs_replan = False
    planner_state.plan_count += 1

    if not plan.steps:
        agent_state.node_states.evaluator.evaluation_status = EVALUATION_STATUS_TERMINAL

    compact_steps = [
        {
            "id": step.id,
            "tool": step.tool,
            "agent": step.agent,
            "hypothesis_preview": step.hypothesis[:500],
            "argument_keys": sorted(step.args),
            "evaluation_fields": list(getattr(step, "evaluation_fields", [])),
        }
        for step in plan.steps
    ]
    payload = {
        "agent_name": agent_state.agent_profile.name,
        "kind": PLAN_KIND,
        "status": "ready" if plan.steps else "completed",
        "data": sanitize_for_json_storage({
            "steps": compact_steps,
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

