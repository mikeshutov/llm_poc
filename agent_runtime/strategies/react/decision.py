from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from langchain_core.tools import BaseTool

from common.data import sanitize_for_json_storage
from common.logging import record_node_log
from agent_runtime.constants import NodeName
from llm.chat_models import build_llm_for_stage, resolve_stage_model_name, resolve_stage_provider_name
from llm.conversation_model_config import OPENAI_PROVIDER, PLANNER_STAGE
from llm.usage import record_llm_call
from rendering.debug import PLAN_KIND
from agent_runtime.models.agent_prompt import AgentPrompt, PromptSectionKeys
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.agent_result import AnswerStatus, ExecutionStatus
from agent_runtime.models.agent_profile import AgentExecutionStrategy
from agent_runtime.models.decision import DecisionResponse, DecisionStatus
from agent_runtime.models.plan import Plan
from retrieval.evidence_store import (
    build_evidence_bundle_from_tool_results,
    build_evidence_steps_from_tool_results,
)
from agent_runtime.planning.prompts.planner_prompt import _compile_tools_rules_from_state
from agent_runtime.planning.prompts.planner_rules import build_planner_rules
from agent_runtime.planning.tool_call_parser import parse_tool_calls_to_plan_steps
from agent_runtime.planning.tool_wrapper import build_tools
from capabilities.discovery.models import AvailableTool
from capabilities.registry.global_registry import available_agent_tools


@dataclass(frozen=True)
class DecisionResult:
    plan: Plan
    status: DecisionStatus
    message: str = ""
    raw_content: str = ""


DecisionPromptBuilder = Callable[[AgentState], tuple[AgentPrompt, list[BaseTool]]]


def record_decision_status(state: AgentState, status: DecisionStatus) -> None:
    state.node_states.planner.decision_status = status
    if state.node_states.planner.plan and state.node_states.planner.plan.steps:
        return
    if status == DecisionStatus.SATISFIED:
        state.result = state.result.copy(answer_status=AnswerStatus.SUCCESS)
    elif status == DecisionStatus.BLOCKED:
        state.result = state.result.copy(
            answer_status=AnswerStatus.FAILED,
            execution_status=ExecutionStatus.FAILED,
        )


def build_decision_prompt(
    state: AgentState,
    *,
    instruction: str,
    rules: str,
    schema: str = "",
    include_evidence: bool = True,
) -> tuple[AgentPrompt, list[BaseTool]]:
    context = _compile_tools_rules_from_state(state)
    evidence = []
    if include_evidence:
        tool_results = state.gather_tool_results()
        evidence = build_evidence_steps_from_tool_results(
            tool_results,
            build_evidence_bundle_from_tool_results(tool_results).evidence_views_by_tool_call_id,
        )
    delegation_tools = available_agent_tools(state)
    available_capabilities = [
        *(AvailableTool.from_tool(tool) for tool in context.tools),
        *(AvailableTool.from_tool(tool, category="agents") for tool in delegation_tools),
    ]
    prompt = AgentPrompt(
        instruction=instruction,
        schema=schema,
        user_profile=(state.execution_context.user_profile if state.agent_profile.include_user_profile else None),
        conversation_context=state.execution_context.conversation_context,
        task=state.inputs.task,
        evidence_ids=state.inputs.evidence_ids,
        available_tools=available_capabilities,
        rules=f"{build_planner_rules(context.rules)}\n\n{rules}",
        evidence=evidence,
        missing_information=state.node_states.evaluator.missing_information,
        working_memory=state.render_working_memory(),
    )
    sections = (
        *((PromptSectionKeys.USER_PROFILE,) if state.agent_profile.include_user_profile else ()),
        PromptSectionKeys.CONVERSATION_CONTEXT,
        PromptSectionKeys.AVAILABLE_TOOLS,
        PromptSectionKeys.RULES,
        PromptSectionKeys.SCHEMA,
        PromptSectionKeys.MISSING_INFORMATION,
        PromptSectionKeys.TASK,
        PromptSectionKeys.EVIDENCE_IDS,
        PromptSectionKeys.WORKING_MEMORY,
    )
    if include_evidence:
        sections = (*sections[:4], PromptSectionKeys.EVIDENCE, *sections[4:])
    for section in sections:
        prompt.include_section(section)
    return prompt, [*context.tools, *delegation_tools]


def _parse_response(
    response: Any,
    tools: list[BaseTool],
) -> tuple[Plan, DecisionResponse]:
    plan = Plan(steps=parse_tool_calls_to_plan_steps(
        getattr(response, "tool_calls", None) or [],
        tools,
    ))
    content = getattr(response, "content", "")
    if isinstance(content, str) and content.strip():
        try:
            return plan, DecisionResponse.model_validate_json(content)
        except (TypeError, ValueError):
            pass
    return plan, DecisionResponse(
        status=DecisionStatus.CONTINUE if plan.steps else DecisionStatus.SATISFIED,
        message="",
    )


def run_decision(
    state: AgentState,
    *,
    stage: str = PLANNER_STAGE,
    callsite: str,
    prompt_builder: DecisionPromptBuilder,
    message_history: list[Any] | None = None,
    tool_choice: str | None = None,
) -> DecisionResult:
    prompt, tools = prompt_builder(state)
    prompt_messages = prompt.build_messages()
    if message_history:
        prompt_messages = [*prompt_messages, *message_history]
    llm = build_llm_for_stage(
        execution_context=state.execution_context,
        llm=state.llm,
        agent=state.resolve_agent_scope(),
        stage=stage,
        agent_profile=state.agent_profile,
        reuse_llm_for_agent_scope=state.resolve_agent_scope() if stage == PLANNER_STAGE else None,
    )
    model_name = resolve_stage_model_name(
        execution_context=state.execution_context,
        agent=state.resolve_agent_scope(),
        stage=stage,
        agent_profile=state.agent_profile,
    )
    provider = resolve_stage_provider_name(
        execution_context=state.execution_context,
        agent=state.resolve_agent_scope(),
        stage=stage,
        agent_profile=state.agent_profile,
    )
    bind_kwargs: dict[str, Any] = {}
    if provider == OPENAI_PROVIDER and model_name.startswith(("gpt-5", "gpt-6")):
        bind_kwargs["reasoning_effort"] = "none"
    if tool_choice is not None:
        bind_kwargs["tool_choice"] = tool_choice

    started_at = perf_counter()
    bound_tools = (
        build_tools(tools, include_evaluation_wrapper=False)
        if state.agent_profile.execution_strategy == AgentExecutionStrategy.REACT
        else build_tools(tools)
    )
    response = llm.bind_tools(bound_tools, **bind_kwargs).invoke(prompt_messages)
    latency_ms = int((perf_counter() - started_at) * 1000)
    plan, decision_response = _parse_response(response, tools)
    record_llm_call(
        raw_response=response,
        model_name=model_name,
        provider=provider,
        conversation_id=state.execution_context.conversation_id,
        roundtrip_id=state.execution_context.roundtrip_id,
        user_id=state.execution_context.user_profile.user_id,
        agent=state.resolve_agent_scope(),
        stage=stage,
        node_name=NodeName.REASONER,
        callsite=callsite,
        latency_ms=latency_ms,
        owner_agent_name=state.agent_profile.name,
        input_object={"messages": prompt_messages},
        output_object={
            "raw_content": response.content,
            "decision_status": decision_response.status,
            "decision_message": decision_response.message,
        },
    )
    record_node_log(
        conversation_id=state.execution_context.conversation_id,
        roundtrip_id=state.execution_context.roundtrip_id,
        event_type=PLAN_KIND,
        source=state.agent_profile.name,
        agent_name=state.agent_profile.name,
        node_name=NodeName.REASONER,
        agent_version=state.agent_profile.version,
        payload={
            "agent_name": state.agent_profile.name,
            "kind": PLAN_KIND,
            "status": decision_response.status.lower(),
            "message": decision_response.message,
            "data": sanitize_for_json_storage({
                "steps": [
                    step.model_dump(mode="json")
                    for step in plan.steps
                ],
            }),
        },
    )
    return DecisionResult(
        plan=plan,
        status=decision_response.status,
        message=decision_response.message,
        raw_content=getattr(response, "content", "") or "",
    )
