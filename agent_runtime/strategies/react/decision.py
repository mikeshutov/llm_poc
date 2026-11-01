from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json
from time import perf_counter
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from common.data import sanitize_for_json_storage
from common.logging import record_node_log
from agent_runtime.constants import NodeName
from llm.chat_models import build_llm_for_stage, resolve_stage_model_name, resolve_stage_provider_name
from llm.conversation_model_config import OPENAI_PROVIDER, PLANNER_STAGE
from llm.usage import record_llm_call
from rendering.debug import PLAN_KIND
from agent_runtime.models.agent_prompt import AgentPrompt, PromptSectionKeys
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.agent_profile import (
    AgentExecutionStrategy,
    PROFILE_MANAGEMENT_AGENT_NAME,
)
from agent_runtime.models.decision import DecisionStatus
from agent_runtime.models.plan import Plan
from retrieval.evidence_store import (
    build_evidence_bundle_from_tool_results,
    build_evidence_steps_from_tool_results,
)
from agent_runtime.planning.prompts.planner_prompt import _compile_tools_rules_from_state
from agent_runtime.planning.prompts.planner_rules import build_planner_rules
from agent_runtime.planning.tool_call_parser import parse_tool_calls_to_plan_steps
from agent_runtime.planning.tool_wrapper import build_tools
from capabilities.discovery.models import AvailableAgent, AvailableTool


@dataclass(frozen=True)
class DecisionResult:
    plan: Plan
    status: DecisionStatus
    raw_content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def _parse_decision_status(response: Any, has_tool_calls: bool) -> DecisionStatus:
    content = getattr(response, "content", "")
    if isinstance(content, str) and content.strip():
        try:
            payload = json.loads(content)
            if isinstance(payload, dict):
                if "status" in payload:
                    try:
                        return DecisionStatus(payload["status"])
                    except (TypeError, ValueError):
                        return DecisionStatus.CONTINUE
                return DecisionStatus.CONTINUE if has_tool_calls else DecisionStatus.SATISFIED
        except (TypeError, ValueError):
            pass
    return DecisionStatus.CONTINUE if has_tool_calls else DecisionStatus.SATISFIED


DecisionPromptBuilder = Callable[[AgentState], tuple[AgentPrompt, list[BaseTool]]]


def _build_agent_delegation_tools(agent_states) -> list[BaseTool]:
    delegation_tools: list[BaseTool] = []
    for agent_state in agent_states:
        profile = agent_state.agent_profile
        args_schema = profile.delegation_args_schema

        def delegate(*, agent_name=profile.name, **kwargs):
            payload = dict(kwargs)
            evidence_ids = payload.get("evidence_ids")
            if isinstance(evidence_ids, list):
                payload["evidence_ids"] = list(dict.fromkeys(
                    evidence_id.strip()
                    for evidence_id in evidence_ids
                    if isinstance(evidence_id, str) and evidence_id.strip()
                ))
            return {"agent": agent_name, **payload}

        delegation_tools.append(
            StructuredTool.from_function(
                func=delegate,
                name=profile.name,
                description=profile.description,
                args_schema=args_schema,
                metadata={"agent_delegation": True},
            )
        )
    return delegation_tools


def build_decision_prompt(
    state: AgentState,
    *,
    instruction: str,
    rules: str,
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
    available_capabilities: list[AvailableTool] = [
        AvailableTool.from_tool(tool)
        for tool in context.tools
    ]
    delegation_tools: list[BaseTool] = []
    if state.available_agent_states is not None:
        capabilities = state.inputs.discovered_capabilities
        allowed_capability_names = {tool.name for tool in context.tools}
        discovered_agent_names = {
            agent.agent
            for agent in ([] if capabilities is None else capabilities.agents)
        }
        delegated_states = [
            delegated
            for delegated in state.available_agent_states.values()
            if delegated.agent_profile.delegatable
            and delegated.agent_profile.name != PROFILE_MANAGEMENT_AGENT_NAME
            and (
                capabilities is None
                or delegated.agent_profile.name in discovered_agent_names
            )
        ]
        delegation_tools = _build_agent_delegation_tools(delegated_states)
        available_capabilities = [
            *[
                tool
                for tool in (
                    []
                    if capabilities is None
                    else capabilities.as_available_tools(include_agents=False)
                )
                if tool.name in allowed_capability_names
            ],
            *[
                    AvailableAgent(
                        agent=delegated.agent_profile.name,
                        description=delegated.agent_profile.description,
                    ).as_available_tool()
                for delegated in delegated_states
            ],
            *[
                AvailableTool.from_tool(tool)
                for tool in state.agent_profile.static_tools
            ],
        ]
    prompt = AgentPrompt(
        instruction=instruction,
        user_profile=state.execution_context.user_profile,
        conversation_context=state.execution_context.conversation_context,
        task=state.inputs.task,
        evidence_ids=state.inputs.evidence_ids,
        available_tools=available_capabilities,
        rules=f"{build_planner_rules(context.rules)}\n\n{rules}",
        evidence=evidence,
        missing_information=state.node_states.evaluator.missing_information,
    )
    sections = (
        PromptSectionKeys.USER_PROFILE,
        PromptSectionKeys.CONVERSATION_CONTEXT,
        PromptSectionKeys.AVAILABLE_TOOLS,
        PromptSectionKeys.RULES,
        PromptSectionKeys.MISSING_INFORMATION,
        PromptSectionKeys.TASK,
        PromptSectionKeys.EVIDENCE_IDS,
    )
    if include_evidence:
        sections = (*sections[:4], PromptSectionKeys.EVIDENCE, *sections[4:])
    for section in sections:
        prompt.include_section(section)
    return prompt, [*context.tools, *delegation_tools]


def _parse_plan(response: Any, tools: list[BaseTool], agent_names: set[str] | None = None) -> Plan:
    return Plan(steps=parse_tool_calls_to_plan_steps(
        getattr(response, "tool_calls", None) or [],
        tools,
        agent_names=agent_names,
    ))


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
    agent_names = {
        delegated.agent_profile.name
        for delegated in (state.available_agent_states or {}).values()
        if delegated.agent_profile.delegatable
        and delegated.agent_profile.name != PROFILE_MANAGEMENT_AGENT_NAME
        and (
            state.inputs.discovered_capabilities is None
            or delegated.agent_profile.name in {
                agent.agent
                for agent in state.inputs.discovered_capabilities.agents
            }
        )
    }
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
    plan = _parse_plan(response, tools, agent_names=agent_names)
    llm_call = record_llm_call(
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
            "tool_calls": getattr(response, "tool_calls", None),
        },
    )
    status = _parse_decision_status(
        response,
        bool(getattr(response, "tool_calls", None)),
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
            "status": status.lower(),
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
        status=status,
        raw_content=getattr(response, "content", "") or "",
        tool_calls=list(getattr(response, "tool_calls", None) or []),
    )
