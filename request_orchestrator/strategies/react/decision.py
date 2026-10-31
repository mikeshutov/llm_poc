from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from enum import StrEnum
from time import perf_counter
from typing import Any

from langchain_core.tools import BaseTool

from common.data import sanitize_for_json_storage
from common.logging import record_node_log
from request_orchestrator.constants import NodeName
from llm.chat_models import build_llm_for_stage, resolve_stage_model_name, resolve_stage_provider_name
from llm.conversation_model_config import OPENAI_PROVIDER, PLANNER_STAGE
from llm.usage import record_llm_call
from rendering.debug import PLAN_KIND
from request_orchestrator.models.agent_prompt import AgentPrompt, PromptSectionKeys
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.plan import Plan
from request_orchestrator.shared.evidence import (
    build_evidence_bundle_from_tool_results,
    build_evidence_steps_from_tool_results,
)
from request_orchestrator.shared.planner.prompts.planner_prompt import _compile_tools_rules_from_state
from request_orchestrator.shared.planner.prompts.planner_rules import build_planner_rules
from request_orchestrator.shared.planner.tool_call_parser import parse_tool_calls_to_plan_steps
from request_orchestrator.shared.planner.tool_wrapper import build_tools


class DecisionStatus(StrEnum):
    CONTINUE = "CONTINUE"
    SATISFIED = "SATISFIED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class DecisionResult:
    plan: Plan
    status: DecisionStatus


def _normalize_status(value: Any) -> str:
    normalized = str(value or "").strip().upper()
    if normalized in {"DONE", "SATISFIED", "COMPLETE", "COMPLETED"}:
        return DecisionStatus.SATISFIED
    if normalized == DecisionStatus.BLOCKED:
        return DecisionStatus.BLOCKED
    return DecisionStatus.CONTINUE


def _parse_decision_status(response: Any, has_tool_calls: bool) -> DecisionStatus:
    content = getattr(response, "content", "")
    if isinstance(content, str) and content.strip():
        try:
            payload = json.loads(content)
            if isinstance(payload, dict):
                return _normalize_status(payload.get("status"))
        except (TypeError, ValueError):
            pass
    return DecisionStatus.CONTINUE if has_tool_calls else DecisionStatus.SATISFIED


DecisionPromptBuilder = Callable[[AgentState], tuple[AgentPrompt, list[BaseTool]]]


def build_decision_prompt(
    state: AgentState,
    *,
    instruction: str,
    rules: str,
) -> tuple[AgentPrompt, list[BaseTool]]:
    context = _compile_tools_rules_from_state(state)
    tool_results = state.gather_tool_results()
    evidence = build_evidence_steps_from_tool_results(
        tool_results,
        build_evidence_bundle_from_tool_results(tool_results).evidence_views_by_tool_call_id,
    )
    prompt = AgentPrompt(
        instruction=instruction,
        user_profile=state.execution_context.user_profile,
        conversation_context=state.execution_context.conversation_context,
        task=state.inputs.task,
        available_tools=context.compiled_tools,
        rules=f"{build_planner_rules(context.rules)}\n\n{rules}",
        evidence=evidence,
        missing_information=state.node_states.evaluator.missing_information,
    )
    for section in (
        PromptSectionKeys.USER_PROFILE,
        PromptSectionKeys.CONVERSATION_CONTEXT,
        PromptSectionKeys.AVAILABLE_TOOLS,
        PromptSectionKeys.RULES,
        PromptSectionKeys.EVIDENCE,
        PromptSectionKeys.MISSING_INFORMATION,
        PromptSectionKeys.TASK,
    ):
        prompt.include_section(section)
    return prompt, context.tools


def _parse_plan(response: Any, tools: list[BaseTool]) -> Plan:
    return Plan(steps=parse_tool_calls_to_plan_steps(
        getattr(response, "tool_calls", None) or [],
        tools,
    ))


def run_decision(
    state: AgentState,
    *,
    stage: str = PLANNER_STAGE,
    callsite: str,
    prompt_builder: DecisionPromptBuilder,
) -> DecisionResult:
    prompt, tools = prompt_builder(state)
    prompt_messages = prompt.build_messages()
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

    started_at = perf_counter()
    response = llm.bind_tools(build_tools(tools), **bind_kwargs).invoke(prompt_messages)
    latency_ms = int((perf_counter() - started_at) * 1000)
    plan = _parse_plan(response, tools)
    llm_call = record_llm_call(
        raw_response=response,
        model_name=model_name,
        provider=provider,
        conversation_id=state.execution_context.conversation_id,
        roundtrip_id=state.execution_context.roundtrip_id,
        user_id=state.execution_context.user_profile.user_id,
        agent=state.resolve_agent_scope(),
        stage=stage,
        node_name=NodeName.REASON,
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
        node_name=NodeName.REASON,
        agent_version=state.agent_profile.version,
        payload={
            "agent_name": state.agent_profile.name,
            "kind": PLAN_KIND,
            "status": status.lower(),
            "data": sanitize_for_json_storage({
                "steps": [
                    {
                        "id": step.id,
                        "tool": step.tool,
                        "agent": step.agent,
                        "hypothesis_preview": step.hypothesis[:500],
                        "argument_keys": sorted(step.args),
                    }
                    for step in plan.steps
                ],
            }),
        },
    )
    return DecisionResult(plan=plan, status=status)
