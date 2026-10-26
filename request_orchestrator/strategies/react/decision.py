from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
from request_orchestrator.models.plan import Plan, PlanStep
from request_orchestrator.shared.evidence import (
    build_evidence_bundle_from_tool_results,
    build_evidence_steps_from_tool_results,
)
from request_orchestrator.shared.planner.prompts.planner_prompt import _compile_tools_rules_from_state
from request_orchestrator.shared.planner.prompts.planner_rules import build_planner_rules
from request_orchestrator.shared.planner.tool_wrapper import build_tools


@dataclass(frozen=True)
class DecisionResult:
    plan: Plan
    status: str


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
    available_fields_by_tool = {
        candidate.name: set(getattr(candidate, "available_result_fields", ())) for candidate in tools
    }
    steps: list[PlanStep] = []
    for index, call in enumerate(getattr(response, "tool_calls", None) or []):
        name = call.get("name", "")
        args = call.get("args", {}) if isinstance(call.get("args", {}), dict) else {}
        parameters = args.get("parameters", args)
        if not isinstance(parameters, dict):
            parameters = {}
        metadata: dict[str, object] = {}
        fields = args.get("evaluation_fields", [])
        if isinstance(fields, list):
            valid_fields = [
                field for field in fields
                if isinstance(field, str)
                and field.strip() in available_fields_by_tool.get(name, set())
            ]
            if valid_fields:
                metadata = {**metadata, "evaluation_fields": valid_fields}
        hypothesis = str(args.get("hypothesis", ""))
        steps.append(PlanStep(
            id=str(call.get("id") or f"E{index + 1}"),
            plan=hypothesis,
            hypothesis=hypothesis,
            metadata=metadata,
            tool=name,
            args=parameters,
        ))
    return Plan(steps=steps)


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
    status = "CONTINUE" if plan.steps else "DONE"
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
