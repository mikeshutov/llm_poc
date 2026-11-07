from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import UUID

from langsmith import traceable
from pydantic import ValidationError

from common.data import bounded_json_value
from common.signatures import build_signature
from common.logging import compact_trace_inputs, compact_trace_outputs, record_node_log
from agent_runtime.constants import NodeName
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.agent_profile import AgentExecutionStrategy, AgentName, AgentProfile
from agent_runtime.models.evaluation_result import (
    EVALUATION_STATUS_PARTIAL_BUT_SUFFICIENT,
    EVALUATION_STATUS_RETRYABLE,
    EVALUATION_STATUS_SATISFIED,
    EVALUATION_STATUS_TERMINAL,
)
from retrieval.evidence import ToolOutcome, ToolOutcomeStatus, ToolResult
from agent_runtime.models.plan import Plan
from agent_runtime.models.plan import PlanStep
from agent_runtime.context import bind_agent_context, bind_runtime_context
from capabilities.registry.global_registry import call_tool
from capabilities.registry.repository.message_repository import MessageToolRepository
from agent_runtime.execution.message_history import append_execution_messages


@dataclass(frozen=True)
class StepExecutionResult:
    step: PlanStep
    args: dict[str, Any]
    output: Any
    error_text: str = ""
    rejection_reason: str = ""
    latency_ms: int = 0


def _substitute_refs(obj, results: dict):
    if isinstance(obj, str):
        if obj.startswith("#E"):
            raw_step_id = obj[1:]
            referenced = results.get(raw_step_id, obj)
            if isinstance(referenced, ToolResult):
                return referenced.result
            return referenced
        return obj
    if isinstance(obj, list):
        return [_substitute_refs(x, results) for x in obj]
    if isinstance(obj, dict):
        return {k: _substitute_refs(v, results) for k, v in obj.items()}
    return obj


def _tool_results_by_local_step_id(agent_state: AgentState) -> dict[str, ToolResult]:
    planner_state = agent_state.node_states.planner
    plan = planner_state.plan
    if plan is None:
        return {}
    results_by_plan_step_id = {
        tool_result.plan_step_id: tool_result
        for tool_result in agent_state.gather_tool_results()
        if tool_result.plan_step_id is not None
    }
    tool_results_by_local_step_id: dict[str, ToolResult] = {}
    for step in plan.steps:
        tool_result = results_by_plan_step_id.get(step.db_id)
        if tool_result is not None:
            tool_results_by_local_step_id[step.id] = tool_result
    return tool_results_by_local_step_id


def _execute_step(
    step: PlanStep,
    *,
    tool_results_by_step_id: dict[str, ToolResult],
    allowed_tool_names: set[str] | None,
    resolved_args: dict[str, Any] | None = None,
    rejection_reason: str = "",
) -> StepExecutionResult:
    args = resolved_args if resolved_args is not None else _substitute_refs(step.args, tool_results_by_step_id)
    started_at = perf_counter()
    if rejection_reason:
        is_duplicate = "already made in this roundtrip" in rejection_reason
        output = ToolResult.error(
            rejection_reason,
            **({"status": "duplicate"} if is_duplicate else {}),
        )
        error_text = rejection_reason
    else:
        try:
            output = call_tool(name=step.tool, tool_input=args, allowed_tool_names=allowed_tool_names)
            error_text = ""
        except ValidationError as e:
            error_text = f"Invalid arguments for tool '{step.tool}': {e.errors(include_url=False)}"
            output = ToolResult(
                result={"error": error_text},
                evidence=[],
            )
        except Exception as e:
            error_text = f"Tool '{step.tool}' failed: {e}"
            output = ToolResult(
                result={"error": error_text, "tool": step.tool},
                evidence=[],
            )
    latency_ms = int((perf_counter() - started_at) * 1000)

    return StepExecutionResult(
        step=step,
        args=args,
        output=output,
        error_text=error_text,
        rejection_reason=rejection_reason,
        latency_ms=latency_ms,
    )


def execute_step(
    step: PlanStep,
    *,
    tool_results_by_step_id: dict[str, ToolResult] | None = None,
    agent_profile: AgentProfile | None = None,
    execution_context=None,
    resolved_args: dict[str, Any] | None = None,
) -> StepExecutionResult:
    """Execute one step with optional agent authorization.

    Top-level planner execution may omit an agent profile. In that case the
    tool registry performs normal existence and input validation without an
    agent-specific allowlist. Subagents should pass their profile so the
    allowlist remains enforced.
    """
    allowed_tool_names = None if agent_profile is None else set(agent_profile.tool_names)
    kwargs = {
        "tool_results_by_step_id": tool_results_by_step_id or {},
        "allowed_tool_names": allowed_tool_names,
    }
    if resolved_args is not None:
        kwargs["resolved_args"] = resolved_args
    if execution_context is None:
        return _execute_step(step, **kwargs)
    with bind_runtime_context(
        conversation_id=execution_context.conversation_id,
        conversation_model_config=execution_context.model_config,
        roundtrip_id=str(execution_context.roundtrip_id) if execution_context.roundtrip_id else None,
        user_id=execution_context.user_profile.user_id,
    ):
        return _execute_step(step, **kwargs)


def _record_step_result(
    agent_state: AgentState,
    *,
    plan: Plan | None,
    tool_repo: MessageToolRepository | None,
    iteration_number: int,
    execution_result: StepExecutionResult,
) -> None:
    execution_context = agent_state.execution_context
    step = execution_result.step
    output = execution_result.output
    if isinstance(output, ToolResult):
        output = output.model_copy(
            update={
                "tool_name": step.tool,
                "plan_step_id": step.db_id,
                "iteration": iteration_number,
                "hypothesis": step.hypothesis,
                "call_metadata": step.metadata,
            }
        )
    evidence_count = len(output.evidence) if isinstance(output, ToolResult) else 0
    status = "rejected" if execution_result.rejection_reason else ("error" if execution_result.error_text else "completed")
    record_node_log(
        roundtrip_id=execution_context.roundtrip_id,
        node_name=NodeName.EXECUTOR,
        stage="tool_call",
        event_type="tool_execution",
        level="error" if status == "error" else ("warning" if status == "rejected" else "info"),
        message=f"Tool execution {status}",
        payload={
            "tool_name": step.tool,
            "step_id": step.id,
            "plan_step_id": step.db_id,
            "status": status,
            "argument_keys": sorted(execution_result.args),
            "argument_preview": bounded_json_value(execution_result.args, max_bytes=4_000),
            "evidence_count": evidence_count,
            "error": execution_result.error_text[:2_000],
            "rejection_reason": execution_result.rejection_reason[:2_000],
        },
        duration_ms=execution_result.latency_ms,
    )
    if tool_repo and execution_context.roundtrip_id:
        tool_call_id = tool_repo.append_tool_call(
            execution_context.roundtrip_id,
            plan,
            step,
            input_payload=execution_result.args,
            output_payload=output,
            evidence=output.evidence if isinstance(output, ToolResult) else [],
            error_message=execution_result.error_text or None,
            duration_ms=execution_result.latency_ms,
            status="rejected" if execution_result.rejection_reason else None,
            is_subagent=agent_state.agent_profile.name != AgentName.REQUEST_ORCHESTRATOR,
        )
        agent_state.result = agent_state.result.with_recorded_tool_call(tool_call_id)


@traceable(name="Executor Node", process_inputs=compact_trace_inputs, process_outputs=compact_trace_outputs)
def run_executor(agent_state: AgentState) -> AgentState:
    planner_state = agent_state.node_states.planner
    plan = planner_state.plan
    if plan is None:
        return agent_state
    if agent_state.available_agent_states is not None:
        result = _run_top_level_executor(agent_state)
        append_execution_messages(result)
        return result
    tool_repo = MessageToolRepository() if isinstance(agent_state.execution_context.roundtrip_id, UUID) else None
    allowed_tool_names = set(agent_state.agent_profile.tool_names)
    iteration_number = planner_state.plan_count
    record_node_log(
        roundtrip_id=agent_state.execution_context.roundtrip_id,
        node_name=NodeName.EXECUTOR,
        stage="start",
        event_type="executor",
        message="Executor started",
        payload={"plan_step_count": len(plan.steps), "iteration": iteration_number},
    )

    with bind_runtime_context(
        conversation_id=agent_state.execution_context.conversation_id,
        conversation_model_config=agent_state.execution_context.model_config,
        roundtrip_id=str(agent_state.execution_context.roundtrip_id) if agent_state.execution_context.roundtrip_id else None,
        user_id=agent_state.execution_context.user_profile.user_id,
    ):
        with bind_agent_context(agent_name=agent_state.agent_profile.name):
            if plan is None or not plan.steps:
                return agent_state
            tool_results_by_step_id = _tool_results_by_local_step_id(agent_state)
            resolved_args_by_step_id: dict[str, dict[str, Any]] = {}
            rejection_reason_by_step_id: dict[str, str] = {}
            seen_request_hashes: set[str] = set()
            for step in plan.steps:
                resolved_args = _substitute_refs(
                    step.args,
                    tool_results_by_step_id,
                )
                resolved_args_by_step_id[step.id] = resolved_args
                if step.tool not in allowed_tool_names:
                    rejection_reason_by_step_id[step.id] = f"Tool '{step.tool}' is not allowed for this agent."
                    continue
                request_hash = build_signature({"tool_name": step.tool, "input": resolved_args})
                if request_hash in seen_request_hashes or (
                    tool_repo is not None
                    and agent_state.execution_context.roundtrip_id is not None
                    and tool_repo.has_request_hash(agent_state.execution_context.roundtrip_id, request_hash)
                ):
                    rejection_reason_by_step_id[step.id] = f"Tool request for '{step.tool}' was already made in this roundtrip."
                    continue
                seen_request_hashes.add(request_hash)

            with ThreadPoolExecutor(max_workers=len(plan.steps)) as executor:
                futures_by_step_id = {
                    step.id: executor.submit(
                        copy_context().run,
                        _execute_step,
                        step,
                        tool_results_by_step_id=tool_results_by_step_id,
                        allowed_tool_names=allowed_tool_names,
                        resolved_args=resolved_args_by_step_id[step.id],
                        rejection_reason=rejection_reason_by_step_id.get(step.id, ""),
                    )
                    for step in plan.steps
                }
                execution_results = [
                    futures_by_step_id[step.id].result()
                    for step in plan.steps
                ]

            for execution_result in execution_results:
                _record_step_result(
                    agent_state,
                    plan=plan,
                    tool_repo=tool_repo,
                    iteration_number=iteration_number,
                    execution_result=execution_result,
                )

    append_execution_messages(agent_state)
    return agent_state


def _run_top_level_executor(agent_state: AgentState) -> AgentState:
    """Execute a common-strategy plan as either a tool call or agent call."""
    from agents.registry import agent_registry
    from agent_runtime.models.agent_inputs import AgentInputs

    plan = agent_state.node_states.planner.plan
    if plan is None:
        return agent_state
    tool_repo = MessageToolRepository() if isinstance(agent_state.execution_context.roundtrip_id, UUID) else None
    iteration_number = agent_state.node_states.planner.plan_count
    for step in plan.steps:
        if step.agent:
            delegated_task = _delegated_task(step)
            request_signature = build_signature({
                "agent_name": step.agent,
                "task": delegated_task,
                "evidence_ids": step.args.get("evidence_ids", []),
            })
            if request_signature in agent_state.executed_request_signatures:
                record_node_log(
                    roundtrip_id=agent_state.execution_context.roundtrip_id,
                    node_name=NodeName.EXECUTOR,
                    stage="deduplication",
                    event_type="tool_execution",
                    level="warning",
                    message="Delegated request skipped as duplicate",
                    payload={"agent": step.agent, "step_id": step.id, "status": "duplicate"},
                )
                agent_state.direct_tool_results.append(
                    _duplicate_result(
                        capability_name=step.agent,
                        plan_step_id=step.db_id,
                    )
                )
                continue
            agent_state.executed_request_signatures.append(request_signature)
            delegated = agent_state.available_agent_states.get(step.agent)
            if delegated is None:
                agent_state.direct_tool_results.append(
                    ToolResult.error(
                        f"Delegated agent '{step.agent}' is not available.",
                        status="error",
                    ).model_copy(update={
                        "tool_name": step.agent,
                        "plan_step_id": step.db_id,
                    })
                )
                continue
            delegated_run = AgentState.new(
                agent_profile=delegated.agent_profile,
                execution_context=delegated.execution_context,
                llm=delegated.llm,
                inputs=AgentInputs.new(
                    task=delegated_task,
                    request_task=delegated.inputs.request_task,
                    tool_names=delegated.inputs.tool_names,
                    tool_category_names=delegated.inputs.tool_category_names,
                    discovered_capabilities=delegated.inputs.discovered_capabilities,
                    discovered_attribute_types=delegated.inputs.discovered_attribute_types,
                    evidence_ids=step.args.get("evidence_ids", []),
                ),
            )
            delegated_run.available_agent_states = delegated.available_agent_states
            runner = agent_registry.get(delegated.agent_profile)
            completed = runner(delegated_run)
            completion_result = _agent_completion_result(
                agent_name=step.agent,
                state=completed,
                plan_step_id=step.db_id,
                task=delegated_task,
            )
            if tool_repo is not None and agent_state.execution_context.roundtrip_id is not None:
                # Persist the delegated agent as a tool-like call so its
                # evidence, especially generated-file URLs, survives into the
                # parent result and UI payload.
                persisted_step = step.model_copy(update={"tool": step.agent, "agent": ""})
                tool_call_id = tool_repo.append_tool_call(
                    agent_state.execution_context.roundtrip_id,
                    plan,
                    persisted_step,
                    input_payload=step.args,
                    output_payload=completion_result,
                    evidence=completion_result.evidence,
                    duration_ms=0,
                    is_subagent=True,
                )
                agent_state.result = agent_state.result.with_recorded_tool_call(tool_call_id)
            else:
                agent_state.direct_tool_results.append(completion_result)
            continue

        resolved_args = _substitute_refs(step.args, {})
        request_signature = build_signature({
            "tool_name": step.tool,
            "input": resolved_args,
        })
        if request_signature in agent_state.executed_request_signatures:
            record_node_log(
                roundtrip_id=agent_state.execution_context.roundtrip_id,
                node_name=NodeName.EXECUTOR,
                stage="deduplication",
                event_type="tool_execution",
                level="warning",
                message="Tool request skipped as duplicate",
                payload={"tool_name": step.tool, "step_id": step.id, "status": "duplicate"},
            )
            agent_state.direct_tool_results.append(_duplicate_result(
                capability_name=step.tool,
                plan_step_id=step.db_id,
            ))
            continue
        if (
            tool_repo is not None
            and agent_state.execution_context.roundtrip_id is not None
            and tool_repo.has_request_hash(agent_state.execution_context.roundtrip_id, request_signature)
        ):
            agent_state.executed_request_signatures.append(request_signature)
            record_node_log(
                roundtrip_id=agent_state.execution_context.roundtrip_id,
                node_name=NodeName.EXECUTOR,
                stage="deduplication",
                event_type="tool_execution",
                level="warning",
                message="Tool request skipped because it was already persisted",
                payload={"tool_name": step.tool, "step_id": step.id, "status": "duplicate"},
            )
            agent_state.direct_tool_results.append(_duplicate_result(
                capability_name=step.tool,
                plan_step_id=step.db_id,
            ))
            continue
        agent_state.executed_request_signatures.append(request_signature)
        execution_result = execute_step(
            step,
            tool_results_by_step_id={},
            resolved_args=resolved_args,
            execution_context=agent_state.execution_context,
        )
        output = execution_result.output
        if not isinstance(output, ToolResult):
            output = ToolResult(result=output, evidence=[])
        execution_result = StepExecutionResult(
            step=step,
            args=resolved_args,
            output=output,
        )
        if tool_repo is not None:
            _record_step_result(
                agent_state,
                plan=plan,
                tool_repo=tool_repo,
                iteration_number=iteration_number,
                execution_result=execution_result,
            )
        else:
            agent_state.direct_tool_results.append(
                output.model_copy(update={
                    "tool_name": step.tool,
                    "plan_step_id": step.db_id,
                    "hypothesis": step.hypothesis,
                    "call_metadata": step.metadata,
                })
            )
    return agent_state


def _duplicate_result(*, capability_name: str, plan_step_id) -> ToolResult:
    return ToolResult(
        plan_step_id=plan_step_id,
        tool_name=capability_name,
        result={
            "error": f"Request for '{capability_name}' was already completed in this roundtrip.",
            "status": "duplicate",
        },
    )


def _agent_completion_result(
    *,
    agent_name: str,
    state: AgentState,
    plan_step_id,
    task: str,
) -> ToolResult:
    """Translate internal subagent state into the parent capability contract."""
    child_results = state.gather_tool_results()
    evidence = []
    seen_evidence_ids = set()
    for child_result in child_results:
        for item in child_result.evidence:
            if item.id in seen_evidence_ids:
                continue
            seen_evidence_ids.add(item.id)
            evidence.append(item)

    status = state.result.answer_status.value
    if state.result.execution_status.value not in {"pending", "success"}:
        status = state.result.execution_status.value
    summaries = [item.summary.strip() for item in evidence if item.summary.strip()]
    summary = (
        f"{agent_name} completed the delegated task. " + " ".join(dict.fromkeys(summaries))
        if summaries
        else f"{agent_name} completed the delegated task."
    )
    outcome = None
    if state.agent_profile.execution_strategy == AgentExecutionStrategy.PLANNER_EXECUTOR_EVALUATOR:
        evaluation_status = state.node_states.evaluator.evaluation_status
        outcome_status = {
            EVALUATION_STATUS_SATISFIED: ToolOutcomeStatus.COMPLETED,
            EVALUATION_STATUS_PARTIAL_BUT_SUFFICIENT: ToolOutcomeStatus.PARTIAL,
            EVALUATION_STATUS_RETRYABLE: ToolOutcomeStatus.NEEDS_INPUT,
            EVALUATION_STATUS_TERMINAL: ToolOutcomeStatus.FAILED,
        }[evaluation_status]
        missing_inputs = list(state.node_states.evaluator.missing_information)
        outcome = ToolOutcome(
            status=outcome_status,
            reason=("; ".join(missing_inputs) if missing_inputs else None),
            missing_inputs=missing_inputs,
        )
    return ToolResult(
        tool_call_id=plan_step_id,
        plan_step_id=plan_step_id,
        tool_name=agent_name,
        result={
            "agent": agent_name,
            "task": task,
            "evidence_ids": state.inputs.evidence_ids,
            "status": status,
            "summary": summary,
            "evidence": [
                bounded_json_value(
                    item.model_dump(mode="json", exclude={"raw_payload"}),
                    max_bytes=8_000,
                )
                for item in evidence
            ],
        },
        evidence=evidence,
        state=outcome,
    )


def _delegated_task(step: PlanStep) -> str:
    """Use the explicit delegation task, falling back only for legacy plans."""
    explicit_task = step.args.get("task") if isinstance(step.args, dict) else None
    if isinstance(explicit_task, str) and explicit_task.strip():
        return explicit_task.strip()
    if step.args:
        return (
            "Execute the delegated request using these parameters:\n"
            + json.dumps(step.args, ensure_ascii=False, indent=2, default=str)
        )
    return step.plan or step.hypothesis
