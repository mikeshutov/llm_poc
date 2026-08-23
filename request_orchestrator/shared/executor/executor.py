from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import UUID

from langsmith import traceable
from pydantic import ValidationError

from common.data import sanitize_for_json_storage
from common.signatures import build_signature
from common.logging import create_conversation_event
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.evidence import ToolResult
from request_orchestrator.models.plan import Plan
from request_orchestrator.models.plan import PlanStep
from request_orchestrator.shared.runtime_context import bind_agent_context, bind_runtime_context
from tool.registry import call_tool
from tool.repository.tool_call_repository import ToolCallRepository
from rendering.debug import TOOL_CALL_KIND


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
        return StepExecutionResult(
            step=step,
            args=args,
            output=ToolResult.error(rejection_reason),
            error_text=rejection_reason,
            rejection_reason=rejection_reason,
        )
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
        latency_ms=latency_ms,
    )


def _record_step_result(
    agent_state: AgentState,
    *,
    plan: Plan | None,
    tool_repo: ToolCallRepository | None,
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
                "iteration": iteration_number,
            }
        )
    payload = {
        "agent_name": agent_state.agent_profile.name,
        "kind": TOOL_CALL_KIND,
        "tool_name": step.tool,
        "iteration": iteration_number,
        "request": sanitize_for_json_storage(execution_result.args),
        "response": sanitize_for_json_storage(output),
        "data": sanitize_for_json_storage({
            "step_plan": step.plan,
            "latency_ms": execution_result.latency_ms,
        }),
    }
    if execution_result.error_text:
        payload["error"] = execution_result.error_text
    create_conversation_event(
        conversation_id=execution_context.conversation_id,
        roundtrip_id=execution_context.roundtrip_id,
        event_type=TOOL_CALL_KIND,
        source=agent_state.agent_profile.name,
        agent_name=agent_state.agent_profile.name,
        node_name="tool_call",
        iteration=iteration_number,
        payload=payload,
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
        )
        agent_state.result = agent_state.result.with_recorded_tool_call(tool_call_id)


@traceable(name="Executor Node")
def run_executor(agent_state: AgentState) -> AgentState:
    planner_state = agent_state.node_states.planner
    plan = planner_state.plan
    if plan is None:
        return agent_state
    tool_repo = ToolCallRepository() if isinstance(agent_state.execution_context.roundtrip_id, UUID) else None
    allowed_tool_names = set(agent_state.agent_profile.tool_names)
    iteration_number = planner_state.plan_count

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

    return agent_state
