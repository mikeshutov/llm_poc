from __future__ import annotations

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
from agent_runtime.models.agent_inputs import AgentInputs
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.agent_profile import AgentExecutionStrategy, AgentName
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
from capabilities.registry.global_registry import available_agent_tools, call_tool
from capabilities.registry.repository.message_repository import MessageToolRepository
from agent_runtime.execution.message_history import append_execution_messages

MAX_IDENTICAL_TOOL_CALLS = 2


@dataclass(frozen=True)
class StepExecutionResult:
    step: PlanStep
    args: dict[str, Any]
    output: ToolResult
    error_text: str = ""
    rejection_reason: str = ""
    latency_ms: int = 0

    @property
    def status(self) -> str:
        return tool_result_status(self.output)


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


def tool_result_status(result: ToolResult) -> str:
    """Classify the common result for persistence and strategy completion."""
    payload = result.result if isinstance(result.result, dict) else {}
    if payload.get("status") in {"rejected", "duplicate"}:
        return "rejected"
    if (
        payload.get("error")
        or payload.get("status") in {"error", "failed"}
        or (result.state is not None and result.state.status == ToolOutcomeStatus.FAILED)
    ):
        return "error"
    return "completed"


def execute_step(
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
            status="duplicate" if is_duplicate else "rejected",
        )
    else:
        try:
            output = call_tool(name=step.tool, tool_input=args, allowed_tool_names=allowed_tool_names)
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
    if not isinstance(output, ToolResult):
        output = ToolResult(result=output, evidence=[])
    result_status = tool_result_status(output)
    error_text = ""
    if result_status in {"error", "rejected"}:
        payload = output.result if isinstance(output.result, dict) else {}
        error_text = str(payload.get("error") or (
            output.state.reason if output.state is not None else None
        ) or f"Tool '{step.tool}' reported {result_status}.")
        if result_status == "rejected":
            rejection_reason = error_text
    latency_ms = int((perf_counter() - started_at) * 1000)

    return StepExecutionResult(
        step=step,
        args=args,
        output=output,
        error_text=error_text,
        rejection_reason=rejection_reason,
        latency_ms=latency_ms,
    )


def execute_agent_tool(
    agent_state: AgentState, agent_name: str, args: dict[str, Any], *, tool_name: str | None = None,
) -> ToolResult:
    """Run an agent-backed registry tool and return the common tool result."""
    if not agent_state.agent_profile.can_delegate_agents:
        return ToolResult.error(
            f"Agent '{agent_state.agent_profile.name}' is not allowed to delegate agents.",
            status="rejected",
        )
    if agent_name == agent_state.agent_profile.name:
        return ToolResult.error(f"Agent '{agent_name}' cannot delegate to itself.", status="rejected")

    delegated = (agent_state.available_agent_states or {}).get(agent_name)
    if tool_name is not None:
        facade = agent_state.agent_profile.tools_by_name.get(tool_name)
        authorized = (
            facade is not None
            and (getattr(facade, "metadata", None) or {}).get("agent_name") == agent_name
        )
    else:
        authorized = agent_name in {tool.name for tool in available_agent_tools(agent_state)}
    if delegated is None or not authorized:
        return ToolResult.error(f"Delegated agent '{agent_name}' is not available.", status="rejected")

    from agents.registry import agent_registry

    delegated_run = AgentState.new(
        agent_profile=delegated.agent_profile,
        execution_context=delegated.execution_context,
        llm=delegated.llm,
        inputs=AgentInputs.new(
            task=args["task"],
            request_task=delegated.inputs.request_task,
            tool_names=delegated.inputs.tool_names,
            tool_category_names=delegated.inputs.tool_category_names,
            discovered_capabilities=delegated.inputs.discovered_capabilities,
            discovered_attribute_types=delegated.inputs.discovered_attribute_types,
            evidence_ids=args.get("evidence_ids", []),
        ),
    )
    delegated_run.available_agent_states = delegated.available_agent_states
    completed = agent_registry.get(delegated.agent_profile)(delegated_run)
    return _agent_completion_result(
        agent_name=agent_name,
        state=completed,
        task=delegated_run.inputs.task,
    )


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
    output = execution_result.output.model_copy(update={
        "tool_name": step.tool,
        "plan_step_id": step.db_id,
        "iteration": iteration_number,
        "hypothesis": step.hypothesis,
        "call_metadata": step.metadata,
    })
    if tool_repo is None:
        agent_state.direct_tool_results.append(output)
    evidence_count = len(output.evidence)
    status = execution_result.status
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
            evidence=output.evidence,
            error_message=execution_result.error_text or None,
            duration_ms=execution_result.latency_ms,
            status=status,
            is_subagent=agent_state.agent_profile.name != AgentName.REQUEST_ORCHESTRATOR,
        )
        agent_state.result = agent_state.result.with_recorded_tool_call(tool_call_id)


@traceable(name="Executor Node", process_inputs=compact_trace_inputs, process_outputs=compact_trace_outputs)
def run_executor(agent_state: AgentState) -> AgentState:
    """Execute all planned registry tools and record their results in plan order."""
    planner_state = agent_state.node_states.planner
    plan = planner_state.plan
    if plan is None or not plan.steps:
        return agent_state
    context = agent_state.execution_context
    tool_repo = MessageToolRepository() if isinstance(context.roundtrip_id, UUID) else None
    allowed = set(agent_state.agent_profile.tool_names)
    allowed.update((agent_state.available_agent_states or {}).keys())
    prior_results = _tool_results_by_local_step_id(agent_state)
    pending: list[tuple[PlanStep, dict[str, Any], str]] = []
    batch_counts: dict[str, int] = {}
    for step in plan.steps:
        args = _substitute_refs(step.args, prior_results)
        rejection = ""
        if step.tool not in allowed:
            rejection = f"Tool '{step.tool}' is not allowed for this agent."
        else:
            signature = build_signature({"tool_name": step.tool, "input": args})
            count = (
                tool_repo.count_request_hash(context.roundtrip_id, signature)
                if tool_repo is not None
                else agent_state.executed_request_signatures.count(signature)
            )
            # Reserve in-flight calls as well as previously completed calls.
            in_batch = batch_counts.get(signature, 0)
            if count + in_batch >= MAX_IDENTICAL_TOOL_CALLS:
                rejection = f"Tool request for '{step.tool}' was already made in this roundtrip."
            else:
                batch_counts[signature] = in_batch + 1
        pending.append((step, args, rejection))

    with bind_runtime_context(
        conversation_id=context.conversation_id,
        conversation_model_config=context.model_config,
        roundtrip_id=str(context.roundtrip_id) if context.roundtrip_id else None,
        user_id=context.user_profile.user_id,
    ), bind_agent_context(agent_name=agent_state.agent_profile.name, agent_state=agent_state):
        with ThreadPoolExecutor(max_workers=len(pending)) as executor:
            futures = [
                executor.submit(
                    copy_context().run, execute_step, step,
                    tool_results_by_step_id=prior_results,
                    allowed_tool_names=allowed,
                    resolved_args=args,
                    rejection_reason=rejection,
                )
                for step, args, rejection in pending
            ]
            results = [future.result() for future in futures]
        for (step, args, rejection), result in zip(pending, results):
            if not rejection:
                agent_state.executed_request_signatures.append(
                    build_signature({"tool_name": step.tool, "input": args})
                )
            _record_step_result(
                agent_state,
                plan=plan,
                tool_repo=tool_repo,
                iteration_number=planner_state.plan_count,
                execution_result=result,
            )
    append_execution_messages(agent_state)
    return agent_state


def _agent_completion_result(
    *,
    agent_name: str,
    state: AgentState,
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
    summaries = [item.summary.strip() for item in evidence if item.summary and item.summary.strip()]
    status_text = {
        "success": "completed the delegated task",
        "partial_success": "partially completed the delegated task",
        "partial": "partially completed the delegated task",
        "no_new_work": "stopped because no new work was available",
        "failed": "failed to complete the delegated task",
        "pending": "has not completed the delegated task",
    }[status]
    summary = " ".join([f"{agent_name} {status_text}.", *dict.fromkeys(summaries)])
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
        tool_name=agent_name,
        result={
            "agent": agent_name,
            "task": task,
            "evidence_ids": state.inputs.evidence_ids,
            "status": status,
            "summary": summary,
            "output_evidence_ids": [item.id for item in evidence],
        },
        evidence=evidence,
        state=outcome,
    )
