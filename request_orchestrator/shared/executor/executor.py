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
from request_orchestrator.agent_runner.models.agent_profile import AgentProfile
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
        output = ToolResult.error(rejection_reason)
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
            "step_hypothesis": step.hypothesis,
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
        agent_version=agent_state.agent_profile.version,
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


def _native_name(request: dict[str, Any]) -> str:
    return str(request.get("name", ""))


def _native_args(request: dict[str, Any]) -> dict[str, Any]:
    args = request.get("args", {})
    return args if isinstance(args, dict) else {}


def _native_parameters(request: dict[str, Any]) -> dict[str, Any]:
    parameters = _native_args(request).get("parameters", {})
    return parameters if isinstance(parameters, dict) else {}


def _native_hypothesis(request: dict[str, Any]) -> str:
    return str(_native_args(request).get("hypothesis", ""))


def _native_metadata(request: dict[str, Any]) -> dict[str, Any]:
    metadata = _native_args(request).get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _execute_native_request(
    request: dict[str, Any],
    *,
    agent_state: AgentState,
    allowed_tool_names: set[str] | None,
    iteration_number: int,
) -> tuple[dict[str, Any], ToolResult, str, int]:
    started_at = perf_counter()
    error_text = ""
    try:
        output = call_tool(
            name=_native_name(request),
            tool_input=_native_parameters(request),
            allowed_tool_names=allowed_tool_names,
        )
        if not isinstance(output, ToolResult):
            output = ToolResult(result=output, evidence=[])
    except ValidationError as exc:
        error_text = f"Invalid arguments for tool '{_native_name(request)}': {exc.errors(include_url=False)}"
        output = ToolResult.error(error_text, tool=_native_name(request))
    except Exception as exc:
        error_text = f"Tool '{_native_name(request)}' failed: {exc}"
        output = ToolResult.error(error_text, tool=_native_name(request))
    latency_ms = int((perf_counter() - started_at) * 1000)
    output = output.model_copy(update={
        "tool_name": _native_name(request),
        "iteration": iteration_number,
        "hypothesis": _native_hypothesis(request),
        "call_metadata": _native_metadata(request),
    })
    return request, output, error_text, latency_ms


def _record_native_result(
    agent_state: AgentState,
    *,
    request: dict[str, Any],
    output: ToolResult,
    plan_step_id: UUID | None,
    error_text: str,
    latency_ms: int,
    tool_repo: ToolCallRepository | None,
    iteration_number: int,
) -> None:
    output = output.model_copy(update={"plan_step_id": plan_step_id})
    execution_context = agent_state.execution_context
    payload = {
        "agent_name": agent_state.agent_profile.name,
        "kind": TOOL_CALL_KIND,
        "tool_name": _native_name(request),
        "iteration": iteration_number,
        "request": sanitize_for_json_storage(_native_parameters(request)),
        "response": sanitize_for_json_storage(output),
        "data": sanitize_for_json_storage({
            "hypothesis": _native_hypothesis(request),
            "metadata": _native_metadata(request),
            "latency_ms": latency_ms,
        }),
    }
    if error_text:
        payload["error"] = error_text
    create_conversation_event(
        conversation_id=execution_context.conversation_id,
        roundtrip_id=execution_context.roundtrip_id,
        event_type=TOOL_CALL_KIND,
        source=agent_state.agent_profile.name,
        agent_name=agent_state.agent_profile.name,
        agent_version=agent_state.agent_profile.version,
        node_name="tool_call",
        iteration=iteration_number,
        payload=payload,
    )
    if tool_repo and execution_context.roundtrip_id:
        tool_call_id = tool_repo.append_native_tool_call(
            execution_context.roundtrip_id,
            request,
            plan_id=agent_state.node_states.planner.plan.db_id if agent_state.node_states.planner.plan else None,
            plan_step_id=plan_step_id,
            input_payload=_native_parameters(request),
            output_payload=output,
            evidence=output.evidence,
            error_message=error_text or None,
            duration_ms=latency_ms,
        )
        output.tool_call_id = tool_call_id
        agent_state.result = agent_state.result.with_recorded_tool_call(tool_call_id)
    if tool_repo is None:
        agent_state.direct_tool_results.append(output)


def _run_native_tool_calls(agent_state: AgentState) -> AgentState:
    tool_calls = agent_state.node_states.planner.tool_calls
    if not tool_calls:
        return agent_state
    tool_repo = ToolCallRepository() if isinstance(agent_state.execution_context.roundtrip_id, UUID) else None
    allowed_tool_names = set(agent_state.agent_profile.tool_names)
    iteration_number = agent_state.node_states.planner.plan_count
    direct_requests = [request for request in tool_calls if _native_name(request) != "delegate_agent"]
    step_ids = {
        id(request): agent_state.node_states.planner.plan.steps[index].db_id
        for index, request in enumerate(tool_calls)
    } if agent_state.node_states.planner.plan else {}
    seen_signatures: set[str] = set()
    executable_requests: list[tuple[dict[str, Any], UUID | None]] = []
    rejected: list[tuple[dict[str, Any], UUID | None, str]] = []
    for request in direct_requests:
        step_id = step_ids.get(id(request))
        name = _native_name(request)
        signature = build_signature({"tool_name": name, "input": _native_parameters(request)})
        if name not in allowed_tool_names:
            rejected.append((request, step_id, f"Tool '{name}' is not allowed for this agent."))
        elif signature in seen_signatures or (tool_repo and tool_repo.has_request_hash(agent_state.execution_context.roundtrip_id, signature)):
            rejected.append((request, step_id, f"Tool request for '{name}' was already made in this roundtrip."))
        else:
            seen_signatures.add(signature)
            executable_requests.append((request, step_id))

    with bind_runtime_context(
        conversation_id=agent_state.execution_context.conversation_id,
        conversation_model_config=agent_state.execution_context.model_config,
        roundtrip_id=str(agent_state.execution_context.roundtrip_id) if agent_state.execution_context.roundtrip_id else None,
        user_id=agent_state.execution_context.user_profile.user_id,
    ):
        with bind_agent_context(agent_name=agent_state.agent_profile.name):
            with ThreadPoolExecutor(max_workers=max(1, len(executable_requests))) as executor:
                futures = [executor.submit(_execute_native_request, request, agent_state=agent_state, allowed_tool_names=allowed_tool_names, iteration_number=iteration_number) for request, step_id in executable_requests]
                results = [future.result() for future in futures]
            for (request, step_id), (_, output, error_text, latency_ms) in zip(executable_requests, results):
                _record_native_result(agent_state, request=request, output=output, plan_step_id=step_id, error_text=error_text, latency_ms=latency_ms, tool_repo=tool_repo, iteration_number=iteration_number)
            for request, step_id, rejection_reason in rejected:
                output = ToolResult.error(rejection_reason, tool=_native_name(request)).model_copy(update={
                    "tool_name": _native_name(request),
                    "iteration": iteration_number,
                    "hypothesis": _native_hypothesis(request),
                    "call_metadata": _native_metadata(request),
                })
                _record_native_result(agent_state, request=request, output=output, plan_step_id=step_id, error_text=rejection_reason, latency_ms=0, tool_repo=tool_repo, iteration_number=iteration_number)

    # Delegated agents mutate their own state and are intentionally dispatched
    # after direct calls so direct tool execution remains parallel and isolated.
    if agent_state.available_agent_states is not None:
        from request_orchestrator.agents.registry import agent_registry
        from request_orchestrator.models.agent_inputs import AgentInputs
        for request in (item for item in tool_calls if _native_name(item) == "delegate_agent"):
            step_id = step_ids.get(id(request))
            parameters = _native_parameters(request)
            agent_name = parameters.get("agent_name", "")
            delegated = agent_state.available_agent_states.get(agent_name)
            if delegated is None:
                continue
            delegated.inputs = AgentInputs.new(task=parameters["task"], request_task=delegated.inputs.request_task)
            delegated = agent_registry.get(delegated.agent_profile)(delegated)
            agent_state.delegated_agent_states[agent_name] = delegated
            output = ToolResult(
                result={
                    "agent_name": agent_name,
                    "task": parameters.get("task", ""),
                    "delegated": True,
                },
                evidence=[],
            ).model_copy(update={
                "tool_name": _native_name(request),
                "iteration": iteration_number,
                "hypothesis": _native_hypothesis(request),
                "call_metadata": _native_metadata(request),
            })
            _record_native_result(
                agent_state,
                request=request,
                output=output,
                plan_step_id=step_id,
                error_text="",
                latency_ms=0,
                tool_repo=tool_repo,
                iteration_number=iteration_number,
            )
    return agent_state


@traceable(name="Executor Node")
def run_executor(agent_state: AgentState) -> AgentState:
    if agent_state.node_states.planner.tool_calls:
        return _run_native_tool_calls(agent_state)
    planner_state = agent_state.node_states.planner
    plan = planner_state.plan
    if plan is None:
        return agent_state
    if agent_state.available_agent_states is not None:
        return _run_top_level_executor(agent_state)
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


def _run_top_level_executor(agent_state: AgentState) -> AgentState:
    """Execute a common-strategy plan as either a tool call or agent call."""
    from request_orchestrator.agents.registry import agent_registry
    from request_orchestrator.models.agent_inputs import AgentInputs

    plan = agent_state.node_states.planner.plan
    if plan is None:
        return agent_state
    tool_repo = ToolCallRepository() if isinstance(agent_state.execution_context.roundtrip_id, UUID) else None
    iteration_number = agent_state.node_states.planner.plan_count
    for step in plan.steps:
        if step.agent:
            request_signature = build_signature({
                "agent_name": step.agent,
                "task": step.plan,
            })
            if request_signature in agent_state.executed_request_signatures:
                continue
            agent_state.executed_request_signatures.append(request_signature)
            delegated = agent_state.available_agent_states.get(step.agent)
            if delegated is None:
                continue
            delegated.inputs = AgentInputs.new(
                task=step.plan,
                request_task=delegated.inputs.request_task,
            )
            runner = agent_registry.get(delegated.agent_profile)
            agent_state.delegated_agent_states[step.agent] = runner(delegated)
            continue

        resolved_args = _substitute_refs(step.args, {})
        request_signature = build_signature({
            "tool_name": step.tool,
            "input": resolved_args,
        })
        if request_signature in agent_state.executed_request_signatures:
            continue
        if (
            tool_repo is not None
            and agent_state.execution_context.roundtrip_id is not None
            and tool_repo.has_request_hash(agent_state.execution_context.roundtrip_id, request_signature)
        ):
            agent_state.executed_request_signatures.append(request_signature)
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
                output.model_copy(update={"tool_name": step.tool})
            )
    return agent_state
