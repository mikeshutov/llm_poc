from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from langsmith import traceable
from pydantic import ValidationError

from request_orchestrator.models.agent_state import AgentState, IterationState
from request_orchestrator.models.plan import PlanStep
from request_orchestrator.models.plan_step_ids import format_plan_step_id
from request_orchestrator.shared.runtime_context import bind_runtime_context
from tool.registry import call_tool
from tool.repository.tool_call_repository import ToolCallRepository
from rendering.debug import TOOL_CALL_KIND


@dataclass(frozen=True)
class StepExecutionResult:
    step: PlanStep
    args: dict[str, Any]
    output: Any
    error_text: str = ""
    latency_ms: int = 0


def _substitute_refs(obj, results: dict, *, iteration_number: int):
    if isinstance(obj, str):
        if obj.startswith("#E"):
            raw_step_id = obj[1:]
            qualified_step_id = format_plan_step_id(iteration_number, raw_step_id)
            return results.get(qualified_step_id, obj)
        return obj
    if isinstance(obj, list):
        return [_substitute_refs(x, results, iteration_number=iteration_number) for x in obj]
    if isinstance(obj, dict):
        return {k: _substitute_refs(v, results, iteration_number=iteration_number) for k, v in obj.items()}
    return obj


def _execute_step(
    step: PlanStep,
    *,
    iteration: IterationState,
    iteration_number: int,
    allowed_tool_names: set[str] | None,
) -> StepExecutionResult:
    args = _substitute_refs(step.args, iteration.results, iteration_number=iteration_number)
    started_at = perf_counter()
    try:
        output = call_tool(name=step.tool, tool_input=args, allowed_tool_names=allowed_tool_names)
        error_text = ""
    except ValidationError as e:
        output = {"error": f"Invalid arguments for tool '{step.tool}': {e.errors(include_url=False)}"}
        error_text = str(output["error"])
    except Exception as e:
        output = {"error": f"Tool '{step.tool}' failed: {e}", "tool": step.tool}
        error_text = str(output["error"])
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
    iteration: IterationState,
    tool_repo: ToolCallRepository | None,
    iteration_number: int,
    execution_result: StepExecutionResult,
) -> None:
    step = execution_result.step
    qualified_step_id = format_plan_step_id(iteration_number, step.id)
    iteration.results[qualified_step_id] = execution_result.output

    agent_state.log_status(
        agent_name=agent_state.agent_profile.name,
        kind=TOOL_CALL_KIND,
        tool_name=step.tool,
        step_id=qualified_step_id,
        iteration=iteration_number,
        request=execution_result.args,
        response=execution_result.output,
        error=execution_result.error_text,
        data={
            "step_plan": step.plan,
            "latency_ms": execution_result.latency_ms,
        },
    )

    if tool_repo and agent_state.roundtrip_id:
        tool_repo.append_tool_call(
            agent_state.roundtrip_id,
            iteration,
            step,
            input_payload=execution_result.args,
            output_payload=execution_result.output,
            error_message=execution_result.error_text or None,
            duration_ms=execution_result.latency_ms,
        )


@traceable(name="Executor Node")
def run_executor(agent_state: AgentState) -> AgentState:
    iteration = agent_state.iteration_trace[-1]
    tool_repo = ToolCallRepository() if agent_state.roundtrip_id and agent_state.agent_profile.persist_tool_calls else None
    allowed_tool_names = agent_state.agent_profile.allowed_tool_names()
    iteration_number = len(agent_state.iteration_trace)

    with bind_runtime_context(
        conversation_id=agent_state.conversation_id,
        conversation_model_config=agent_state.conversation_model_config,
        roundtrip_id=str(agent_state.roundtrip_id) if agent_state.roundtrip_id else None,
        user_id=agent_state.user_profile.user_id,
    ):
        plan = iteration.plan
        if plan is None or not plan.steps:
            return agent_state

        with ThreadPoolExecutor(max_workers=len(plan.steps)) as executor:
            futures_by_step_id = {
                step.id: executor.submit(
                    copy_context().run,
                    _execute_step,
                    step,
                    iteration=iteration,
                    iteration_number=iteration_number,
                    allowed_tool_names=allowed_tool_names,
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
                iteration=iteration,
                tool_repo=tool_repo,
                iteration_number=iteration_number,
                execution_result=execution_result,
            )

    return agent_state
