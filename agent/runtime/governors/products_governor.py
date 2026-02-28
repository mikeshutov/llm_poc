from dataclasses import dataclass
from typing import Any

from agent.models.tool_call_trace import ToolCallTrace
from agent.runtime.governors.base import GovernorAction, LoopGovernor
from agent.runtime.planner_trace import build_tool_trace
from agent.state import _AgentState, _LastRetrievalOutput

MIN_FALLBACK_RESULTS = 3


@dataclass(frozen=True)
class _GuardResult:
    action: str  # none | continue | stop
    reason: str | None = None
    tool_name: str | None = None
    status: str | None = None
    input_payload: dict[str, Any] | None = None
    done: bool | None = None


def _append_guard_trace(
    *,
    state: _AgentState,
    turn_index: int,
    tool_name: str,
    status: str,
    reason: str | None,
    input_payload: dict[str, Any],
    goal: str,
    done: bool | None,
) -> ToolCallTrace:
    trace = build_tool_trace(
        call_index=state.loop_state.call_index,
        turn_index=turn_index,
        tool_name=tool_name,
        status=status,
        reason=reason,
        input_payload=input_payload,
        output_payload=None,
        error_message=None,
        duration_ms=0,
        goal=goal,
        done=done,
    )
    state.add_traces([trace])
    state.loop_state.call_index += 1
    return trace


def _maybe_require_tool_before_finalize(
    *,
    goal_reached: bool,
    executed_tool_names: set[str],
    required_tool_name: str,
    tool_name: str,
    skipped_status: str,
) -> _GuardResult:
    if not goal_reached:
        return _GuardResult(action="none")
    if required_tool_name in executed_tool_names:
        return _GuardResult(action="none")
    return _GuardResult(
        action="continue",
        reason=f"Finalize rejected: required tool '{required_tool_name}' not executed this turn.",
        tool_name=tool_name,
        status=skipped_status,
        input_payload={},
        done=False,
    )


def _maybe_force_fallback_turn(
    *,
    turn_index: int,
    max_turns: int,
    retrieval_called_this_turn: bool,
    allow_web_fallback: bool,
    allow_web_fallback_next_turn: bool,
    retrieval_output: _LastRetrievalOutput,
    weather_fetched_this_turn: bool,
    goal_reached: bool,
    tool_name: str,
    skipped_status: str,
) -> _GuardResult:
    should_force = (
        turn_index + 1 < max_turns
        and retrieval_called_this_turn
        and not allow_web_fallback
        and retrieval_output.internal_count == 0
        and retrieval_output.external_count == 0
        and not weather_fetched_this_turn
        and goal_reached
    )
    if not should_force:
        return _GuardResult(action="none")
    return _GuardResult(
        action="continue",
        reason="Deferring finalize: forcing a follow-up retrieval turn before web fallback is eligible.",
        tool_name=tool_name,
        status=skipped_status,
        input_payload={
            "internal_count": retrieval_output.internal_count,
            "external_count": retrieval_output.external_count,
            "allow_web_fallback_this_turn": allow_web_fallback,
            "allow_web_fallback_next_turn": allow_web_fallback_next_turn,
        },
        done=False,
    )


def _maybe_stop_no_progress(
    *,
    previous_query_text: str,
    current_query_text: str,
    previous_filters_dump: dict[str, Any] | None,
    current_filters_dump: dict[str, Any] | None,
    has_existing_results: bool,
    tool_name: str,
    skipped_status: str,
) -> _GuardResult:
    should_stop = (
        current_query_text == previous_query_text
        and current_filters_dump == previous_filters_dump
        and has_existing_results
    )
    if not should_stop:
        return _GuardResult(action="none")
    return _GuardResult(
        action="stop",
        reason="No progress: query/filters unchanged; stopping iteration.",
        tool_name=tool_name,
        status=skipped_status,
        input_payload={
            "query_text": current_query_text,
            "common_filters": current_filters_dump,
        },
        done=False,
    )


def _maybe_stop_fallback_threshold(
    *,
    retrieval_output: _LastRetrievalOutput,
    tool_name: str,
    skipped_status: str,
) -> _GuardResult:
    should_stop = (
        retrieval_output.internal_count == 0
        and retrieval_output.external_count >= MIN_FALLBACK_RESULTS
    )
    if not should_stop:
        return _GuardResult(action="none")
    return _GuardResult(
        action="stop",
        reason="Fallback threshold met; stopping further retrieval with available external results.",
        tool_name=tool_name,
        status=skipped_status,
        input_payload={
            "internal_count": retrieval_output.internal_count,
            "external_count": retrieval_output.external_count,
            "min_external_results": MIN_FALLBACK_RESULTS,
        },
        done=False,
    )


class ProductsGovernor(LoopGovernor):
    def after_turn(
        self,
        *,
        state: _AgentState,
        turn_index: int,
        allow_web_fallback: bool,
        step_tool_history: list[dict[str, Any]],
        previous_query_text: str,
        previous_filters_dump: dict[str, Any] | None,
        skipped_status: str,
    ) -> GovernorAction:
        executed_tool_names = {str(call.get("name") or "") for call in step_tool_history}
        require_retrieval_guard = _maybe_require_tool_before_finalize(
            goal_reached=state.loop_state.goal_reached,
            executed_tool_names=executed_tool_names,
            required_tool_name="find_products",
            tool_name="find_products",
            skipped_status=skipped_status,
        )
        if require_retrieval_guard.action == "continue":
            _append_guard_trace(
                state=state,
                turn_index=turn_index,
                tool_name=require_retrieval_guard.tool_name or "find_products",
                status=require_retrieval_guard.status or skipped_status,
                reason=require_retrieval_guard.reason,
                input_payload=require_retrieval_guard.input_payload or {},
                goal=state.loop_state.goal,
                done=require_retrieval_guard.done,
            )
            state.loop_state.goal_reached = False

        force_guard = _maybe_force_fallback_turn(
            turn_index=turn_index,
            max_turns=state.loop_state.max_turns,
            retrieval_called_this_turn=state.retrieval_called_this_turn,
            allow_web_fallback=allow_web_fallback,
            allow_web_fallback_next_turn=state.compute_allow_web_fallback(turn_index + 1),
            retrieval_output=state.safe_retrieval_output(),
            weather_fetched_this_turn=state.weather_fetched_this_turn,
            goal_reached=state.loop_state.goal_reached,
            tool_name="find_products",
            skipped_status=skipped_status,
        )
        if force_guard.action == "continue":
            _append_guard_trace(
                state=state,
                turn_index=turn_index,
                tool_name=force_guard.tool_name or "find_products",
                status=force_guard.status or skipped_status,
                reason=force_guard.reason,
                input_payload=force_guard.input_payload or {},
                goal=state.loop_state.goal,
                done=force_guard.done,
            )
            state.loop_state.goal_reached = False
            return GovernorAction(action="continue", stop_reason="force_follow_up_fallback_turn")

        current_filters_dump = (
            state.loop_state.current_common_filters.model_dump()
            if state.loop_state.current_common_filters
            else None
        )
        has_existing_results = bool(
            state.last_product_response.internal_results or state.last_product_response.external_results
        )
        no_progress = _maybe_stop_no_progress(
            previous_query_text=previous_query_text,
            current_query_text=state.loop_state.current_query_text,
            previous_filters_dump=previous_filters_dump,
            current_filters_dump=current_filters_dump,
            has_existing_results=has_existing_results,
            tool_name="redundant_fetch_guard",
            skipped_status=skipped_status,
        )
        if no_progress.action == "stop":
            _append_guard_trace(
                state=state,
                turn_index=turn_index,
                tool_name=no_progress.tool_name or "redundant_fetch_guard",
                status=no_progress.status or skipped_status,
                reason=no_progress.reason,
                input_payload=no_progress.input_payload or {},
                goal=state.loop_state.goal,
                done=no_progress.done,
            )
            state.loop_state.mark_stop(stop_reason="no_progress", goal_reached=False)
            return GovernorAction(
                action="stop",
                stop_reason="no_progress",
                mark_stop_reason="no_progress",
            )

        fallback_guard = _maybe_stop_fallback_threshold(
            retrieval_output=state.safe_retrieval_output(),
            tool_name="fallback_stop_guard",
            skipped_status=skipped_status,
        )
        if fallback_guard.action == "stop":
            _append_guard_trace(
                state=state,
                turn_index=turn_index,
                tool_name=fallback_guard.tool_name or "fallback_stop_guard",
                status=fallback_guard.status or skipped_status,
                reason=fallback_guard.reason,
                input_payload=fallback_guard.input_payload or {},
                goal=state.loop_state.goal,
                done=fallback_guard.done,
            )
            state.loop_state.mark_stop(stop_reason="fallback_threshold_met", goal_reached=False)
            return GovernorAction(
                action="stop",
                stop_reason="fallback_threshold_met",
                mark_stop_reason="fallback_threshold_met",
            )

        return GovernorAction(action="none")
