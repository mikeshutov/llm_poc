import json
import re
from dataclasses import dataclass
from typing import Any

from agent.models.tool_call_status import ToolCallStatus
from agent.prompts.agent_planner_prompt import AGENT_PLANNER_PROMPT
from agent.prompts.general_info_planner_prompt import GENERAL_INFO_PLANNER_PROMPT
from agent.runtime.governors import governor_for
from agent.runtime.planner_policy import ordered_tool_calls
from agent.runtime.planner_trace import build_tool_trace
from agent.runtime.route_kind import RouteKind
from agent.runtime.tool_dispatcher import ToolDispatcher
from agent.state import _AgentState
from agent.tools.agent_planner_tool import (
    PLANNER_FIND_PRODUCTS_TOOL,
    PLANNER_GENERIC_WEB_SEARCH_TOOL,
    PLANNER_GET_HISTORICAL_MONTH_WEATHER_TOOL,
    PLANNER_RESOLVE_CITY_LOCATION_TOOL,
    PRODUCT_CATEGORY_LOOKUP_TOOL,
)
from agent.tools.tool_name import ToolName
from common.message_constants import CONTENT_KEY, ROLE_KEY, ROLE_SYSTEM
from intent_layer.models.parsed_request import ParsedRequest
from llm.clients.llm_client import LlmClient
from websearch.models.search_type import SearchType


@dataclass(frozen=True)
class AgentLoopResult:
    final_decision_args: dict[str, Any] | None
    executed_tool_calls: list[dict[str, Any]]
    iteration_summaries: list[dict[str, Any]]
    planner_status: str
    planner_error: str | None
    next_call_index: int
    stop_reason: str | None


def _default_decision(state: _AgentState) -> dict[str, Any]:
    return {
        "goal": state.loop_state.goal,
        "done": False,
    }


def _message_text(raw_message: Any) -> str:
    content = getattr(raw_message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        try:
            value = json.loads(fenced.group(1))
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    return None


def _extract_final_decision(raw_message: Any) -> dict[str, Any] | None:
    payload = _parse_json_object(_message_text(raw_message))
    if not isinstance(payload, dict):
        return None
    if not any(key in payload for key in ("done", "goal")):
        return None
    return payload


def _planner_config(
    *,
    route_kind: RouteKind,
    state: _AgentState,
    parsed_query: ParsedRequest,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    if route_kind == RouteKind.GENERAL_INFO:
        query_text = parsed_query.query_details.query_text if parsed_query.query_details else ""
        search_type = (
            parsed_query.query_details.search_type.value
            if parsed_query.query_details and parsed_query.query_details.search_type
            else SearchType.WEB_SEARCH.value
        )
        return (
            GENERAL_INFO_PLANNER_PROMPT,
            [PLANNER_GENERIC_WEB_SEARCH_TOOL],
            {
                "goal": "Answer general information request",
                "query_text": query_text,
                "search_type": search_type,
                "previous_tool_calls": state.last_planner_tool_calls,
                "decision_schema": {
                    "goal": "string",
                    "done": "boolean",
                    "query_refinement_reason": "string?",
                },
            },
        )

    product_response = state.safe_product_response()
    retrieval_output = state.safe_retrieval_output()
    return (
        AGENT_PLANNER_PROMPT,
        [
            PRODUCT_CATEGORY_LOOKUP_TOOL,
            PLANNER_FIND_PRODUCTS_TOOL,
            PLANNER_RESOLVE_CITY_LOCATION_TOOL,
            PLANNER_GET_HISTORICAL_MONTH_WEATHER_TOOL,
        ],
        {
            "turn_index": state.loop_state.turn_index,
            "max_turns": state.loop_state.max_turns,
            "original_query_text": state.loop_state.original_query_text,
            "current_refined_query_text": state.loop_state.refined_query_text,
            "query_refinement_history": [
                {"label": "original", "query_text": state.loop_state.original_query_text},
                {"label": "current_refined", "query_text": state.loop_state.refined_query_text},
            ],
            "current_common_filters": (
                state.loop_state.current_common_filters.model_dump()
                if state.loop_state.current_common_filters
                else None
            ),
            "current_goal": state.loop_state.goal,
            "weather_context": state.last_weather_context,
            "last_discovery_tools": state.last_discovery_tools,
            "retrieval_summary": {
                "source_priority": {"primary": "internal_catalog", "fallback": "web_search"},
                "web_fallback_condition": "internal_count == 0",
                "fallback_stop_policy": {
                    "enabled": True,
                    "min_external_results": 3,
                    "condition": "internal_count == 0",
                },
                "redundant_fetch_guard_enabled": True,
                "web_fallback_used": (
                    retrieval_output.internal_count == 0 and retrieval_output.external_count > 0
                ),
                "internal_count": retrieval_output.internal_count,
                "external_count": retrieval_output.external_count,
                "internal_sample": [
                    {"id": p.id, "name": p.name, "price": p.price, "source": str(p.source)}
                    for p in product_response.internal_results[:5]
                ],
                "external_sample": [
                    {"id": p.id, "name": p.name, "price": p.price, "source": str(p.source)}
                    for p in product_response.external_results[:5]
                ],
            },
            "previous_tool_calls": state.last_planner_tool_calls,
            "decision_schema": {
                "goal": "string",
                "done": "boolean",
                "query_refinement_reason": "string?",
            },
        },
    )


def build_step_messages(
    *,
    conversation_entries: list[dict[str, Any]],
    step_input: dict[str, Any],
    previous_tool_calls: list[dict[str, Any]],
    turn_index: int,
) -> list[dict[str, Any]]:
    return [
        *conversation_entries,
        {
            ROLE_KEY: ROLE_SYSTEM,
            CONTENT_KEY: json.dumps(
                {
                    "step_input": step_input,
                    "previous_tool_calls": previous_tool_calls,
                    "turn_index": turn_index,
                },
                default=str,
            ),
        },
    ]


def plan_step(
    *,
    llm: LlmClient,
    step_prompt: str,
    step_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[Any, list[Any]]:
    model_result = llm.call_with_tools(
        step_prompt,
        step_messages,
        tools=tools,
        temperature=0.0,
    )
    ordered_calls = ordered_tool_calls(model_result.tool_calls or [])
    return model_result, ordered_calls


def execute_tools(
    *,
    state: _AgentState,
    dispatcher: ToolDispatcher,
    ordered_calls: list[Any],
    turn_index: int,
) -> list[dict[str, Any]]:
    step_tool_history: list[dict[str, Any]] = []
    for tool_call in ordered_calls:
        name = tool_call.name
        args = tool_call.args if isinstance(tool_call.args, dict) else {}

        result = dispatcher.dispatch(name, args)
        trace = build_tool_trace(
            call_index=state.loop_state.call_index,
            turn_index=turn_index,
            tool_name=name,
            status=result.status,
            reason="Agent requested tool execution.",
            input_payload=args,
            output_payload=result.output_payload,
            error_message=result.error_message,
            duration_ms=result.duration_ms,
            goal=state.loop_state.goal,
            done=None,
        )
        state.add_traces([trace])
        state.loop_state.call_index += 1

        executed_entry = {
            "name": name,
            "args": args,
            "status": result.status,
            "output": result.output_payload,
            "error": result.error_message,
            "duration_ms": result.duration_ms,
            "call_index": trace.call_index,
            "turn_index": trace.turn_index,
        }
        step_tool_history.append(executed_entry)
        state.apply_tool_execution(
            name=name,
            args=args,
            status=result.status,
            output=result.output_payload,
            error=result.error_message,
            turn_index=trace.turn_index,
            call_index=trace.call_index,
            raw_result=result.raw_result,
        )
    return step_tool_history


def compact_tool_history(step_tool_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for call in step_tool_history:
        compact.append(
            {
                "name": call.get("name"),
                "status": call.get("status"),
                "call_index": call.get("call_index"),
                "turn_index": call.get("turn_index"),
            }
        )
    return compact


def run_agent_loop(
    *,
    llm: LlmClient,
    conversation_entries: list[dict[str, Any]],
    parsed_query: ParsedRequest,
    state: _AgentState,
    decision_allowed_tools: set[ToolName],
    route_kind: RouteKind,
) -> AgentLoopResult:
    all_executed_tool_calls: list[dict[str, Any]] = []
    final_decision_args: dict[str, Any] | None = None
    planner_status = ToolCallStatus.SUCCESS.value
    planner_error: str | None = None
    stop_reason: str | None = None
    governor = governor_for(route_kind)

    for turn_index in range(state.loop_state.max_turns):
        state.begin_turn(turn_index)
        previous_query_text = state.loop_state.current_query_text
        previous_filters_dump = (
            state.loop_state.current_common_filters.model_dump()
            if state.loop_state.current_common_filters
            else None
        )
        allow_web_fallback = state.compute_allow_web_fallback(turn_index)
        step_prompt, tools, step_input = _planner_config(
            route_kind=route_kind,
            state=state,
            parsed_query=parsed_query,
        )
        step_tool_history: list[dict[str, Any]] = []
        final_decision_args = None
        dispatcher = ToolDispatcher(
            allowed_tools=decision_allowed_tools,
            state=state,
            allow_web_fallback=allow_web_fallback,
        )
        step_messages = build_step_messages(
            conversation_entries=conversation_entries,
            step_input=step_input,
            previous_tool_calls=state.last_planner_tool_calls,
            turn_index=turn_index,
        )
        planner_request_payload = {
            "system_prompt": step_prompt,
            "messages": step_messages,
            "tools": tools,
            "temperature": 0.0,
        }
        model_result, ordered_calls = plan_step(
            llm=llm,
            step_prompt=step_prompt,
            step_messages=step_messages,
            tools=tools,
        )
        step_tool_history = execute_tools(
            state=state,
            dispatcher=dispatcher,
            ordered_calls=ordered_calls,
            turn_index=turn_index,
        )
        all_executed_tool_calls.extend(step_tool_history)

        # Resolve final decision for this turn from assistant JSON or defaults.
        parsed_decision = _extract_final_decision(model_result.raw_message)
        decision_trace_reason: str
        decision_trace_status: str
        decision_trace_error: str | None = None
        if parsed_decision is not None:
            planner_status = ToolCallStatus.SUCCESS.value
            planner_error = None
            final_decision_args = parsed_decision
            decision_trace_reason = "Parsed decision JSON from assistant message."
            decision_trace_status = ToolCallStatus.SUCCESS.value
        elif ordered_calls:
            planner_status = ToolCallStatus.SUCCESS.value
            planner_error = None
            final_decision_args = _default_decision(state)
            decision_trace_reason = "No decision JSON; defaulted decision after tool execution."
            decision_trace_status = ToolCallStatus.SUCCESS.value
        else:
            planner_status = ToolCallStatus.ERROR.value
            planner_error = "Model returned neither tool calls nor a valid decision JSON object."
            final_decision_args = _default_decision(state)
            decision_trace_reason = "Model did not produce tool calls or valid decision JSON; defaulted decision."
            decision_trace_status = ToolCallStatus.ERROR.value
            decision_trace_error = planner_error

        state.last_planner_tool_calls = compact_tool_history(step_tool_history)
        state.apply_decision(final_decision_args)

        state.add_traces(
            [
                build_tool_trace(
                    call_index=state.loop_state.call_index,
                    turn_index=turn_index,
                    tool_name="planner_decision",
                    status=decision_trace_status,
                    reason=decision_trace_reason,
                    input_payload={
                        "turn_index": turn_index,
                        "planner_request": planner_request_payload,
                        "tool_calls_count": len(ordered_calls),
                        "tool_calls_requested": [{"name": call.name, "args": call.args} for call in ordered_calls],
                        "tool_calls_executed": step_tool_history,
                        "goal_requested": final_decision_args.get("goal"),
                        "done_requested": final_decision_args.get("done"),
                    },
                    output_payload=final_decision_args,
                    error_message=decision_trace_error,
                    duration_ms=0,
                    goal=state.loop_state.goal,
                    done=state.loop_state.goal_reached,
                )
            ]
        )
        state.loop_state.call_index += 1

        # Apply route governor after decision projection.
        governor_action = governor.after_turn(
            state=state,
            turn_index=turn_index,
            allow_web_fallback=allow_web_fallback,
            step_tool_history=step_tool_history,
            previous_query_text=previous_query_text,
            previous_filters_dump=previous_filters_dump,
            skipped_status=ToolCallStatus.SKIPPED.value,
        )

        # Determine turn control in one place.
        turn_action = "none"
        turn_stop_reason: str | None = None
        turn_mark_stop_reason: str | None = None
        if planner_status == ToolCallStatus.ERROR.value:
            turn_action = "stop"
            turn_stop_reason = "planner_error"
            turn_mark_stop_reason = "planner_error"
        elif state.loop_state.goal_reached:
            turn_action = "stop"
            turn_stop_reason = "goal_reached"
        elif governor_action.action == "continue":
            turn_action = "continue"
            turn_stop_reason = governor_action.stop_reason
        elif governor_action.action == "stop":
            turn_action = "stop"
            turn_stop_reason = governor_action.stop_reason
            turn_mark_stop_reason = governor_action.mark_stop_reason

        state.add_iteration_summary(
            {
                "turn_index": turn_index,
                "planner_status": planner_status,
                "planner_error": planner_error,
                "goal": state.loop_state.goal,
                "goal_reached": state.loop_state.goal_reached,
                "executed_calls": len(step_tool_history),
                "stop_reason": turn_stop_reason,
            }
        )

        if turn_action == "continue":
            continue

        if turn_action == "stop":
            if turn_mark_stop_reason and state.loop_state.turn_stop_reason is None:
                state.loop_state.mark_stop(
                    stop_reason=turn_mark_stop_reason,
                    goal_reached=state.loop_state.goal_reached,
                )
            stop_reason = turn_stop_reason
            break

    if stop_reason is None:
        stop_reason = "max_turns_reached"

    return AgentLoopResult(
        final_decision_args=final_decision_args,
        executed_tool_calls=all_executed_tool_calls,
        iteration_summaries=state.iteration_history,
        planner_status=planner_status,
        planner_error=planner_error,
        next_call_index=state.loop_state.call_index,
        stop_reason=stop_reason,
    )
