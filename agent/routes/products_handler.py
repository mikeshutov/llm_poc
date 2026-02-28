import json
from typing import Any

from agent.context import AgentContext
from agent.models.agent_result import AgentResult
from agent.models.tool_call_status import ToolCallStatus
from agent.routes.finalize_result import finalize_result
from agent.runtime.agent_loop_runner import run_agent_loop
from agent.runtime.route_kind import RouteKind
from agent.runtime.runtime_utils import safe_fallback
from agent.runtime.trace_service import trace_and_advance
from agent.tools.tool_name import ToolName
from common.time import now_ms
from response_layer.cards_mapper import product_results_to_cards
from response_layer.models.response_payload import ResponsePayload
from response_layer.response_layer import generate_response


def handle_products_route(
    *,
    ctx: AgentContext,
) -> AgentResult:
    state = ctx.state
    run_agent_loop(
        llm=ctx.llm,
        conversation_entries=ctx.conversation_entries,
        parsed_query=ctx.parsed_query,
        state=state,
        decision_allowed_tools=set(ctx.decision.allowed_tools),
        route_kind=RouteKind.PRODUCTS,
    )

    response_started = now_ms()
    last_product_response = state.safe_product_response()
    response_input = {
        "conversation_id": ctx.conversation_id,
        "query_results_preview": {
            "internal_count": len(last_product_response.internal_results),
            "external_count": len(last_product_response.external_results),
        },
        "weather_context_present": bool(state.last_weather_context),
        "goal": state.loop_state.goal,
        "goal_reached": state.loop_state.goal_reached,
        "turn_stop_reason": state.loop_state.turn_stop_reason,
    }
    query_results_payload = json.dumps(
        {
            "internal_results": last_product_response.internal_results,
            "external_results": last_product_response.external_results,
            "weather_context": state.last_weather_context,
            "turn_stop_reason": state.loop_state.turn_stop_reason,
        },
        default=str,
    )

    state.set_response_status(ToolCallStatus.SUCCESS.value, None)
    try:
        answer = generate_response(
            conversation_entries=ctx.conversation_entries,
            query_results=query_results_payload,
            conversation_id=ctx.conversation_id,
        )
        response_output: dict[str, Any] = {
            "has_cards": bool(answer.cards),
            "follow_up_present": bool(answer.follow_up),
        }
    except Exception as exc:
        state.set_response_status(ToolCallStatus.ERROR.value, str(exc))
        answer = safe_fallback(last_product_response)
        response_output = {
            "has_cards": bool(answer.cards),
            "fallback_used": True,
        }

    trace_and_advance(
        state,
        turn_index=max(0, state.iterations_used - 1),
        tool_name=ToolName.GENERATE_RESPONSE.value,
        status=state.response_status or ToolCallStatus.ERROR.value,
        reason="Compose the final user-facing answer from the final retrieval set.",
        input_payload=response_input,
        output_payload=response_output,
        error_message=state.response_error,
        goal=state.loop_state.goal,
        done=state.loop_state.goal_reached,
        duration_ms=now_ms() - response_started,
    )

    if not answer.cards:
        mapped_cards = product_results_to_cards(last_product_response, limit=10)
        answer = ResponsePayload(
            response=answer.response,
            cards=mapped_cards,
            follow_up=answer.follow_up,
        )

    return finalize_result(
        ctx=ctx,
        answer=answer,
        goal=state.loop_state.goal,
    )
