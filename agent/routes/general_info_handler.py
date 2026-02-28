import json

from agent.context import AgentContext
from agent.models.agent_result import AgentResult
from agent.models.tool_call_status import ToolCallStatus
from agent.routes.finalize_result import finalize_result
from agent.runtime.agent_loop_runner import run_agent_loop
from agent.runtime.route_kind import RouteKind
from agent.runtime.trace_service import trace_and_advance
from agent.tools.tool_name import ToolName
from common.time import now_ms
from response_layer.cards_mapper import news_results_to_cards
from response_layer.models.response_payload import ResponsePayload
from response_layer.response_layer import generate_response
from websearch.models.search_type import SearchType

GENERAL_INFO_GOAL = "Answer general information request"


def handle_general_info_route(
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
        route_kind=RouteKind.GENERAL_INFO,
    )
    if not state.loop_state.goal:
        state.loop_state.goal = GENERAL_INFO_GOAL
    turn_index = max(0, state.iterations_used - 1)
    search_payload = state.last_generic_search_payload or {}

    response_started = now_ms()
    state.set_response_status(ToolCallStatus.SUCCESS.value, None)
    try:
        answer = generate_response(
            conversation_entries=ctx.conversation_entries,
            query_results=json.dumps(search_payload),
            conversation_id=ctx.conversation_id,
        )
    except Exception as exc:
        state.set_response_status(ToolCallStatus.ERROR.value, str(exc))
        answer = ResponsePayload(
            response="I had trouble finalizing that information request.",
            cards=[],
            follow_up="Try narrowing the topic or timeframe.",
        )

    if (
        ctx.parsed_query.query_details.search_type == SearchType.NEWS_SEARCH
        and state.response_status == ToolCallStatus.SUCCESS.value
        and not answer.cards
    ):
        news_cards = news_results_to_cards(search_payload, limit=5)
        if news_cards:
            answer = ResponsePayload(
                response=answer.response,
                cards=news_cards,
                follow_up=answer.follow_up,
            )

    trace_and_advance(
        state,
        turn_index=turn_index,
        tool_name=ToolName.GENERATE_RESPONSE.value,
        status=state.response_status or ToolCallStatus.ERROR.value,
        reason="Compose final response for general information route.",
        input_payload={"conversation_id": ctx.conversation_id},
        output_payload={"has_cards": bool(answer.cards)},
        error_message=state.response_error,
        goal=GENERAL_INFO_GOAL,
        done=True,
        duration_ms=now_ms() - response_started,
    )

    return finalize_result(
        ctx=ctx,
        answer=answer,
        goal=GENERAL_INFO_GOAL,
    )
