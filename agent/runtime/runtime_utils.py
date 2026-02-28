from dataclasses import asdict
from typing import Any

from agent.models.tool_call_trace import ToolCallTrace
from agent.policy import RouteDecision
from agent.tools.tool_name import ToolName
from products.models.product_search_results import ProductSearchResults
from response_layer.cards_mapper import product_results_to_cards
from response_layer.models.response_payload import ResponsePayload


def safe_fallback(product_response: ProductSearchResults) -> ResponsePayload:
    cards = product_results_to_cards(product_response, limit=10)
    return ResponsePayload(
        response="I had trouble finalizing product recommendations, but here are the closest matches I found.",
        cards=cards,
        follow_up="Want me to narrow these by price, color, or style?",
    )


def unsupported_response() -> ResponsePayload:
    return ResponsePayload(
        response=(
            "I can't help with that yet. I can currently help with product search and web/news information lookups."
        ),
        cards=[],
        follow_up="Try: 'Find black running shoes under $100' or 'latest trail running news'.",
    )


def append_trace(
    traces: list[ToolCallTrace],
    call_index: int,
    turn_index: int,
    tool_name: str,
    status: str,
    reason: str,
    input_payload: dict[str, Any],
    output_payload: dict[str, Any] | None = None,
    error_message: str | None = None,
    goal: str | None = None,
    done: bool | None = None,
    duration_ms: int | None = None,
) -> int:
    traces.append(
        ToolCallTrace(
            call_index=call_index,
            turn_index=turn_index,
            tool_name=tool_name,
            status=status,
            reason=reason,
            input_payload=input_payload,
            output_payload=output_payload,
            error_message=error_message,
            duration_ms=duration_ms,
            goal=goal,
            done=done,
        )
    )
    return call_index + 1


def build_debug_trace(
    decision: RouteDecision,
    parsed_intent: str,
    max_turns: int,
    iterations_used: int,
    goal: str,
    goal_reached: bool,
    traces: list[ToolCallTrace],
) -> dict[str, Any]:
    return {
        "agent": "generic_agent",
        "intent": parsed_intent,
        "route": decision.route.value,
        "supported": decision.supported,
        "reason": decision.reason,
        "allowed_tools": sorted(tool.value for tool in decision.allowed_tools),
        "goal": goal,
        "goal_reached": goal_reached,
        "max_turns": max_turns,
        "iterations_used": iterations_used,
        "turns": [
            {
                "turn_index": t.turn_index,
                "goal": t.goal,
                "done": t.done,
                "planner_status": t.status,
                "planner_reason": t.reason,
                "planner_error": t.error_message,
                "planner_output": t.output_payload,
                "retrieval_counts": t.input_payload.get("retrieval_counts", {}),
                "query_text": (
                    t.input_payload.get("current_refined_query_text")
                    or t.input_payload.get("current_query_text", "")
                ),
                "common_filters": t.input_payload.get("current_common_filters"),
            }
            for t in traces
            if t.tool_name in {ToolName.PLAN_AGENT_STEP.value, "planner_decision"}
        ],
        "tool_traces": [asdict(t) for t in traces],
    }
