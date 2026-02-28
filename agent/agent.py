from __future__ import annotations

from collections.abc import Callable

from agent.context import AgentContext
from agent.models.agent_result import AgentResult
from agent.models.tool_call_trace import ToolCallTrace
from agent.policy import UnknownHandlingPolicy
from agent.routes.general_info_handler import handle_general_info_route
from agent.routes.products_handler import handle_products_route
from agent.routes.route_name import RouteName
from agent.routes.unsupported_handler import handle_unsupported_route
from agent.runtime.trace_collector import TraceCollector
from agent.state import _AgentState
from intent_layer.models.parsed_request import ParsedRequest
from llm.clients.llm_client import LlmClient


def run_agent(
    conversation_entries: list[dict],
    parsed_query: ParsedRequest,
    conversation_id: str,
    max_turns: int = 10,
    on_trace: Callable[[ToolCallTrace], None] | None = None,
    *,
    llm: LlmClient | None = None,
    policy: UnknownHandlingPolicy | None = None,
) -> AgentResult:
    policy = policy or UnknownHandlingPolicy()
    llm = llm or LlmClient()
    decision = policy.decide(parsed_query)

    state = _AgentState.new(parsed_query=parsed_query, max_turns=max_turns)
    state.tool_traces = TraceCollector(on_trace=on_trace)

    ctx = AgentContext(
        conversation_entries=conversation_entries,
        parsed_query=parsed_query,
        conversation_id=conversation_id,
        max_turns=max_turns,
        decision=decision,
        llm=llm,
        state=state,
    )

    handlers: dict[RouteName, Callable[[AgentContext], AgentResult]] = {
        RouteName.UNSUPPORTED: handle_unsupported_route,
        RouteName.GENERAL_INFO: handle_general_info_route,
        RouteName.PRODUCTS: handle_products_route,
    }
    return handlers.get(ctx.decision.route, handle_unsupported_route)(ctx=ctx)
