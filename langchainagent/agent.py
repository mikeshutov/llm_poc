from __future__ import annotations

from collections.abc import Callable

from langsmith import traceable

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
from langchainagent.executor.executor import run_executor
from langchainagent.planner.planner import run_planner
from langchainagent.synthesis.synthesis import run_synthesis
from langchainagent.agentstate.model import AgentState
from llm.clients.llm_client import LlmClient
from langgraph.graph import MessagesState, START, StateGraph, END
from response_layer.response_layer import generate_response


def flatten_messages(messages: list[dict], limit: int = 12) -> str:
    return "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in messages[-limit:]
        if m.get("content")
    )


def router(state: AgentState) -> str:
    # Stop if goal reached
    if state.goal_reached:
        return "solve"
    # Stop if max iterations reached
    if len(state.iteration_trace) >= state.max_turns:
        return "solve"
    # Otherwise continue planning
    return "plan"

@traceable(name="LangChain Agent Node")
def run_langchain_agent(
    conversation_entries: list[dict],
    parsed_query: ParsedRequest,
    conversation_id: str,
    max_turns: int = 10,
    on_trace: Callable[[ToolCallTrace], None] | None = None,
    *,
    llm: LlmClient | None = None,
    policy: UnknownHandlingPolicy | None = None,
) -> AgentResult:
    agentState = AgentState.new(flatten_messages(conversation_entries),max_turns)
    builder = StateGraph(AgentState)
    builder.add_node("plan", run_planner)
    builder.add_node("tool", run_executor)
    builder.add_node("solve", run_synthesis)
    builder.set_entry_point("plan")
    builder.add_edge("plan", "tool")
    builder.add_conditional_edges(
        "tool",
        router,
        {
            "plan": "plan",
            "solve": "solve",
        },
    )
    builder.add_edge("solve", END)
    agent_graph = builder.compile()

    # # create a graph to see what our chain looks like
    # png = agent_graph.get_graph(xray=1).draw_mermaid_png(
    #     background_color="white"
    # )

    # with open("graph.png", "wb") as f:
    #     f.write(png)

    final_state = agent_graph.invoke(
        agentState,
        config={"configurable": {"thread_id": conversation_id}},
    )

    result_text = final_state.result if isinstance(final_state, AgentState) else final_state.get("result", "")
    answer = generate_response(
        conversation_entries=conversation_entries,
        query_results=result_text,
        conversation_id=conversation_id,
    )

    return AgentResult(
        answer=answer,
        debug_trace={},
        tool_traces=[],
        goal=agentState.task,
        goal_reached=agentState.goal_reached,
        iterations_used=len(agentState.iteration_trace),
    )
