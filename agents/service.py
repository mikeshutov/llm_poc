from __future__ import annotations

from uuid import UUID

from langsmith import traceable
from langgraph.graph import END, StateGraph

from agents.agentstate.model import AgentState
from agents.classify.classify import classify
from agents.graph_constants import CLASSIFICATION_EDGE, GENERAL_AGENT_EDGE, SYNTHESIZE_EDGE
from agents.impl.general.agent import general_agent_graph
from agents.impl.general.router.router import router
from agents.models.agent_result import AgentResult
from agents.synthesis.synthesis import run_synthesis
from conversation.context_builder import build_roundtrip_context
from conversation.models.conversation_models import ConversationRoundtrip
from conversation.repository.repo_factory import get_conversation_repo
from conversation.utils import flatten_conversation_entries


def _build_main_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node(CLASSIFICATION_EDGE, classify)
    builder.add_node(GENERAL_AGENT_EDGE, general_agent_graph())
    builder.add_node(SYNTHESIZE_EDGE, run_synthesis)
    builder.set_entry_point(CLASSIFICATION_EDGE)

    builder.add_conditional_edges(
        CLASSIFICATION_EDGE,
        router,
        {
            SYNTHESIZE_EDGE: SYNTHESIZE_EDGE,
            GENERAL_AGENT_EDGE: GENERAL_AGENT_EDGE,
        },
    )

    builder.add_edge(GENERAL_AGENT_EDGE, SYNTHESIZE_EDGE)
    builder.add_edge(SYNTHESIZE_EDGE, END)
    return builder.compile()


@traceable(name="Agents Entry Point")
def run_agent_for_query(
    conversation_id: str,
    user_query: str,
    context_limit: int = 5,
) -> tuple[AgentResult, ConversationRoundtrip]:
    repo = get_conversation_repo()
    roundtrip = repo.create_pending_roundtrip(UUID(conversation_id), user_query)

    conversation_entries = build_roundtrip_context(
        conversation_id,
        user_query,
        limit=context_limit,
    )

    agent_state = AgentState.new(
        task=flatten_conversation_entries(conversation_entries),
        conversation_entries=conversation_entries,
        roundtrip_id=roundtrip.id,
    )

    final_state = _build_main_graph().invoke(
        agent_state,
        config={"configurable": {"thread_id": conversation_id}},
    )

    final = final_state if isinstance(final_state, AgentState) else AgentState(**final_state)
    if final.result is None:
        raise ValueError("Agent finished without setting result")

    result = final.result
    repo.update_roundtrip(roundtrip.id, result.raw_response, result.to_payload_for_update_roundtrip())
    # TODO: enable once summarization is improved
    # threading.Thread(target=summarize_tool_calls, args=(roundtrip.id,), daemon=True).start()
    return result, roundtrip
