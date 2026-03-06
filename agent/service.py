from conversation.context_builder import build_roundtrip_context
from agent.agent import run_agent
from agent.models.agent_result import AgentResult


def run_agent_for_query(
    conversation_id: str,
    user_query: str,
    context_limit: int = 5,
) -> AgentResult:
    conversation_entries = build_roundtrip_context(
        conversation_id,
        user_query,
        limit=context_limit,
    )

    # gateway: cache lookup, validation, gating (future)

    return run_agent(
        conversation_entries=conversation_entries,
        conversation_id=conversation_id,
    )
