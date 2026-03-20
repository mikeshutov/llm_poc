import threading
from uuid import UUID

from agents.general.agent import run_agent
from agents.general.models.agent_result import AgentResult
from tool.summarize_tool_call import summarize_tool_calls
from conversation.context_builder import build_roundtrip_context
from conversation.models.conversation_models import ConversationRoundtrip
from conversation.repository.repo_factory import get_conversation_repo


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

    # gateway: cache lookup, validation, gating (future) not sure if here or inside the agent execution
    result = run_agent(
        conversation_entries=conversation_entries,
        conversation_id=conversation_id,
        roundtrip_id=str(roundtrip.id),
    )

    repo.update_roundtrip(roundtrip.id, result.raw_response, result.to_payload_for_update_roundtrip())
    #TODO: enable this once we improve summarization.
    #threading.Thread(target=summarize_tool_calls, args=(roundtrip.id,), daemon=True).start()
    return result, roundtrip
