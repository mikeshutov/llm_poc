import threading
from uuid import UUID

from agent.agent import run_agent
from agent.agentstate.model import GeoMetadata
from agent.models.agent_result import AgentResult
from common.model_constants import LLM_MODEL
from llm.clients.embeddings import embed_text
from tool.summarize_tool_call import summarize_tool_calls
from conversation.context_builder import build_roundtrip_context
from conversation.models.conversation_models import ConversationRoundtrip
from conversation.repository.repo_factory import get_conversation_repo


def run_agent_for_query(
    conversation_id: str,
    user_query: str,
    context_limit: int = 5,
    geometadata: GeoMetadata | None = None,
) -> tuple[AgentResult, ConversationRoundtrip]:
    repo = get_conversation_repo()
    roundtrip = repo.create_pending_roundtrip(UUID(conversation_id), user_query, model=LLM_MODEL)

    conversation_context = build_roundtrip_context(
        conversation_id,
        limit=context_limit,
    )

    result = run_agent(
        conversation_context=conversation_context,
        user_query=user_query,
        conversation_id=conversation_id,
        roundtrip_id=str(roundtrip.id),
        geometadata=geometadata,
    )

    roundtrip_summary_embedding = embed_text(result.roundtrip_summary) if result.roundtrip_summary else None
    repo.update_roundtrip(
        roundtrip.id,
        result.raw_response,
        result.to_payload_for_update_roundtrip(),
        roundtrip_summary=result.roundtrip_summary,
        roundtrip_summary_embedding=roundtrip_summary_embedding,
    )
    #TODO: enable this once we improve summarization.
    #threading.Thread(target=summarize_tool_calls, args=(roundtrip.id,), daemon=True).start()
    roundtrip.generated_response = result.raw_response
    roundtrip.roundtrip_summary = result.roundtrip_summary or None
    roundtrip.roundtrip_summary_embedding = roundtrip_summary_embedding
    return result, roundtrip
