import threading
from uuid import UUID

from request_orchestrator.agents.main_agent.agent import run_agent
from request_orchestrator.models.agent_state import GeoMetadata, build_geometadata
from request_orchestrator.models.agent_result import AgentResult
from llm.clients.embeddings import embed_text
from tool.summarize_tool_call import summarize_tool_calls
from conversation.context_builder import build_roundtrip_context
from personalization.profile.service import build_user_profile
from conversation.models.conversation_models import ConversationRoundtrip
from conversation.repository.repo_factory import get_conversation_repo


def run_request_orchestrator_for_query(
    conversation_id: str,
    user_query: str,
    context_limit: int = 5,
    geometadata: GeoMetadata | None = None,
) -> tuple[AgentResult, ConversationRoundtrip]:
    repo = get_conversation_repo()
    resolved_model_config = repo.resolve_conversation_model_config(UUID(conversation_id))
    roundtrip = repo.create_pending_roundtrip(
        UUID(conversation_id),
        user_query,
        model=resolved_model_config.main_agent.planner,
        metadata={"resolved_model_config": resolved_model_config.to_metadata_payload()},
    )

    conversation_context = build_roundtrip_context(
        conversation_id,
        limit=context_limit,
    )

    resolved_geometadata = build_geometadata() if geometadata is None else geometadata
    user_profile = build_user_profile(geometadata=resolved_geometadata)

    result = run_agent(
        conversation_context=conversation_context,
        user_query=user_query,
        conversation_id=conversation_id,
        roundtrip_id=str(roundtrip.id),
        user_profile=user_profile,
        conversation_model_config=resolved_model_config,
    )

    roundtrip_summary_embedding = embed_text(result.roundtrip_summary) if result.roundtrip_summary else None
    roundtrip = repo.update_roundtrip(
        roundtrip.id,
        result.raw_response,
        result.to_payload_for_update_roundtrip(),
        roundtrip_summary=result.roundtrip_summary,
        roundtrip_summary_embedding=roundtrip_summary_embedding,
    )
    #TODO: enable this once we improve summarization.
    #threading.Thread(target=summarize_tool_calls, args=(roundtrip.id,), daemon=True).start()
    return result, roundtrip
