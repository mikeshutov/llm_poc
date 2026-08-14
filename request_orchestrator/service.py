import threading
from time import perf_counter
from uuid import UUID

from integrations.ip_api import IpApiClient
from request_orchestrator.models.agent_state import GeoLocation, GeoMetadata, build_geometadata
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.main_state import MainState
from request_orchestrator.orchestrator import run_agent
from request_orchestrator.shared.runtime_context import bind_runtime_context
from llm.clients.embeddings import embed_text
from tool.summarize_tool_call import summarize_tool_calls
from conversation.context_builder import build_roundtrip_context
from personalization.profile.service import build_user_profile
from conversation.models.conversation_models import ConversationRoundtrip
from conversation.repository.repo_factory import get_conversation_repo

_ip_api_client = IpApiClient()


def _resolve_geometadata(geometadata: GeoMetadata | None) -> GeoMetadata:
    if geometadata is not None:
        return geometadata

    try:
        location = _ip_api_client.get_location()
        return build_geometadata(
            timezone=location.timezone,
            location=GeoLocation(
                city=location.city,
                region=location.region_name or location.region,
                country=location.country,
                latitude=location.lat,
                longitude=location.lon,
                timezone=location.timezone,
            ),
        )
    except Exception:
        return build_geometadata()


def run_request_orchestrator_for_query(
    conversation_id: str,
    user_query: str,
    user_id: str | None = None,
    context_limit: int = 5,
    geometadata: GeoMetadata | None = None,
) -> tuple[AgentResult, ConversationRoundtrip]:
    started_at = perf_counter()
    repo = get_conversation_repo()
    conversation = repo.get_conversation(UUID(conversation_id))
    if conversation is None:
        raise ValueError(f"Conversation not found: {conversation_id}")

    resolved_user_id = user_id.strip() if isinstance(user_id, str) else None
    if not resolved_user_id:
        raise ValueError("user_id is required")

    conversation_user_id = conversation.user_id.strip() if isinstance(conversation.user_id, str) else conversation.user_id
    if resolved_user_id != conversation_user_id:
        raise ValueError(
            f"Conversation {conversation_id} belongs to user {conversation.user_id}, not {resolved_user_id}"
        )

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

    resolved_geometadata = _resolve_geometadata(geometadata)
    user_profile = build_user_profile(
        user_id=resolved_user_id,
        geometadata=resolved_geometadata,
    )
    main_state = MainState.new(
        task=user_query,
        max_turns=10,
        conversation_context=conversation_context,
        user_profile=user_profile,
        conversation_id=conversation_id,
        roundtrip_id=roundtrip.id,
        conversation_model_config=resolved_model_config,
    )

    with bind_runtime_context(
        conversation_id=conversation_id,
        conversation_model_config=resolved_model_config,
        roundtrip_id=str(roundtrip.id),
        user_id=user_profile.user_id,
    ):
        result = run_agent(main_state)

    roundtrip_latency_ms = int((perf_counter() - started_at) * 1000)
    if isinstance(result, AgentResult):
        result = result.with_roundtrip_latency(roundtrip_latency_ms)
    payload = result.to_payload()

    roundtrip_summary_embedding = embed_text(result.roundtrip_summary) if result.roundtrip_summary else None
    roundtrip = repo.update_roundtrip(
        roundtrip.id,
        result.raw_response,
        payload,
        roundtrip_summary=result.roundtrip_summary,
        roundtrip_summary_embedding=roundtrip_summary_embedding,
    )
    #TODO: enable this once we improve summarization.
    #threading.Thread(target=summarize_tool_calls, args=(roundtrip.id,), daemon=True).start()
    return result, roundtrip
