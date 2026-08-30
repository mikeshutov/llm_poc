from __future__ import annotations

import importlib
from uuid import uuid4

from conversation.models.conversation_models import ConversationRoundtrip
from request_orchestrator.shared.runtime_context import bind_runtime_context
from request_orchestrator.shared.tool_adapter.memories.get_memory_detail import get_memory_detail


def test_get_memory_detail_returns_typed_result() -> None:
    class FakeConversationRepo:
        def __init__(self) -> None:
            self.roundtrip = ConversationRoundtrip(
                id=uuid4(),
                conversation_id=uuid4(),
                message_index=7,
                user_prompt="What did we decide about returns?",
                generated_response="We said we would offer 30-day returns.",
                roundtrip_summary="The conversation decided on a 30-day return policy.",
                roundtrip_summary_embedding=None,
                    response_payload={
                        "result": [
                            {
                                "content": "We said we would offer 30-day returns.",
                                "evidence_ids": [],
                            }
                        ]
                    },
                parsed_query={"topic": "returns"},
                created_at="2026-08-01T12:00:00+00:00",
                model="gpt-5.6-terra",
            )

        def get_roundtrip_for_user(self, roundtrip_id, user_id: str | None):
            assert str(roundtrip_id) == str(self.roundtrip.id)
            assert user_id == "user-123"
            return self.roundtrip

    module = importlib.import_module(
        "request_orchestrator.shared.tool_adapter.memories.get_memory_detail"
    )
    fake_repo = FakeConversationRepo()
    original_repo_getter = module.get_conversation_repo
    module.get_conversation_repo = lambda: fake_repo
    try:
        with bind_runtime_context(
            conversation_id="conversation-1",
            conversation_model_config=None,
            roundtrip_id="roundtrip-1",
            user_id="user-123",
        ):
            result = get_memory_detail.invoke({"roundtrip_id": str(fake_repo.roundtrip.id)})
    finally:
        module.get_conversation_repo = original_repo_getter

    assert result.result.memory_type == "roundtrip"
    assert result.result.roundtrip_id == str(fake_repo.roundtrip.id)
    assert result.result.summary == "The conversation decided on a 30-day return policy."
    assert fake_repo.roundtrip.parsed_query == {"topic": "returns"}
    assert result.evidence[0].item_id == str(fake_repo.roundtrip.id)
    assert result.evidence[0].title == "Memory detail for message 7"
    assert result.evidence[0].summary == "The conversation decided on a 30-day return policy."
    assert result.evidence[0].item_id == str(fake_repo.roundtrip.id)


def test_get_memory_detail_rejects_invalid_roundtrip_id() -> None:
    result = get_memory_detail.invoke({"roundtrip_id": "not-a-uuid"})

    assert result.result.error == "Invalid roundtrip_id 'not-a-uuid'."
    assert result.evidence == []
    assert result.evidence == []
