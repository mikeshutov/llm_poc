from __future__ import annotations

import importlib

from request_orchestrator.models.evidence import EvidenceView
from request_orchestrator.shared.runtime_context import bind_runtime_context
from request_orchestrator.shared.tool_adapter.memories.lookup_evidence import lookup_evidence


def test_lookup_evidence_returns_a_batch_in_request_order() -> None:
    first = EvidenceView(title="First evidence", summary="First summary")
    second = EvidenceView(title="Second evidence", summary="Second summary")

    class FakeConversationRepo:
        def get_evidence_by_ids_for_user(self, evidence_ids, user_id):
            assert evidence_ids == ["second", "missing", "first"]
            assert user_id == "user-123"
            return {"first": first, "second": second}

    module = importlib.import_module(
        "request_orchestrator.shared.tool_adapter.memories.lookup_evidence"
    )
    original_repo_getter = module.get_conversation_repo
    module.get_conversation_repo = lambda: FakeConversationRepo()
    try:
        with bind_runtime_context(
            conversation_id="conversation-1",
            conversation_model_config=None,
            roundtrip_id="roundtrip-1",
            user_id="user-123",
        ):
            result = lookup_evidence.invoke(
                {"evidence_ids": ["second", "missing", "first", "second", " "]}
            )
    finally:
        module.get_conversation_repo = original_repo_getter

    assert [evidence.evidence_id for evidence in result.result.evidence] == [second.id, first.id]
    assert all(not hasattr(evidence, "raw_payload") for evidence in result.result.evidence)
    assert result.evidence == [second, first]
