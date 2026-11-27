from __future__ import annotations

import importlib

from agent_runtime.context import bind_agent_context
from capabilities.adapters.memories.lookup_evidence import lookup_evidence


def test_lookup_evidence_returns_a_batch_in_request_order(agent_state_factory, evidence_factory) -> None:
    first = evidence_factory( summary="First summary",
        data={
            "title": "First evidence",
        })
    second = evidence_factory( summary="Second summary",
        data={
            "title": "Second evidence",
        })

    class FakeEvidenceRepository:
        def get_by_ids(self, evidence_ids, *, user_id=None):
            assert evidence_ids == ["second", "missing", "first"]
            assert user_id == "user-123"
            return {"first": first, "second": second}

    module = importlib.import_module(
        "capabilities.adapters.memories.lookup_evidence"
    )
    original_evidence_repository = module.EvidenceRepository
    module.EvidenceRepository = FakeEvidenceRepository
    try:
        with bind_agent_context(agent_state_factory(user_id="user-123")):
            result = lookup_evidence.invoke(
                {"evidence_ids": ["second", "missing", "first", "second", " "]}
            )
    finally:
        module.EvidenceRepository = original_evidence_repository

    assert [evidence.id for evidence in result.result.evidence] == [second.id, first.id]
    assert all(not hasattr(evidence, "raw_payload") for evidence in result.result.evidence)
    assert result.evidence == [second, first]




