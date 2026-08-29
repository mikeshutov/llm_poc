from uuid import UUID

import pytest
from pydantic import ValidationError

from request_orchestrator.models.evidence import EvidenceUrl, EvidenceView, ToolMetadata, ToolResult
from common.signatures import build_signature


def test_compact_evidence_view_excludes_hydrated_fields() -> None:
    evidence = EvidenceView(
        id=UUID("c8271821-2d4c-51a1-bc00-1f4932d052d7"),
        item_id="card-1",
        tool_name="search_magic_cards",
        title="Knight of the Reliquary",
        summary="A creature card.",
        urls=[EvidenceUrl(url="https://example.com/card")],
        raw_payload={"oracle_text": "Grows with lands."},
        llm_metadata={"rarity": "mythic", "legal_formats": ["commander"]},
    )

    assert evidence.compact_view() == {
        "evidence_id": "c8271821-2d4c-51a1-bc00-1f4932d052d7",
        "title": "Knight of the Reliquary",
        "summary": "A creature card.",
        "metadata": {"rarity": "mythic", "legal_formats": ["commander"]},
    }
    assert "metadata" not in evidence.model_dump()
    assert evidence.model_dump()["llm_metadata"] == {"rarity": "mythic", "legal_formats": ["commander"]}


def test_tool_result_uses_a_typed_tool_metadata_object() -> None:
    result = ToolResult(
        result={"status": "ok"},
        tool_metadata=ToolMetadata(retrieved_count=20, reranked=True),
    )

    assert result.tool_metadata.model_dump(exclude_none=True) == {
        "retrieved_count": 20,
        "reranked": True,
    }
    assert result.model_dump(exclude_none=True)["tool_metadata"] == {
        "retrieved_count": 20,
        "reranked": True,
    }
    with pytest.raises(ValidationError):
        ToolMetadata(unsupported_metadata=True)


def test_request_hash_is_stable_for_equivalent_input() -> None:
    first = build_signature(
        {"tool_name": "search_magic_cards", "input": {"query": "Knight of the Reliquary", "limit": 5}}
    )
    second = build_signature(
        {"tool_name": "search_magic_cards", "input": {"limit": 5, "query": "Knight of the Reliquary"}}
    )

    assert first == second
    assert len(first) == 64


def test_request_hash_changes_for_different_tool_requests() -> None:
    request = {"query": "Knight of the Reliquary", "limit": 5}

    assert build_signature({"tool_name": "search_magic_cards", "input": request}) != build_signature(
        {"tool_name": "generic_web_search", "input": request}
    )
    assert build_signature({"tool_name": "search_magic_cards", "input": request}) != build_signature(
        {"tool_name": "search_magic_cards", "input": {"query": "Black Lotus", "limit": 5}}
    )


def test_signature_is_stable_for_equivalent_domain_data() -> None:
    assert build_signature({"title": "Black Lotus", "metadata": {"set": "LEA"}}) == build_signature(
        {"metadata": {"set": "LEA"}, "title": "Black Lotus"}
    )
