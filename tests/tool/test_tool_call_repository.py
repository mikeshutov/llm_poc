from uuid import UUID

from request_orchestrator.models.evidence import EvidenceUrl, EvidenceView
from common.signatures import build_signature


def test_evidence_view_for_llm_excludes_hydrated_fields() -> None:
    evidence = EvidenceView(
        id=UUID("c8271821-2d4c-51a1-bc00-1f4932d052d7"),
        step_id="main_agent:P1E1",
        item_id="card-1",
        tool_name="search_magic_cards",
        title="Knight of the Reliquary",
        summary="A creature card.",
        urls=[EvidenceUrl(url="https://example.com/card")],
        raw_payload={"oracle_text": "Grows with lands."},
        llm_metadata={"rarity": "mythic", "legal_formats": ["commander"]},
    )

    assert evidence.for_llm() == {
        "evidence_id": "c8271821-2d4c-51a1-bc00-1f4932d052d7",
        "item_id": "card-1",
        "title": "Knight of the Reliquary",
        "summary": "A creature card.",
        "metadata": {"rarity": "mythic", "legal_formats": ["commander"]},
    }
    assert "metadata" not in evidence.model_dump()
    assert evidence.model_dump()["llm_metadata"] == {"rarity": "mythic", "legal_formats": ["commander"]}


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
