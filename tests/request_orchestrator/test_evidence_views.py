from uuid import uuid4

from request_orchestrator.models.agent_prompt import AgentPrompt, EvidenceStep, PromptSectionKeys
from request_orchestrator.models.evidence import EvidenceView


def test_evidence_views_exclude_internal_and_raw_fields() -> None:
    evidence = EvidenceView(
        id=uuid4(),
        tool_call_id=uuid4(),
        hash="internal-hash",
        item_id="item-1",
        tool_name="search_products",
        title="Product",
        summary="A product.",
        urls=[{"url": "https://example.com"}],
        image_url="https://example.com/image.jpg",
        source="catalog",
        entity_type="product_results",
        location_name="Toronto",
        llm_metadata={"price": 25.0},
        raw_payload={"internal": "value"},
    )

    assert evidence.compact_view() == {
        "evidence_id": str(evidence.id),
        "title": "Product",
        "summary": "A product.",
        "metadata": {"price": 25.0},
    }
    assert evidence.hydrated_view() == {
        "evidence_id": str(evidence.id),
        "title": "Product",
        "summary": "A product.",
        "metadata": {"price": 25.0},
        "item_id": "item-1",
        "tool_name": "search_products",
        "urls": [{"url": "https://example.com", "url_type": "website"}],
        "image_url": "https://example.com/image.jpg",
        "published_at": "",
        "source": "catalog",
        "entity_type": "product_results",
        "location_name": "Toronto",
    }


def test_agent_prompt_serializes_compact_evidence_view() -> None:
    evidence = EvidenceView(
        title="Product",
        summary="A product.",
        llm_metadata={"price": 25.0},
        raw_payload={"internal": "value"},
    )
    prompt = AgentPrompt(
        instruction="Use the evidence.",
        evidence=[EvidenceStep(type="product_results", evidence=[evidence])],
    )
    prompt.include_section(PromptSectionKeys.EVIDENCE)

    assert prompt.sections_raw[PromptSectionKeys.EVIDENCE][0]["evidence"] == [evidence.compact_view()]


def test_agent_prompt_serializes_evaluator_evidence_view() -> None:
    evidence = EvidenceView(
        title="Product",
        summary="A product with current pricing.",
        urls=[{"url": "https://example.com"}],
        image_url="https://example.com/image.jpg",
        source="catalog",
        entity_type="product_results",
        llm_metadata={"price": 25.0, "currency": "CAD"},
        raw_payload={"internal": "value"},
    )
    prompt = AgentPrompt(
        instruction="Evaluate the evidence.",
        evidence=[EvidenceStep(type="product_results", evidence=[evidence])],
        evidence_view="evaluator",
    )
    prompt.include_section(PromptSectionKeys.EVIDENCE)

    assert prompt.sections_raw[PromptSectionKeys.EVIDENCE][0]["evidence"] == [
        {
            "evidence_id": str(evidence.id),
            "summary": "A product with current pricing.",
            "present_data": [
                "title",
                "summary",
                "urls",
                "image_url",
                "source",
                "entity_type",
                "currency",
                "price",
            ],
        }
    ]
