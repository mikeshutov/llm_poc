from uuid import uuid4
import sys
from types import ModuleType, SimpleNamespace

if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = ModuleType("yfinance")

if "pycountry" not in sys.modules:
    pycountry_module = ModuleType("pycountry")
    pycountry_module.countries = SimpleNamespace(
        lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper())
    )
    sys.modules["pycountry"] = pycountry_module

from request_orchestrator.models.agent_prompt import (
    EVIDENCE_VIEW_EVALUATOR,
    AgentPrompt,
    EvidenceStep,
    PromptSectionKeys,
)
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from request_orchestrator.shared.evidence import build_evidence_steps_from_tool_results


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
        evidence_view=EVIDENCE_VIEW_EVALUATOR,
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


def test_evidence_steps_present_empty_search_results_by_type() -> None:
    tool_result = ToolResult(
        tool_call_id=uuid4(),
        tool_name="generic_web_search",
        result={"results": []},
    )
    evidence_steps = build_evidence_steps_from_tool_results(
        [tool_result],
        evidence_views_by_tool_call_id={},
    )
    prompt = AgentPrompt(instruction="Use the evidence.", evidence=evidence_steps)
    prompt.include_section(PromptSectionKeys.EVIDENCE)

    assert prompt.sections_raw[PromptSectionKeys.EVIDENCE] == [
        {"type": "web_search_results", "no_results": True}
    ]


def test_evidence_steps_do_not_label_tool_errors_as_no_results() -> None:
    tool_result = ToolResult(
        tool_call_id=uuid4(),
        tool_name="generic_web_search",
        result={"error": "Search provider unavailable."},
    )
    evidence_steps = build_evidence_steps_from_tool_results(
        [tool_result],
        evidence_views_by_tool_call_id={},
    )
    prompt = AgentPrompt(instruction="Use the evidence.", evidence=evidence_steps)
    prompt.include_section(PromptSectionKeys.EVIDENCE)

    assert prompt.sections_raw[PromptSectionKeys.EVIDENCE] == [
        {"type": "web_search_results"}
    ]
