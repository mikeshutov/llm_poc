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

from agent_runtime.models.agent_prompt import (
    AgentPrompt,
    EvidenceStep,
    PromptSectionKeys,
)
from retrieval.evidence import EvidenceView, ToolResult
from retrieval.evidence_store import build_evidence_steps_from_tool_results
from agent_runtime.evaluation.prompts.evaluator_prompt import build_evaluator_prompt
from langchain_core.messages import SystemMessage


def test_evidence_views_exclude_internal_and_raw_fields() -> None:
    evidence = EvidenceView(
        id=uuid4(),
        summary="A product.",
        source="catalog",
        type="product_results",
        metadata={"price": 25.0},
        data={
            "item_id": "item-1",
            "title": "Product",
            "urls": [{"url": "https://example.com"}],
            "image_url": "https://example.com/image.jpg",
            "location_name": "Toronto",
        })

    assert evidence.model_dump(mode="json") == {
        "id": str(evidence.id),
        "type": "product_results",
        "source": "catalog",
        "summary": "A product.",
        "data": {
            "item_id": "item-1",
            "title": "Product",
            "urls": [{"url": "https://example.com"}],
            "image_url": "https://example.com/image.jpg",
            "location_name": "Toronto",
        },
        "metadata": {"price": 25.0},
    }


def test_agent_prompt_serializes_compact_evidence_view() -> None:
    evidence = EvidenceView(
        source="catalog",
        summary="A product.",
        metadata={"price": 25.0},
        data={
            "title": "Product",
        })
    prompt = AgentPrompt(
        instruction="Use the evidence.",
        evidence=[EvidenceStep(type="product_results", evidence=[evidence])],
    )
    prompt.include_section(PromptSectionKeys.EVIDENCE)

    assert prompt.sections_raw[PromptSectionKeys.EVIDENCE][0]["evidence"] == [evidence.model_dump(mode="json")]


def test_agent_prompt_serializes_evaluator_evidence_view() -> None:
    evidence = EvidenceView(
        summary="A product with current pricing.",
        source="catalog",
        type="product_results",
        metadata={"price": 25.0, "currency": "CAD"},
        data={
            "title": "Product",
            "urls": [{"url": "https://example.com"}],
            "image_url": "https://example.com/image.jpg",
        })
    prompt = AgentPrompt(
        instruction="Evaluate the evidence.",
        evidence=[EvidenceStep(type="product_results", evidence=[evidence])],
    )
    prompt.include_section(PromptSectionKeys.EVIDENCE)

    assert prompt.sections_raw[PromptSectionKeys.EVIDENCE][0]["evidence"] == [
        {
            "id": str(evidence.id),
            "type": "product_results",
            "source": "catalog",
            "summary": "A product with current pricing.",
            "data": {
                "title": "Product",
                "urls": [{"url": "https://example.com"}],
                "image_url": "https://example.com/image.jpg",
            },
            "metadata": {"price": 25.0, "currency": "CAD"},
        }
    ]


def test_evaluator_evidence_is_not_duplicated_into_prompt_messages() -> None:
    evidence = EvidenceView( summary="Useful evidence.",
        data={
            "title": "Current result",
        })
    state = SimpleNamespace(
        execution_context=SimpleNamespace(user_profile=None),
        task="Evaluate the current result.",
    )
    prompt = build_evaluator_prompt(
        state,
        [EvidenceStep(type="web_search_results", evidence=[evidence])],
    )

    messages = prompt.build_messages()

    assert isinstance(messages[0], SystemMessage)
    assert "Current result" not in messages[0].content
    assert all("Current result" not in message.content for message in messages)


def test_evidence_steps_present_empty_search_results_by_type() -> None:
    tool_result = ToolResult(
        tool_call_id=uuid4(),
        tool_name="web_search",
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
        tool_name="web_search",
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




