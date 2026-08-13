from rendering.rendering import (
    EvidenceCard,
    InlineEvidenceReference,
    _build_block_cards,
    _build_inline_evidence,
    get_renderable_result_blocks,
)
from request_orchestrator.models.evidence import HydratedEvidence
from request_orchestrator.models.synthesized_result import SynthesisResultBlock
from tool.constants import TOOL_NAME_GET_CURRENT_WEATHER
from tool.constants import TOOL_RESULT_TYPE_WEATHER


def test_get_renderable_result_blocks_prefers_structured_result_payload() -> None:
    blocks = get_renderable_result_blocks(
        "fallback text",
        {
            "result": [
                {
                    "content": "First paragraph.",
                    "evidence_ids": ["P1E1R1", "", None],
                },
                {
                    "content": "Second paragraph.",
                    "evidence_ids": ["P1E2R3"],
                },
            ]
        },
    )

    assert blocks == [
        SynthesisResultBlock(content="First paragraph.", evidence_ids=["P1E1R1"]),
        SynthesisResultBlock(content="Second paragraph.", evidence_ids=["P1E2R3"]),
    ]


def test_get_renderable_result_blocks_falls_back_to_flat_content() -> None:
    blocks = get_renderable_result_blocks("Flat answer.", None)

    assert blocks == [SynthesisResultBlock(content="Flat answer.", evidence_ids=[])]


def test_build_block_cards_uses_hydrated_evidence_with_links() -> None:
    cards = _build_block_cards(
        SynthesisResultBlock(content="News summary", evidence_ids=["P1E1R1"]),
        {
            "P1E1R1": HydratedEvidence(
                evidence_id="P1E1R1",
                title="Article Title",
                summary="Article summary",
                url="https://example.com/article",
                image_url="https://example.com/article.jpg",
                source="news_search",
            )
        },
    )

    assert cards == [
        EvidenceCard(
            id="P1E1R1",
            name="Article Title",
            description="Article summary",
            url="https://example.com/article",
            image_url="https://example.com/article.jpg",
            source="news_search",
        )
    ]


def test_build_inline_evidence_skips_card_like_evidence() -> None:
    evidence = _build_inline_evidence(
        SynthesisResultBlock(content="News summary", evidence_ids=["P1E1R1", "P1E2R1"]),
        {
            "P1E1R1": HydratedEvidence(
                evidence_id="P1E1R1",
                title="Article Title",
                summary="Article summary",
                url="https://example.com/article",
                image_url="",
                source="news_search",
            ),
            "P1E2R1": HydratedEvidence(
                evidence_id="P1E2R1",
                title="Weather Result",
                summary="25.9 C in Toronto",
                url="",
                image_url="",
                source=TOOL_NAME_GET_CURRENT_WEATHER,
                entity_type=TOOL_RESULT_TYPE_WEATHER,
            ),
        },
    )

    assert evidence == [
        InlineEvidenceReference(
            evidence_id="P1E2R1",
            title="Weather Result",
            source=TOOL_NAME_GET_CURRENT_WEATHER,
        ),
    ]
