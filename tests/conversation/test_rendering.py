from rendering.rendering import (
    EvidenceCard,
    InlineEvidenceReference,
    _build_block_cards,
    _build_inline_evidence,
    _is_inline_label_evidence,
    _is_inline_link_evidence,
    _render_result_block,
    get_renderable_result_blocks,
)
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView
from request_orchestrator.models.synthesized_result import SynthesisResultBlock
from tool.constants import TOOL_NAME_GET_CURRENT_WEATHER
from tool.constants import TOOL_NAME_GENERIC_WEB_SEARCH
from tool.constants import TOOL_RESULT_TYPE_RULES
from tool.constants import TOOL_RESULT_TYPE_WEATHER


def test_get_renderable_result_blocks_prefers_structured_result_payload() -> None:
    blocks = get_renderable_result_blocks(
        "fallback text",
        {
            "result": [
                {
                    "content": "First paragraph.",
                    "evidence_ids": ["25a4bcc1-2b18-5a36-940c-29c535bae654", "", None],
                },
                {
                    "content": "Second paragraph.",
                    "evidence_ids": ["6b4ecb0d-c842-51dd-828a-685e39f6f714"],
                },
            ]
        },
    )

    assert blocks == [
        SynthesisResultBlock(content="First paragraph.", evidence_ids=["25a4bcc1-2b18-5a36-940c-29c535bae654"]),
        SynthesisResultBlock(content="Second paragraph.", evidence_ids=["6b4ecb0d-c842-51dd-828a-685e39f6f714"]),
    ]


def test_get_renderable_result_blocks_falls_back_to_flat_content() -> None:
    blocks = get_renderable_result_blocks("Flat answer.", None)

    assert blocks == [SynthesisResultBlock(content="Flat answer.", evidence_ids=[])]


def test_build_block_cards_uses_hydrated_evidence_with_links() -> None:
    cards = _build_block_cards(
        SynthesisResultBlock(content="News summary", evidence_ids=["25a4bcc1-2b18-5a36-940c-29c535bae654"]),
        {
            "25a4bcc1-2b18-5a36-940c-29c535bae654": EvidenceView(
                evidence_id="25a4bcc1-2b18-5a36-940c-29c535bae654",
                title="Article Title",
                summary="Article summary",
                urls=[EvidenceUrl(url="https://example.com/article", url_type=EvidenceUrlType.WEBSITE)],
                image_url="https://example.com/article.jpg",
                source="news_search",
            )
        },
    )

    assert cards == [
        EvidenceCard(
            id="25a4bcc1-2b18-5a36-940c-29c535bae654",
            name="Article Title",
            description="Article summary",
            url="https://example.com/article",
            image_url="https://example.com/article.jpg",
            source="news_search",
        )
    ]


def test_build_inline_evidence_skips_card_like_evidence() -> None:
    evidence = _build_inline_evidence(
        SynthesisResultBlock(content="News summary", evidence_ids=["25a4bcc1-2b18-5a36-940c-29c535bae654", "c38c296c-3e94-56a7-86e1-dfe071c82fcc"]),
        {
            "25a4bcc1-2b18-5a36-940c-29c535bae654": EvidenceView(
                evidence_id="25a4bcc1-2b18-5a36-940c-29c535bae654",
                title="Article Title",
                summary="Article summary",
                urls=[EvidenceUrl(url="https://example.com/article", url_type=EvidenceUrlType.WEBSITE)],
                image_url="",
                source="news_search",
            ),
            "c38c296c-3e94-56a7-86e1-dfe071c82fcc": EvidenceView(
                evidence_id="c38c296c-3e94-56a7-86e1-dfe071c82fcc",
                title="Weather Result",
                summary="25.9 C in Toronto",
                urls=[],
                image_url="",
                source=TOOL_NAME_GET_CURRENT_WEATHER,
                entity_type=TOOL_RESULT_TYPE_WEATHER,
            ),
        },
    )

    assert evidence == [
        InlineEvidenceReference(
            evidence_id="c38c296c-3e94-56a7-86e1-dfe071c82fcc",
            title="Weather Result",
            source=TOOL_NAME_GET_CURRENT_WEATHER,
        ),
    ]
    assert _is_inline_label_evidence(
        EvidenceView(
            evidence_id="c38c296c-3e94-56a7-86e1-dfe071c82fcc",
            title="Weather Result",
            summary="25.9 C in Toronto",
            urls=[],
            image_url="",
            source=TOOL_NAME_GET_CURRENT_WEATHER,
            entity_type=TOOL_RESULT_TYPE_WEATHER,
        )
    )


def test_build_inline_evidence_includes_generic_web_search_results() -> None:
    evidence = _build_inline_evidence(
        SynthesisResultBlock(content="Web summary", evidence_ids=["25a4bcc1-2b18-5a36-940c-29c535bae654"]),
        {
            "25a4bcc1-2b18-5a36-940c-29c535bae654": EvidenceView(
                evidence_id="25a4bcc1-2b18-5a36-940c-29c535bae654",
                title="Article Title",
                summary="Article summary",
                urls=[EvidenceUrl(url="https://example.com/article", url_type=EvidenceUrlType.WEBSITE)],
                image_url="https://example.com/article.jpg",
                source=TOOL_NAME_GENERIC_WEB_SEARCH,
                tool_name=TOOL_NAME_GENERIC_WEB_SEARCH,
            )
        },
    )

    assert evidence == [
        InlineEvidenceReference(
            evidence_id="25a4bcc1-2b18-5a36-940c-29c535bae654",
            title="Article Title",
            source=TOOL_NAME_GENERIC_WEB_SEARCH,
        ),
    ]
    assert _is_inline_link_evidence(
        EvidenceView(
            evidence_id="25a4bcc1-2b18-5a36-940c-29c535bae654",
            title="Article Title",
            summary="Article summary",
            urls=[EvidenceUrl(url="https://example.com/article", url_type=EvidenceUrlType.WEBSITE)],
            image_url="https://example.com/article.jpg",
            source=TOOL_NAME_GENERIC_WEB_SEARCH,
            tool_name=TOOL_NAME_GENERIC_WEB_SEARCH,
        )
    )


def test_build_inline_evidence_includes_generic_web_search_results_from_source_only() -> None:
    evidence = _build_inline_evidence(
        SynthesisResultBlock(content="Web summary", evidence_ids=["25a4bcc1-2b18-5a36-940c-29c535bae654"]),
        {
            "25a4bcc1-2b18-5a36-940c-29c535bae654": EvidenceView(
                evidence_id="25a4bcc1-2b18-5a36-940c-29c535bae654",
                title="Article Title",
                summary="Article summary",
                urls=[EvidenceUrl(url="https://example.com/article", url_type=EvidenceUrlType.WEBSITE)],
                image_url="https://example.com/article.jpg",
                source=TOOL_NAME_GENERIC_WEB_SEARCH,
                tool_name="",
            )
        },
    )

    assert evidence == [
        InlineEvidenceReference(
            evidence_id="25a4bcc1-2b18-5a36-940c-29c535bae654",
            title="Article Title",
            source=TOOL_NAME_GENERIC_WEB_SEARCH,
        ),
    ]


def test_render_result_block_renders_weather_as_inline_markdown_link(monkeypatch) -> None:
    calls: dict[str, list[object]] = {"markdown": [], "write": [], "caption": []}

    monkeypatch.setattr(
        "rendering.rendering.st.markdown",
        lambda value, **kwargs: calls["markdown"].append((value, kwargs)),
    )
    monkeypatch.setattr("rendering.rendering.st.write", lambda value: calls["write"].append(value))
    monkeypatch.setattr("rendering.rendering.st.caption", lambda value: calls["caption"].append(value))

    cards = _render_result_block(
        SynthesisResultBlock(content="Toronto is warm today.", evidence_ids=["25a4bcc1-2b18-5a36-940c-29c535bae654"]),
        {
            "25a4bcc1-2b18-5a36-940c-29c535bae654": EvidenceView(
                evidence_id="25a4bcc1-2b18-5a36-940c-29c535bae654",
                title="Get Current Weather",
                summary="25.9 C in Toronto",
                urls=[EvidenceUrl(url="https://open-meteo.com/", url_type=EvidenceUrlType.WEBSITE)],
                image_url="",
                source=TOOL_NAME_GET_CURRENT_WEATHER,
                tool_name=TOOL_NAME_GET_CURRENT_WEATHER,
                entity_type=TOOL_RESULT_TYPE_WEATHER,
            )
        },
    )

    assert cards == []
    assert calls["write"] == []
    assert calls["caption"] == []
    assert len(calls["markdown"]) == 1
    rendered_value, rendered_kwargs = calls["markdown"][0]
    assert "Toronto is warm today." in rendered_value
    assert "https://open-meteo.com/" in rendered_value
    assert rendered_kwargs == {"unsafe_allow_html": True}


def test_render_result_block_renders_magic_card_rulings_as_a_table(monkeypatch) -> None:
    rendered_rulings: list[list[EvidenceView]] = []
    monkeypatch.setattr("rendering.rendering.st.write", lambda value: None)
    monkeypatch.setattr(
        "rendering.rendering.render_magic_card_rulings",
        lambda items: rendered_rulings.append(list(items)),
    )

    cards = _render_result_block(
        SynthesisResultBlock(content="Humility rulings", evidence_ids=["25a4bcc1-2b18-5a36-940c-29c535bae654"]),
        {
            "25a4bcc1-2b18-5a36-940c-29c535bae654": EvidenceView(
                evidence_id="25a4bcc1-2b18-5a36-940c-29c535bae654",
                title="Humility Ruling 1",
                summary="Humility applies in layers 6 and 7b.",
                entity_type=TOOL_RESULT_TYPE_RULES,
                published_at="2004-10-04",
            )
        },
    )

    assert cards == []
    assert len(rendered_rulings) == 1
    assert rendered_rulings[0][0].published_at == "2004-10-04"
