from rendering.rendering import (
    EvidenceCard,
    InlineEvidenceReference,
    _build_block_cards,
    _build_inline_evidence,
    _is_inline_label_evidence,
    _is_inline_link_evidence,
    _render_result_block,
    get_renderable_result_blocks,
    render_assistant_content,
    serialize_roundtrip_payload,
)
from rendering.debug import _build_llm_call_payload
from retrieval.evidence import EvidenceUrl, EvidenceUrlType, EvidenceView
from agent_runtime.models.orchestrator_payload import OrchestratorPayload
from agent_runtime.models.synthesized_result import SynthesisResultBlock
from capabilities.registry.constants import TOOL_NAME_GET_CURRENT_WEATHER
from capabilities.registry.constants import TOOL_NAME_GENERIC_WEB_SEARCH
from capabilities.registry.constants import TOOL_RESULT_TYPE_FILE_DETAILS
from capabilities.registry.constants import TOOL_RESULT_TYPE_RULES
from capabilities.registry.constants import TOOL_RESULT_TYPE_MEAL_RESULTS
from capabilities.registry.constants import TOOL_RESULT_TYPE_WEATHER


def test_debug_llm_payload_reads_compact_input_and_output_metadata() -> None:
    payload = _build_llm_call_payload(
        {
            "model": "test-model",
            "metadata": {
                "input": {"hash": "input-hash", "message_count": 2},
                "output": {"hash": "output-hash", "preview": "ok"},
            },
        }
    )

    assert payload["input_object"] == {"hash": "input-hash", "message_count": 2}
    assert payload["output_object"] == {"hash": "output-hash", "preview": "ok"}
    assert payload["metadata"] == {}


def test_serialize_roundtrip_payload_preserves_persisted_evidence() -> None:
    evidence = EvidenceView(title="Pasta Primavera", summary="A recipe")
    payload = serialize_roundtrip_payload(OrchestratorPayload(
        result=[{"content": "Here is pasta.", "evidence_ids": [str(evidence.id)]}],
        evidence_by_id={str(evidence.id): evidence},
    ))

    assert payload["result"][0]["evidence_ids"] == [str(evidence.id)]
    assert payload["evidence_by_id"][str(evidence.id)]["title"] == "Pasta Primavera"


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
            entity_type="",
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


def test_render_assistant_content_renders_meal_recipe_cards(monkeypatch) -> None:
    rendered_meals: list[EvidenceView] = []
    monkeypatch.setattr("rendering.rendering.st.markdown", lambda value: None)
    monkeypatch.setattr("rendering.rendering.fetch_llm_usage_for_roundtrip", lambda value: None)
    monkeypatch.setattr("rendering.rendering.fetch_trace_messages_for_roundtrip", lambda value: {})
    monkeypatch.setattr("rendering.rendering.render_agent_logs", lambda value: None)
    monkeypatch.setattr("rendering.rendering.render_meal_evidence_cards", rendered_meals.extend)

    render_assistant_content(
        "Here are recipes.",
        {
            "result": [{"content": "Here are recipes.", "evidence_ids": ["meal-1"]}],
            "evidence_by_id": {
                "meal-1": EvidenceView(
                    evidence_id="meal-1",
                    title="Pasta Primavera",
                    summary="Vegetable pasta",
                    urls=[EvidenceUrl(url="https://example.com/pasta", url_type=EvidenceUrlType.WEBSITE)],
                    image_url="https://example.com/pasta.jpg",
                    source="search_meals",
                    entity_type=TOOL_RESULT_TYPE_MEAL_RESULTS,
                ).model_dump(mode="json")
            },
        },
    )

    assert len(rendered_meals) == 1
    assert rendered_meals[0].entity_type == TOOL_RESULT_TYPE_MEAL_RESULTS


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


def test_render_result_block_renders_file_citations_as_inline_file_names(monkeypatch) -> None:
    calls: dict[str, list[object]] = {"markdown": [], "caption": []}

    monkeypatch.setattr(
        "rendering.rendering.st.markdown",
        lambda value, **kwargs: calls["markdown"].append((value, kwargs)),
    )
    monkeypatch.setattr("rendering.rendering.st.caption", lambda value: calls["caption"].append(value))

    cards = _render_result_block(
        SynthesisResultBlock(content="Your resume lists three years of experience.", evidence_ids=["file-evidence"]),
        {
            "file-evidence": EvidenceView(
                evidence_id="file-evidence",
                title="resume.pdf",
                summary="Experience at Acme Corp.",
                urls=[EvidenceUrl(url="app/static/files/resume.pdf", url_type=EvidenceUrlType.WEBSITE)],
                source="search_file_for_details",
                entity_type=TOOL_RESULT_TYPE_FILE_DETAILS,
            )
        },
    )

    assert cards == []
    assert len(calls["markdown"]) == 1
    rendered_value, _ = calls["markdown"][0]
    assert "app/static/files/resume.pdf" in rendered_value
    assert calls["caption"] == []


def test_render_result_block_renders_weather_as_inline_html_link(monkeypatch) -> None:
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
    assert rendered_kwargs == {}


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
