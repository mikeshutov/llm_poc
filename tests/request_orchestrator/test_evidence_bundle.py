from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

if 'yfinance' not in sys.modules:
    sys.modules['yfinance'] = ModuleType('yfinance')

if 'pycountry' not in sys.modules:
    pycountry_module = ModuleType('pycountry')
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules['pycountry'] = pycountry_module

from integrations.brave.models import NewsSearchResponse, WebSearchResponse
from integrations.meal_db.models import MealSearchResult
from integrations.wikidata.models import SparqlResult
from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE
from request_orchestrator.models.evidence import EvidenceView, HydratedEvidence
from request_orchestrator.shared.tool_adapter.search.wikipedia_search import WikipediaSearchResponse
from integrations.wikipedia.models import WikipediaPageSummary, WikipediaSearchResult
from request_orchestrator.models.evidence import ToolResult
from request_orchestrator.models.plan import Plan
from request_orchestrator.shared.evidence import (
    build_evidence_bundle_from_tool_results,
    build_evidence_steps_from_tool_results,
)
from request_orchestrator.shared.tool_adapter.food.search_meals import _tool_result as meal_tool_result
from request_orchestrator.shared.tool_adapter.news.hn_search import _tool_result as hn_tool_result
from request_orchestrator.shared.tool_adapter.search.generic_web_search import _web_search_tool_result
from request_orchestrator.shared.tool_adapter.search.structured_facts_lookup import _tool_result as structured_facts_tool_result
from request_orchestrator.shared.tool_adapter.search.wikipedia_search import _tool_result as wikipedia_tool_result
from integrations.open_meteo.models import CurrentWeather, GeocodedLocation
from request_orchestrator.shared.tool_adapter.search.brave_news_search import _tool_result as news_tool_result
from request_orchestrator.shared.tool_adapter.weather.get_current_weather import CurrentWeatherResult, _tool_result as current_weather_tool_result


@dataclass
class IterationState:
    plan: Plan
    results: dict[str, ToolResult]


def _tool_result(result, tool_name: str) -> ToolResult:
    if tool_name == "news_search":
        return news_tool_result(result)
    if tool_name == "structured_facts_lookup":
        return structured_facts_tool_result(result)
    if tool_name == "get_current_weather":
        return current_weather_tool_result(result)
    if tool_name == "generic_web_search":
        return _web_search_tool_result(result)
    if tool_name == "search_meals":
        return meal_tool_result(result)
    if tool_name == "wikipedia_search":
        return wikipedia_tool_result(result)
    if tool_name == "custom_lookup":
        hydrated = HydratedEvidence(
            item_id="fallback-1",
            tool_name="custom_lookup",
            title="Custom Lookup",
            summary="A generic fallback record.",
            source="custom_lookup",
            entity_type="generic",
            raw_payload=result,
        )
        return ToolResult(
            result=result,
            evidence_views=[
                EvidenceView(
                    item_id=hydrated.item_id,
                    title=hydrated.title,
                    summary=hydrated.summary,
                    metadata={},
                )
            ],
            hydrated_evidence=[hydrated],
        )
    raise AssertionError(f"Unsupported test tool_name {tool_name}")


def _gather_tool_results(iterations: list[IterationState]) -> list[ToolResult]:
    gathered: list[ToolResult] = []
    for iteration in iterations:
        tool_name_by_step_id = {
            step_id: next(
                (step.tool for step in iteration.plan.steps if step_id.endswith(step.id)),
                "",
            )
            for step_id in iteration.results
        }
        for step_id, tool_result in iteration.results.items():
            gathered.append(
                tool_result.model_copy(
                    update={
                        "step_id": tool_result.step_id or step_id,
                        "tool_name": tool_result.tool_name or tool_name_by_step_id.get(step_id, ""),
                    }
                )
            )
    return gathered


def _build_evidence_bundle(iterations: list[IterationState]):
    return build_evidence_bundle_from_tool_results(_gather_tool_results(iterations))


def _build_evidence_steps(iterations: list[IterationState], evidence_views_by_step_id: dict[str, list[EvidenceView]]):
    return build_evidence_steps_from_tool_results(
        _gather_tool_results(iterations),
        evidence_views_by_step_id,
    )


build_evidence_bundle = _build_evidence_bundle
build_evidence_steps = _build_evidence_steps


def test_build_evidence_bundle_creates_canonical_news_records() -> None:
    iteration = IterationState(
        plan=Plan.model_validate(
            {
                "steps": [
                    {
                        "id": "E1",
                        "plan": "Search the news.",
                        "tool": "news_search",
                        "args": {"query": "toronto weather"},
                    }
                ]
            }
        ),
        results={
            "P1E1": _tool_result(
                NewsSearchResponse.model_validate(
                {
                    "query": {"original": "toronto weather"},
                    "results": [
                        {
                            "title": "Toronto sees clear skies",
                            "url": "https://example.com/news-1",
                            "description": "Sunny conditions continue.",
                            "thumbnail_url": "https://example.com/news-1.jpg",
                        },
                        {
                            "title": "Humidity rises in Toronto",
                            "url": "https://example.com/news-2",
                            "description": "Sticky air expected this afternoon.",
                        },
                    ],
                }
                ),
                "news_search",
            )
        },
    )

    bundle = build_evidence_bundle([iteration])

    assert set(bundle.hydrated_evidence_by_id) == {"P1E1R1", "P1E1R2"}
    assert [view.evidence_id for view in bundle.evidence_views_by_step_id["P1E1"]] == ["P1E1R1", "P1E1R2"]

    first = bundle.hydrated_evidence_by_id["P1E1R1"]
    assert first.step_id == "P1E1"
    assert first.item_id == "https://example.com/news-1"
    assert first.url == "https://example.com/news-1"
    assert [(entry.url_type, entry.url) for entry in first.urls] == [
        ("website", "https://example.com/news-1"),
    ]
    assert first.title == "Toronto sees clear skies"
    assert first.summary == "Sunny conditions continue."
    assert first.image_url == "https://example.com/news-1.jpg"
    assert first.source == "news_search"
    assert first.entity_type == "news_results"


def test_build_evidence_bundle_preserves_item_id_separately_from_evidence_id() -> None:
    iteration = IterationState(
        plan=Plan.model_validate(
            {
                "steps": [
                    {
                        "id": "E1",
                        "plan": "Look up structured facts.",
                        "tool": "structured_facts_lookup",
                        "args": {"query": "Toronto"},
                    }
                ]
            }
        ),
        results={
            "P1E1": _tool_result(
                SparqlResult(
                    sparql="SELECT * WHERE {}",
                    vars=["qid", "itemLabel", "url"],
                    bindings=[
                        {
                            "qid": {"value": "Q172"},
                            "itemLabel": {"value": "Toronto"},
                            "url": {"value": "https://www.wikidata.org/wiki/Q172"},
                        }
                    ],
                ),
                "structured_facts_lookup",
            )
        },
    )

    bundle = build_evidence_bundle([iteration])

    record = bundle.hydrated_evidence_by_id["P1E1R1"]
    assert record.evidence_id == "P1E1R1"
    assert record.item_id == "Q172"
    assert record.url == "https://www.wikidata.org/wiki/Q172"
    assert [(entry.url_type, entry.url) for entry in record.urls] == [
        ("website", "https://www.wikidata.org/wiki/Q172"),
    ]
    assert record.title == "Toronto"
    assert record.summary == "qid=Q172, itemLabel=Toronto, url=https://www.wikidata.org/wiki/Q172"
    assert bundle.evidence_views_by_step_id["P1E1"][0].item_id == "Q172"


def test_build_evidence_bundle_unwraps_nested_structured_fact_values() -> None:
    iteration = IterationState(
        plan=Plan.model_validate(
            {
                "steps": [
                    {
                        "id": "E3",
                        "plan": "Look up well-known facts and transit basics about Toronto.",
                        "tool": "structured_facts_lookup",
                        "args": {"sparql": "SELECT ?item ?itemLabel WHERE {}"},
                    }
                ]
            }
        ),
        results={
            "P1E3": _tool_result(
                SparqlResult(
                    sparql="SELECT ?item ?itemLabel WHERE {}",
                    vars=["item", "itemLabel"],
                    bindings=[
                        {
                            "item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q172"},
                            "itemLabel": {"xml:lang": "en", "type": "literal", "value": "Toronto"},
                        }
                    ],
                ),
                "structured_facts_lookup",
            )
        },
    )

    bundle = build_evidence_bundle([iteration])

    record = bundle.hydrated_evidence_by_id["P1E3R1"]
    assert record.item_id == "Toronto"
    assert record.title == "Toronto"
    assert record.summary == "item=http://www.wikidata.org/entity/Q172, itemLabel=Toronto"


def test_build_evidence_bundle_normalizes_weather_to_single_record() -> None:
    iteration = IterationState(
        plan=Plan.model_validate(
            {
                "steps": [
                    {
                        "id": "E1",
                        "plan": "Get weather.",
                        "tool": "get_current_weather",
                        "args": {"location": "Toronto"},
                    }
                ]
            }
        ),
        results={
            "P1E1": _tool_result(
                CurrentWeatherResult(
                    location=GeocodedLocation(
                        name="Toronto",
                        country="Canada",
                        latitude=43.7,
                        longitude=-79.4,
                        timezone="America/Toronto",
                    ),
                    weather=CurrentWeather(
                        latitude=43.7,
                        longitude=-79.4,
                        timezone="America/Toronto",
                        elevation=100.0,
                        time="2026-08-12T11:15",
                        temperature=21.2,
                        windspeed=4.3,
                        winddirection=180.0,
                        weathercode=0,
                        is_day=True,
                    ),
                ),
                "get_current_weather",
            )
        },
    )

    bundle = build_evidence_bundle([iteration])

    record = bundle.hydrated_evidence_by_id["P1E1R1"]
    assert record.title == "Get Current Weather"
    assert record.item_id == "Toronto"
    assert record.location_name == "Toronto"
    assert record.url == "https://open-meteo.com/"
    assert record.source == "get_current_weather"
    assert record.summary == "21.2 C in Toronto, Canada, wind 4.3 km/h, at 2026-08-12T11:15"


def test_build_evidence_bundle_normalizes_generic_lists_and_singletons() -> None:
    iteration = IterationState(
        plan=Plan.model_validate(
            {
                "steps": [
                    {
                        "id": "E1",
                        "plan": "Search the web.",
                        "tool": "generic_web_search",
                        "args": {"query_text": "best ramen toronto"},
                    },
                    {
                        "id": "E2",
                        "plan": "Collect fallback metadata.",
                        "tool": "custom_lookup",
                        "args": {},
                    },
                ]
            }
        ),
        results={
            "P1E1": _tool_result(
                WebSearchResponse.model_validate(
                    {
                        "query": "best ramen toronto",
                        "results": [
                            {
                                "title": "Ramen spot",
                                "url": "https://example.com/ramen",
                                "description": "Popular local ramen shop.",
                                "image_url": "https://example.com/ramen.jpg",
                            }
                        ],
                    }
                ),
                "generic_web_search",
            ),
            "P1E2": _tool_result({"id": "fallback-1", "description": "A generic fallback record."}, "custom_lookup"),
        },
    )

    bundle = build_evidence_bundle([iteration])

    web_record = bundle.hydrated_evidence_by_id["P1E1R1"]
    assert web_record.item_id == "https://example.com/ramen"
    assert web_record.title == "Ramen spot"
    assert web_record.summary == "Popular local ramen shop."
    assert web_record.image_url == "https://example.com/ramen.jpg"
    assert web_record.source == "generic_web_search"
    assert web_record.entity_type == "web_search_results"
    assert web_record.metadata == {}

    fallback_record = bundle.hydrated_evidence_by_id["P1E2R1"]
    assert fallback_record.item_id == "fallback-1"
    assert fallback_record.title == "Custom Lookup"
    assert fallback_record.summary == "A generic fallback record."
    assert fallback_record.url == ""
    assert fallback_record.urls == []


def test_build_evidence_bundle_uses_pre_normalized_tool_evidence_when_present() -> None:
    iteration = IterationState(
        plan=Plan.model_validate(
            {
                "steps": [
                    {
                        "id": "E1",
                        "plan": "Search the web.",
                        "tool": "generic_web_search",
                        "args": {"query_text": "best ramen toronto"},
                    }
                ]
            }
        ),
        results={
            "P1E1": ToolResult.model_validate(
                {
                    "result": {
                        "results": [
                            {
                                "title": "Ramen spot",
                                "url": "https://example.com/ramen",
                                "description": "Popular local ramen shop.",
                            }
                        ]
                    },
                    "evidence_views": [
                        {
                            "item_id": "ramen-1",
                            "title": "Pre-normalized Ramen Spot",
                            "summary": "Normalized by the tool layer.",
                            "metadata": {"quality": "high"},
                        }
                    ],
                    "hydrated_evidence": [
                        {
                            "item_id": "ramen-1",
                            "tool_name": "generic_web_search",
                            "title": "Pre-normalized Ramen Spot",
                            "summary": "Normalized by the tool layer.",
                            "url": "https://example.com/ramen",
                            "urls": [{"url": "https://example.com/ramen", "url_type": "website"}],
                            "source": "generic_web_search",
                            "entity_type": "web_search_results",
                            "metadata": {"quality": "high"},
                        }
                    ],
                }
            )
        },
    )

    bundle = build_evidence_bundle([iteration])

    record = bundle.hydrated_evidence_by_id["P1E1R1"]
    assert record.evidence_id == "P1E1R1"
    assert record.step_id == "P1E1"
    assert record.item_id == "ramen-1"
    assert record.title == "Pre-normalized Ramen Spot"
    assert record.summary == "Normalized by the tool layer."
    assert record.url == "https://example.com/ramen"
    assert record.metadata == {"quality": "high"}


def test_build_evidence_bundle_prefers_tool_result_step_context_when_present() -> None:
    iteration = IterationState(
        plan=Plan.model_validate(
            {
                "steps": [
                    {
                        "id": "E1",
                        "plan": "Search the web.",
                        "tool": "generic_web_search",
                        "args": {"query_text": "best ramen toronto"},
                    }
                ]
            }
        ),
        results={
            "P1E1": ToolResult.model_validate(
                {
                    "step_id": "P9E3",
                    "tool_name": "generic_web_search",
                    "iteration": 9,
                    "result": {"results": [{"title": "Ramen spot"}]},
                    "evidence_views": [
                        {
                            "item_id": "ramen-1",
                            "title": "Prepared Ramen Spot",
                            "summary": "Prepared on write.",
                        }
                    ],
                    "hydrated_evidence": [
                        {
                            "item_id": "ramen-1",
                            "title": "Prepared Ramen Spot",
                            "summary": "Prepared on write.",
                            "source": "generic_web_search",
                            "entity_type": "web_search_results",
                        }
                    ],
                }
            )
        },
    )

    bundle = build_evidence_bundle([iteration])
    evidence_steps = build_evidence_steps([iteration], bundle.evidence_views_by_step_id)

    assert set(bundle.hydrated_evidence_by_id) == {"P9E3R1"}
    assert [view.evidence_id for view in bundle.evidence_views_by_step_id["P9E3"]] == ["P9E3R1"]
    assert evidence_steps[0].type == "web_search_results"
    assert [evidence.evidence_id for evidence in evidence_steps[0].evidence] == ["P9E3R1"]


def test_build_evidence_bundle_falls_back_to_evidence_views_when_hydrated_records_are_missing() -> None:
    iteration = IterationState(
        plan=Plan.model_validate(
            {
                "steps": [
                    {
                        "id": "E1",
                        "plan": "Search the web.",
                        "tool": "generic_web_search",
                        "args": {"query_text": "best ramen toronto"},
                    }
                ]
            }
        ),
        results={
            "P1E1": ToolResult.model_validate(
                {
                    "result": {"results": [{"title": "Ramen spot"}]},
                    "evidence_views": [
                        {
                            "item_id": "ramen-1",
                            "title": "View-only Ramen Spot",
                            "summary": "Evidence survived as a view only.",
                            "metadata": {"quality": "high"},
                        }
                    ],
                    "hydrated_evidence": [],
                }
            )
        },
    )

    bundle = build_evidence_bundle([iteration])
    evidence_steps = build_evidence_steps([iteration], bundle.evidence_views_by_step_id)

    assert [view.evidence_id for view in bundle.evidence_views_by_step_id["P1E1"]] == ["P1E1R1"]
    record = bundle.hydrated_evidence_by_id["P1E1R1"]
    assert record.item_id == "ramen-1"
    assert record.title == "View-only Ramen Spot"
    assert record.summary == "Evidence survived as a view only."
    assert record.source == "generic_web_search"
    assert record.entity_type == "web_search_results"
    assert record.metadata == {"quality": "high"}
    assert len(evidence_steps) == 1
    assert evidence_steps[0].type == "web_search_results"
    assert [evidence.title for evidence in evidence_steps[0].evidence] == ["View-only Ramen Spot"]


def test_build_evidence_bundle_uses_meal_entry_description_instead_of_wrapper_metadata() -> None:
    iteration = IterationState(
        plan=Plan.model_validate(
            {
                "steps": [
                    {
                        "id": "E4",
                        "plan": "Look up another widely associated Toronto/Canadian food recipe for variety.",
                        "tool": "search_meals",
                        "args": {"query": "butter tart"},
                    }
                ]
            }
        ),
        results={
            "P1E4": _tool_result(
                MealSearchResult.model_validate(
                    {
                        "meals": [
                            {
                                "idMeal": "meal-1",
                                "strMeal": "Butter Tart",
                                "strInstructions": "Bake the tart shells, fill with butter tart filling, and bake until set.",
                                "strMealThumb": "https://example.com/butter-tart.jpg",
                                "strSource": "https://example.com/butter-tart-recipe",
                                "strYoutube": "https://youtube.com/watch?v=butter-tart",
                                "strIngredient1": "Butter",
                                "strMeasure1": "1/2 cup",
                                "strIngredient2": "Sugar",
                                "strMeasure2": "1 cup",
                            }
                        ],
                        "retrieved_count": 1,
                        "reranked": True,
                    }
                ),
                "search_meals",
            )
        },
    )

    bundle = build_evidence_bundle([iteration])
    evidence_steps = build_evidence_steps([iteration], bundle.evidence_views_by_step_id)

    record = bundle.hydrated_evidence_by_id["P1E4R1"]
    assert record.item_id == "meal-1"
    assert record.title == "Butter Tart"
    assert record.summary == "Bake the tart shells, fill with butter tart filling, and bake until set."
    assert record.url == "https://example.com/butter-tart-recipe"
    assert [(entry.url_type, entry.url) for entry in record.urls] == [
        ("website", "https://example.com/butter-tart-recipe"),
        ("youtube", "https://youtube.com/watch?v=butter-tart"),
    ]
    assert record.image_url == "https://example.com/butter-tart.jpg"
    assert record.metadata == {
        "category": None,
        "area": None,
        "tags": None,
        "ingredients": [
            {"name": "Butter", "measure": "1/2 cup"},
            {"name": "Sugar", "measure": "1 cup"},
        ],
    }
    assert bundle.evidence_views_by_step_id["P1E4"][0].metadata == record.metadata
    assert evidence_steps[0].metadata == {
        "retrieved_count": 1,
        "reranked": True,
    }


def test_build_evidence_bundle_normalizes_wikipedia_results_per_page() -> None:
    iteration = IterationState(
        plan=Plan.model_validate(
            {
                "steps": [
                    {
                        "id": "E2",
                        "plan": "Search Wikipedia.",
                        "tool": "wikipedia_search",
                        "args": {"query": "staple foods of the United Kingdom"},
                    }
                ]
            }
        ),
        results={
            "P1E2": _tool_result(
                WikipediaSearchResponse(
                    query="staple foods of the United Kingdom",
                    results=[
                        WikipediaSearchResult(
                            title="British cuisine",
                            description="Overview of British cooking traditions",
                            url="https://en.wikipedia.org/wiki/British_cuisine",
                        )
                    ],
                    top_result_summary=WikipediaPageSummary(
                        title="British cuisine",
                        summary="British cuisine covers the cooking traditions of the United Kingdom.",
                        url="https://en.wikipedia.org/wiki/British_cuisine",
                        page_id=123,
                    ),
                ),
                "wikipedia_search",
            )
        },
    )

    bundle = build_evidence_bundle([iteration])

    record = bundle.hydrated_evidence_by_id["P1E2R1"]
    assert record.item_id == "https://en.wikipedia.org/wiki/British_cuisine"
    assert record.title == "British cuisine"
    assert record.summary == "British cuisine covers the cooking traditions of the United Kingdom."
    assert record.url == "https://en.wikipedia.org/wiki/British_cuisine"


def test_build_evidence_bundle_uses_wikipedia_top_result_summary_when_results_are_empty() -> None:
    iteration = IterationState(
        plan=Plan.model_validate(
            {
                "steps": [
                    {
                        "id": "E2",
                        "plan": "Search Wikipedia.",
                        "tool": "wikipedia_search",
                        "args": {"query": "staple foods of the United Kingdom"},
                    }
                ]
            }
        ),
        results={
            "P1E2": _tool_result(
                WikipediaSearchResponse(
                    query="staple foods of the United Kingdom",
                    results=[],
                    top_result_summary=WikipediaPageSummary(
                        title="British cuisine",
                        summary="British cuisine covers the cooking traditions of the United Kingdom.",
                        url="https://en.wikipedia.org/wiki/British_cuisine",
                        page_id=123,
                    ),
                ),
                "wikipedia_search",
            )
        },
    )

    bundle = build_evidence_bundle([iteration])

    record = bundle.hydrated_evidence_by_id["P1E2R1"]
    assert record.title == "British cuisine"
    assert record.summary == "British cuisine covers the cooking traditions of the United Kingdom."
    assert record.url == "https://en.wikipedia.org/wiki/British_cuisine"


def test_build_evidence_bundle_skips_empty_wikipedia_wrapper_without_results() -> None:
    iteration = IterationState(
        plan=Plan.model_validate(
            {
                "steps": [
                    {
                        "id": "E2",
                        "plan": "Search Wikipedia.",
                        "tool": "wikipedia_search",
                        "args": {"query": "British cuisine staple foods"},
                    }
                ]
            }
        ),
        results={
            "P1E2": _tool_result(
                WikipediaSearchResponse(
                    query="British cuisine staple foods",
                    results=[],
                    top_result_summary=None,
                ),
                "wikipedia_search",
            )
        },
    )

    bundle = build_evidence_bundle([iteration])

    assert bundle.hydrated_evidence_by_id == {}
    assert bundle.evidence_views_by_step_id == {}


def test_build_evidence_steps_merges_deck_results_into_one_group() -> None:
    iterations = [
        IterationState(
            plan=Plan.model_validate(
                {
                    "steps": [
                        {
                            "id": "E1",
                            "plan": "Look up one commander.",
                            "tool": "get_commander_details",
                            "args": {"commander_name": "Uril, the Miststalker"},
                        }
                    ]
                }
            ),
            results={
                "P1E1": ToolResult.model_validate(
                    {
                        "result": {"commander_name": "Uril, the Miststalker"},
                        "metadata": {"commander_slug": "uril-the-miststalker"},
                        "evidence_views": [
                            {
                                "item_id": "uril-the-miststalker",
                                "title": "Uril, the Miststalker (Commander)",
                                "summary": "Aura-focused Naya Voltron commander.",
                                "metadata": {"top_themes": "Auras, Voltron"},
                            }
                        ],
                        "hydrated_evidence": [
                            {
                                "item_id": "uril-the-miststalker",
                                "tool_name": "get_commander_details",
                                "title": "Uril, the Miststalker (Commander)",
                                "summary": "Aura-focused Naya Voltron commander.",
                                "source": "get_commander_details",
                                "entity_type": "decks",
                                "metadata": {"top_themes": "Auras, Voltron"},
                            }
                        ],
                    }
                )
            },
        ),
        IterationState(
            plan=Plan.model_validate(
                {
                    "steps": [
                        {
                            "id": "E1",
                            "plan": "Look up another commander.",
                            "tool": "get_commander_details",
                            "args": {"commander_name": "Sigarda, Host of Herons"},
                        }
                    ]
                }
            ),
            results={
                "P2E1": ToolResult.model_validate(
                    {
                        "result": {"commander_name": "Sigarda, Host of Herons"},
                        "metadata": {"commander_slug": "sigarda-host-of-herons"},
                        "evidence_views": [
                            {
                                "item_id": "sigarda-host-of-herons",
                                "title": "Sigarda, Host of Herons (Commander)",
                                "summary": "Hexproof Selesnya aura commander.",
                                "metadata": {"top_themes": "Auras, Enchantress"},
                            }
                        ],
                        "hydrated_evidence": [
                            {
                                "item_id": "sigarda-host-of-herons",
                                "tool_name": "get_commander_details",
                                "title": "Sigarda, Host of Herons (Commander)",
                                "summary": "Hexproof Selesnya aura commander.",
                                "source": "get_commander_details",
                                "entity_type": "decks",
                                "metadata": {"top_themes": "Auras, Enchantress"},
                            }
                        ],
                    }
                )
            },
        ),
    ]

    bundle = build_evidence_bundle(iterations)
    evidence_steps = build_evidence_steps(iterations, bundle.evidence_views_by_step_id)

    assert len(evidence_steps) == 1
    assert evidence_steps[0].type == "decks"
    assert evidence_steps[0].metadata == {
        "commander_slug": ["uril-the-miststalker", "sigarda-host-of-herons"],
    }
    assert [evidence.item_id for evidence in evidence_steps[0].evidence] == [
        "uril-the-miststalker",
        "sigarda-host-of-herons",
    ]


def test_build_evidence_steps_puts_wrapper_search_metadata_on_parent_step() -> None:
    iterations = [
        IterationState(
            plan=Plan.model_validate(
                {
                    "steps": [
                        {
                            "id": "E1",
                            "plan": "Search the web.",
                            "tool": "generic_web_search",
                            "args": {"query_text": "Toronto transit fares"},
                        }
                    ]
                }
            ),
            results={
                "P1E1": _tool_result(
                    WebSearchResponse.model_validate(
                        {
                            "query": "Toronto transit fares service major routes status TTC GO Transit Toronto",
                            "results": [
                                {
                                    "title": "TTC fares",
                                    "url": "https://example.com/ttc-fares",
                                    "description": "Fare details.",
                                }
                            ],
                            "retrieved_count": 20,
                            "reranked": True,
                        }
                    ),
                    "generic_web_search",
                )
            },
        )
    ]

    bundle = build_evidence_bundle(iterations)
    evidence_steps = build_evidence_steps(iterations, bundle.evidence_views_by_step_id)

    assert len(evidence_steps) == 1
    assert evidence_steps[0].metadata == {
        "query": "Toronto transit fares service major routes status TTC GO Transit Toronto",
        "retrieved_count": 20,
        "reranked": True,
        "search_type": "web_search",
    }
    assert evidence_steps[0].evidence[0].metadata == {}
