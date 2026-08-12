from __future__ import annotations

import sys
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
from request_orchestrator.models.agent_state import IterationState
from request_orchestrator.models.plan import Plan
from request_orchestrator.shared.evidence import build_evidence_bundle
from integrations.open_meteo.models import CurrentWeather, GeocodedLocation
from request_orchestrator.shared.tool_adapter.weather.get_current_weather import CurrentWeatherResult


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
            "P1E1": NewsSearchResponse.model_validate(
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
    assert first.image_url == ""
    assert first.source == "news_search"
    assert first.entity_type == "generic_result"


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
            "P1E1": SparqlResult(
                sparql="SELECT * WHERE {}",
                vars=["qid", "itemLabel", "url"],
                bindings=[
                    {
                        "qid": "Q172",
                        "itemLabel": "Toronto",
                        "url": "https://www.wikidata.org/wiki/Q172",
                    }
                ],
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
            "P1E3": SparqlResult(
                sparql="SELECT ?item ?itemLabel WHERE {}",
                vars=["item", "itemLabel"],
                bindings=[
                    {
                        "item": {"type": "uri", "value": "http://www.wikidata.org/entity/Q172"},
                        "itemLabel": {"xml:lang": "en", "type": "literal", "value": "Toronto"},
                    }
                ],
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
            "P1E1": CurrentWeatherResult(
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
            )
        },
    )

    bundle = build_evidence_bundle([iteration])

    record = bundle.hydrated_evidence_by_id["P1E1R1"]
    assert record.title == "Get Current Weather"
    assert record.item_id == "Toronto"
    assert record.location_name == "Toronto"
    assert record.url == ""
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
            "P1E1": WebSearchResponse.model_validate(
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
            "P1E2": {"id": "fallback-1", "description": "A generic fallback record."},
        },
    )

    bundle = build_evidence_bundle([iteration])

    web_record = bundle.hydrated_evidence_by_id["P1E1R1"]
    assert web_record.item_id == "https://example.com/ramen"
    assert web_record.title == "Ramen spot"
    assert web_record.summary == "Popular local ramen shop."
    assert web_record.image_url == "https://example.com/ramen.jpg"
    assert web_record.source == "generic_web_search"
    assert web_record.entity_type == "generic_result"

    fallback_record = bundle.hydrated_evidence_by_id["P1E2R1"]
    assert fallback_record.item_id == "fallback-1"
    assert fallback_record.title == "Custom Lookup"
    assert fallback_record.summary == "A generic fallback record."
    assert fallback_record.url == ""
    assert fallback_record.urls == []


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
            "P1E4": MealSearchResult.model_validate(
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
            )
        },
    )

    bundle = build_evidence_bundle([iteration])

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
        "ingredients": [
            {"name": "Butter", "measure": "1/2 cup"},
            {"name": "Sugar", "measure": "1 cup"},
        ]
    }
    assert bundle.evidence_views_by_step_id["P1E4"][0].metadata == {
        "ingredients": [
            {"name": "Butter", "measure": "1/2 cup"},
            {"name": "Sugar", "measure": "1 cup"},
        ]
    }
