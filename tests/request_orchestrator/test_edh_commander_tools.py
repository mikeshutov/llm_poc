from __future__ import annotations

import importlib

from integrations.edhrec.models import EdhrecCommanderPage
from request_orchestrator.models.evidence import ToolResult
from request_orchestrator.shared.tool_adapter.games.get_commander_cards import get_commander_cards
from request_orchestrator.shared.tool_adapter.games.get_commander_details import get_commander_details


def _build_page() -> EdhrecCommanderPage:
    return EdhrecCommanderPage.model_validate(
        {
            "header": "Uril, the Miststalker (Commander)",
            "container": {
                "title": "Uril, the Miststalker (Commander)",
                "description": "Popular decks and cards for Uril, the Miststalker",
                "json_dict": {
                    "cardlists": [
                        {
                            "header": "High Synergy Cards",
                            "tag": "highsynergycards",
                            "cardviews": [
                                {
                                    "id": "card-a",
                                    "name": "Ethereal Armor",
                                    "sanitized": "ethereal-armor",
                                    "slug": "ethereal-armor",
                                    "url": "/cards/ethereal-armor",
                                    "synergy": 0.80,
                                    "num_decks": 4832,
                                    "potential_decks": 5852,
                                    "trend_zscore": -0.32,
                                },
                                {
                                    "id": "card-b",
                                    "name": "Rancor",
                                    "sanitized": "rancor",
                                    "slug": "rancor",
                                    "url": "/cards/rancor",
                                    "synergy": 0.75,
                                    "num_decks": 4626,
                                    "potential_decks": 5852,
                                    "trend_zscore": 0.22,
                                },
                            ],
                        }
                    ]
                },
            },
            "similar": [
                "Mazzy, Truesword Paladin",
                "Sigarda, Host of Herons",
            ],
            "creature": 17,
            "artifact": 4,
            "enchantment": 30,
            "instant": 6,
            "sorcery": 7,
            "land": 35,
            "panels": {
                "taglinks": [
                    {"count": 1218, "slug": "auras", "value": "Auras"},
                    {"count": 575, "slug": "voltron", "value": "Voltron"},
                ],
                "combocounts": [
                    {
                        "value": "Aggravated Assault + Bear Umbra",
                        "alt": "Aggravated Assault + Bear Umbra",
                        "href": "/combos/gruul/3750-4228",
                    },
                    {
                        "value": "See More...",
                        "alt": "See More...",
                        "href": "/combos/uril-the-miststalker",
                    },
                ],
            },
        }
    )


def test_get_commander_details_returns_typed_result() -> None:
    class FakeEdhrecClient:
        def get_commander_page(self, commander_name: str) -> tuple[str, EdhrecCommanderPage]:
            assert commander_name == "Uril, the Miststalker"
            return "uril-the-miststalker", _build_page()

    module = importlib.import_module(
        "request_orchestrator.shared.tool_adapter.games.get_commander_details"
    )
    original_client = module._edhrec_client
    module._edhrec_client = FakeEdhrecClient()
    try:
        result = get_commander_details.invoke({"commander_name": "Uril, the Miststalker"})
    finally:
        module._edhrec_client = original_client

    assert isinstance(result, ToolResult)
    assert result.result.model_dump() == {
        "query": "Uril, the Miststalker",
        "commander_name": "Uril, the Miststalker",
        "commander_slug": "uril-the-miststalker",
        "page_url": "https://edhrec.com/commanders/uril-the-miststalker",
        "title": "Uril, the Miststalker (Commander)",
        "description": "Popular decks and cards for Uril, the Miststalker",
        "top_themes": "Auras, Voltron",
        "combo_highlights": ["Aggravated Assault + Bear Umbra"],
        "similar_commanders": ["Mazzy, Truesword Paladin", "Sigarda, Host of Herons"],
    }
    assert result.evidence_views[0].evidence_object == result.result.model_dump()


def test_get_commander_cards_returns_reranked_card_evidence() -> None:
    class FakeEdhrecClient:
        def get_commander_page(self, commander_name: str) -> tuple[str, EdhrecCommanderPage]:
            assert commander_name == "Uril, the Miststalker"
            return "uril-the-miststalker", _build_page()

    module = importlib.import_module(
        "request_orchestrator.shared.tool_adapter.games.get_commander_cards"
    )
    original_client = module._edhrec_client
    original_rerank = module.rerank_edhrec_cards
    module._edhrec_client = FakeEdhrecClient()
    module.rerank_edhrec_cards = lambda cards, **kwargs: [cards[1], cards[0]]
    try:
        result = get_commander_cards.invoke({"commander_name": "Uril, the Miststalker", "limit": 2})
    finally:
        module._edhrec_client = original_client
        module.rerank_edhrec_cards = original_rerank

    assert isinstance(result, ToolResult)
    assert result.result.model_dump() == {
        "query": "Uril, the Miststalker",
        "commander_name": "Uril, the Miststalker",
        "commander_slug": "uril-the-miststalker",
        "returned_count": 2,
        "cards": [
            {
                "name": "Rancor",
                "slug": "rancor",
                "card_url": "https://edhrec.com/cards/rancor",
                "section": "High Synergy Cards",
                "synergy": 0.75,
                "num_decks": 4626,
                "potential_decks": 5852,
                "trend_zscore": 0.22,
            },
            {
                "name": "Ethereal Armor",
                "slug": "ethereal-armor",
                "card_url": "https://edhrec.com/cards/ethereal-armor",
                "section": "High Synergy Cards",
                "synergy": 0.8,
                "num_decks": 4832,
                "potential_decks": 5852,
                "trend_zscore": -0.32,
            },
        ],
    }
    assert len(result.evidence_views) == 2
    assert result.evidence_views[0].title == "Rancor"
    assert result.evidence_views[0].evidence_object == result.result.cards[0].model_dump()
    assert result.hydrated_evidence[0].urls[0].url == "https://edhrec.com/cards/rancor"
