from __future__ import annotations

import importlib

from integrations.scryfall.models import ScryfallCardSearchResult
from request_orchestrator.models.evidence import ToolResult
from request_orchestrator.shared.tool_adapter.games.search_magic_cards import search_magic_cards


def test_search_magic_cards_returns_typed_result() -> None:
    class FakeScryfallClient:
        def search_cards(self, query: str) -> ScryfallCardSearchResult:
            assert query == "Black Lotus"
            return ScryfallCardSearchResult.model_validate(
                {
                    "object": "list",
                    "total_cards": 1,
                    "has_more": False,
                    "data": [
                        {
                            "id": "card-1",
                            "name": "Black Lotus",
                            "mana_cost": "{0}",
                            "cmc": 0,
                            "type_line": "Artifact",
                            "oracle_text": "{T}, Sacrifice Black Lotus: Add three mana of any one color.",
                            "colors": [],
                            "color_identity": [],
                            "scryfall_uri": "https://scryfall.com/card/vma/4/black-lotus",
                            "set_name": "Vintage Masters",
                            "rarity": "bonus",
                            "games": ["paper"],
                            "prices": {"usd": None, "usd_foil": None},
                            "legalities": {"commander": "not_legal"},
                            "image_uris": {
                                "small": "https://cards.scryfall.io/small/front/b/l/black-lotus.jpg",
                                "normal": "https://cards.scryfall.io/normal/front/b/l/black-lotus.jpg",
                            },
                        }
                    ],
                }
            )

    module = importlib.import_module(
        "request_orchestrator.shared.tool_adapter.games.search_magic_cards"
    )
    original_client = module._scryfall_client
    module._scryfall_client = FakeScryfallClient()
    try:
        result = search_magic_cards.invoke({"query": "Black Lotus", "limit": 3})
    finally:
        module._scryfall_client = original_client

    assert isinstance(result, ToolResult)
    assert result.metadata == {
        "has_more": False,
        "returned_count": 1,
        "total_cards": 1,
        "warnings": [],
    }
    assert result.result.model_dump() == {
        "total_cards": 1,
        "has_more": False,
        "returned_count": 1,
        "warnings": [],
        "cards": [
            {
                "id": "card-1",
                "name": "Black Lotus",
                "mana_cost": "{0}",
                "cmc": 0.0,
                "type_line": "Artifact",
                "oracle_text": "{T}, Sacrifice Black Lotus: Add three mana of any one color.",
                "colors": [],
                "color_identity": [],
                "image_url": "https://cards.scryfall.io/normal/front/b/l/black-lotus.jpg",
                "scryfall_uri": "https://scryfall.com/card/vma/4/black-lotus",
                "set_name": "Vintage Masters",
                "rarity": "bonus",
            }
        ],
    }
    assert result.evidence_views[0].item_id == "card-1"
    assert result.evidence_views[0].title == "Black Lotus"
    assert "Artifact" in result.evidence_views[0].summary
    assert "prices" not in result.evidence_views[0].metadata
    assert "legalities" not in result.evidence_views[0].metadata
    assert "games" not in result.evidence_views[0].metadata
    assert "query" not in result.evidence_views[0].metadata
    assert result.hydrated_evidence[0].item_id == "card-1"
    assert result.hydrated_evidence[0].urls[0].url == "https://scryfall.com/card/vma/4/black-lotus"


def test_search_magic_cards_applies_commander_color_identity_filter() -> None:
    class FakeScryfallClient:
        def search_cards(self, query: str) -> ScryfallCardSearchResult:
            assert query == 'id<=wbg type:instant (o:land or o:"sacrifice a creature")'
            return ScryfallCardSearchResult.model_validate(
                {
                    "object": "list",
                    "total_cards": 0,
                    "has_more": False,
                    "data": [],
                }
            )

    module = importlib.import_module(
        "request_orchestrator.shared.tool_adapter.games.search_magic_cards"
    )
    original_client = module._scryfall_client
    module._scryfall_client = FakeScryfallClient()
    try:
        result = search_magic_cards.invoke(
            {
                "query": 'type:instant (o:land or o:"sacrifice a creature")',
                "commander_color_identity": "abzan",
                "limit": 5,
            }
        )
    finally:
        module._scryfall_client = original_client

    assert isinstance(result, ToolResult)
    assert result.metadata == {
        "has_more": False,
        "returned_count": 0,
        "total_cards": 0,
        "warnings": [],
    }
    assert result.result.cards == []
