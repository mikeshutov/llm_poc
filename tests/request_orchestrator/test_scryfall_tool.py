from __future__ import annotations

import importlib

from integrations.scryfall.models import MagicCardPriceEntry, ScryfallCardSearchResult
from request_orchestrator.models.evidence import ToolResult
from request_orchestrator.shared.tool_adapter.games.mtg_color_identity import apply_commander_color_identity_filter
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
        result = search_magic_cards.invoke({"query": "Black Lotus"})
    finally:
        module._scryfall_client = original_client

    assert isinstance(result, ToolResult)
    assert result.metadata == {
        "page": 1,
        "page_size": 15,
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
                "color_identity": [],
                "image_url": "https://cards.scryfall.io/normal/front/b/l/black-lotus.jpg",
                "scryfall_uri": "https://scryfall.com/card/vma/4/black-lotus",
                "set_name": "Vintage Masters",
                "rarity": "bonus",
                "pricing": [],
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
                "page": 1,
            }
        )
    finally:
        module._scryfall_client = original_client

    assert isinstance(result, ToolResult)
    assert result.metadata == {
        "page": 1,
        "page_size": 15,
        "has_more": False,
        "returned_count": 0,
        "total_cards": 0,
        "warnings": [],
    }
    assert result.result.cards == []


def test_apply_commander_color_identity_filter_normalizes_aliases() -> None:
    assert apply_commander_color_identity_filter("type:creature", "abzan") == "id<=wbg type:creature"
    assert apply_commander_color_identity_filter("type:creature", "naya") == "id<=rgw type:creature"
    assert apply_commander_color_identity_filter("type:creature", "wubrg") == "id<=wubrg type:creature"


def test_apply_commander_color_identity_filter_preserves_existing_filter() -> None:
    assert apply_commander_color_identity_filter("id<=wug type:instant", "abzan") == "id<=wug type:instant"


def test_search_magic_cards_can_include_pricing() -> None:
    class FakeScryfallClient:
        def search_cards(
            self,
            query: str,
            *,
            unique: str = "cards",
            order: str = "name",
            dir: str = "auto",
        ) -> ScryfallCardSearchResult:
            if query == "Black Lotus":
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
                                "color_identity": [],
                                "scryfall_uri": "https://scryfall.com/card/vma/4/black-lotus",
                                "set_name": "Vintage Masters",
                                "rarity": "bonus",
                                "games": ["paper"],
                                "prices": {"usd": "12345.67", "usd_foil": "23456.78", "eur": "11111.11", "tix": "999.99"},
                                "legalities": {"commander": "not_legal"},
                                "image_uris": {
                                    "normal": "https://cards.scryfall.io/normal/front/b/l/black-lotus.jpg",
                                },
                            }
                        ],
                    }
                )
            assert query == '!"Black Lotus"'
            assert unique == "prints"
            assert order == "released"
            assert dir == "desc"
            return ScryfallCardSearchResult.model_validate(
                {
                    "object": "list",
                    "total_cards": 2,
                    "has_more": False,
                    "data": [
                        {
                            "id": "card-1",
                            "name": "Black Lotus",
                            "scryfall_uri": "https://scryfall.com/card/vma/4/black-lotus",
                            "set_name": "Vintage Masters",
                            "rarity": "bonus",
                            "prices": {"usd": "12345.67", "usd_foil": "23456.78", "eur": "11111.11", "tix": "999.99"},
                            "image_uris": {"normal": "https://cards.scryfall.io/normal/front/b/l/black-lotus.jpg"},
                        },
                        {
                            "id": "card-2",
                            "name": "Black Lotus",
                            "scryfall_uri": "https://scryfall.com/card/ced/233/black-lotus",
                            "set_name": "Collectors' Edition",
                            "rarity": "rare",
                            "prices": {"usd": "10000.00"},
                            "image_uris": {"normal": "https://cards.scryfall.io/normal/front/c/e/black-lotus.jpg"},
                        },
                    ],
                }
            )

    module = importlib.import_module(
        "request_orchestrator.shared.tool_adapter.games.search_magic_cards"
    )
    original_client = module._scryfall_client
    module._scryfall_client = FakeScryfallClient()
    try:
        result = search_magic_cards.invoke({"query": "Black Lotus", "include_pricing": True})
    finally:
        module._scryfall_client = original_client

    assert isinstance(result, ToolResult)
    assert result.result.cards[0].pricing == [
        MagicCardPriceEntry(
            set_name="Vintage Masters",
            scryfall_uri="https://scryfall.com/card/vma/4/black-lotus",
            image_url="https://cards.scryfall.io/normal/front/b/l/black-lotus.jpg",
            usd="12345.67",
            usd_foil="23456.78",
            usd_etched=None,
            eur="11111.11",
            eur_foil=None,
            tix="999.99",
        ),
        MagicCardPriceEntry(
            set_name="Collectors' Edition",
            scryfall_uri="https://scryfall.com/card/ced/233/black-lotus",
            image_url="https://cards.scryfall.io/normal/front/c/e/black-lotus.jpg",
            usd="10000.00",
            usd_foil=None,
            usd_etched=None,
            eur=None,
            eur_foil=None,
            tix=None,
        ),
    ]
    assert result.evidence_views[0].metadata["pricing"] == [
        {
            "set": "Vintage Masters",
            "usd": "12345.67",
            "usd_foil": "23456.78",
            "eur": "11111.11",
            "magic_online": "999.99",
        },
        {
            "set": "Collectors' Edition",
            "usd": "10000.00",
        },
    ]


def test_search_magic_cards_supports_internal_pagination() -> None:
    class FakeScryfallClient:
        def search_cards(self, query: str) -> ScryfallCardSearchResult:
            assert query == "Knight of the Reliquary"
            return ScryfallCardSearchResult.model_validate(
                {
                    "object": "list",
                    "total_cards": 20,
                    "has_more": False,
                    "data": [
                        {
                            "id": f"card-{index}",
                            "name": f"Card {index}",
                            "type_line": "Creature",
                            "color_identity": ["G", "W"],
                            "games": ["paper"],
                            "prices": {"usd": None, "usd_foil": None},
                            "legalities": {"commander": "legal"},
                        }
                        for index in range(1, 21)
                    ],
                }
            )

    module = importlib.import_module(
        "request_orchestrator.shared.tool_adapter.games.search_magic_cards"
    )
    original_client = module._scryfall_client
    module._scryfall_client = FakeScryfallClient()
    try:
        result = search_magic_cards.invoke({"query": "Knight of the Reliquary", "page": 2})
    finally:
        module._scryfall_client = original_client

    assert result.metadata == {
        "page": 2,
        "page_size": 15,
        "has_more": False,
        "returned_count": 5,
        "total_cards": 20,
        "warnings": [],
    }
    assert [card.id for card in result.result.cards] == [
        "card-16",
        "card-17",
        "card-18",
        "card-19",
        "card-20",
    ]
