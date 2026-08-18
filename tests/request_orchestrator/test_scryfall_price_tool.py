from __future__ import annotations

import importlib

from request_orchestrator.models.evidence import ToolResult
from request_orchestrator.shared.tool_adapter.games.get_magic_card_price import get_magic_card_price


def test_get_magic_card_price_returns_pricing_result() -> None:
    class FakeScryfallClient:
        def get_card_by_name(self, name: str, *, fuzzy: bool = True):
            assert name == "Black Lotus"
            assert fuzzy is True
            return type(
                "FakeCard",
                (),
                {
                    "id": "card-1",
                    "name": "Black Lotus",
                    "set_name": "Vintage Masters",
                    "rarity": "bonus",
                    "scryfall_uri": "https://scryfall.com/card/vma/4/black-lotus",
                    "prices": {
                        "usd": "12345.67",
                        "usd_foil": "23456.78",
                        "usd_etched": None,
                        "eur": "11111.11",
                        "eur_foil": None,
                        "tix": "999.99",
                    },
                    "image_uris": type(
                        "FakeImageUris",
                        (),
                        {
                            "normal": "https://cards.scryfall.io/normal/front/b/l/black-lotus.jpg",
                            "large": None,
                            "small": None,
                        },
                    )(),
                    "card_faces": None,
                },
            )()

    module = importlib.import_module(
        "request_orchestrator.shared.tool_adapter.games.get_magic_card_price"
    )
    original_client = module._scryfall_client
    module._scryfall_client = FakeScryfallClient()
    try:
        result = get_magic_card_price.invoke({"card_name": "Black Lotus"})
    finally:
        module._scryfall_client = original_client

    assert isinstance(result, ToolResult)
    assert result.result.model_dump() == {
        "id": "card-1",
        "name": "Black Lotus",
        "set_name": "Vintage Masters",
        "rarity": "bonus",
        "scryfall_uri": "https://scryfall.com/card/vma/4/black-lotus",
        "image_url": "https://cards.scryfall.io/normal/front/b/l/black-lotus.jpg",
        "usd": "12345.67",
        "usd_foil": "23456.78",
        "usd_etched": None,
        "eur": "11111.11",
        "eur_foil": None,
        "tix": "999.99",
    }
    assert result.evidence_views[0].title == "Black Lotus Price"
    assert result.evidence_views[0].metadata["usd"] == "12345.67"
    assert result.hydrated_evidence[0].urls[0].url == "https://scryfall.com/card/vma/4/black-lotus"
