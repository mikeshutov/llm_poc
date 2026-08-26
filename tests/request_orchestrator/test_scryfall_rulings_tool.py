from __future__ import annotations

import importlib

from integrations.scryfall.models import ScryfallCard, ScryfallRulingList
from request_orchestrator.models.evidence import ToolResult
from request_orchestrator.shared.tool_adapter.games.get_magic_card_rulings import get_magic_card_rulings


def test_get_magic_card_rulings_returns_typed_result() -> None:
    class FakeScryfallClient:
        def get_card_by_name(self, name: str, *, fuzzy: bool = True) -> ScryfallCard:
            assert name == "Humility"
            assert fuzzy is True
            return ScryfallCard.model_validate(
                {
                    "id": "card-1",
                    "name": "Humility",
                    "type_line": "Enchantment",
                    "oracle_text": "All creatures lose all abilities and have base power and toughness 1/1.",
                    "scryfall_uri": "https://scryfall.com/card/tmp/14/humility",
                    "set_name": "Tempest",
                    "rarity": "rare",
                    "legalities": {"commander": "legal", "legacy": "legal", "modern": "not_legal"},
                }
            )

        def get_card_rulings(self, card_id: str) -> ScryfallRulingList:
            assert card_id == "card-1"
            return ScryfallRulingList.model_validate(
                {
                    "object": "list",
                    "has_more": False,
                    "data": [
                        {
                            "oracle_id": "oracle-1",
                            "source": "wotc",
                            "published_at": "2004-10-04",
                            "comment": "Humility's effect applies in layer 6 and 7b.",
                        },
                        {
                            "oracle_id": "oracle-1",
                            "source": "wotc",
                            "published_at": "2020-08-07",
                            "comment": "Effects that set power and toughness later may overwrite Humility's 1/1 effect.",
                        },
                    ],
                }
            )

    module = importlib.import_module(
        "request_orchestrator.shared.tool_adapter.games.get_magic_card_rulings"
    )
    original_client = module._scryfall_client
    module._scryfall_client = FakeScryfallClient()
    try:
        result = get_magic_card_rulings.invoke({"card_name": "Humility"})
    finally:
        module._scryfall_client = original_client

    assert isinstance(result, ToolResult)
    assert result.tool_metadata.model_dump(exclude_none=True) == {"ruling_source": "wotc"}
    assert result.result.model_dump() == {
        "resolved_card": {
            "id": "card-1",
            "name": "Humility",
            "type_line": "Enchantment",
            "oracle_text": "All creatures lose all abilities and have base power and toughness 1/1.",
            "scryfall_uri": "https://scryfall.com/card/tmp/14/humility",
            "set_name": "Tempest",
            "rarity": "rare",
            "legal_formats": ["commander", "legacy"],
        },
        "ruling_count": 2,
        "rulings": [
            {
                "oracle_id": "oracle-1",
                "source": "wotc",
                "published_at": "2004-10-04",
                "comment": "Humility's effect applies in layer 6 and 7b.",
            },
            {
                "oracle_id": "oracle-1",
                "source": "wotc",
                "published_at": "2020-08-07",
                "comment": "Effects that set power and toughness later may overwrite Humility's 1/1 effect.",
            },
        ],
    }
    assert len(result.evidence) == 2
    assert result.evidence[0].title == "Humility Ruling 1"
    assert "query" not in result.evidence[0].llm_metadata
    assert result.evidence[0].llm_metadata["legal_formats"] == ["commander", "legacy"]
    assert not {"card_id", "card_name", "set_name", "rarity", "oracle_id"} & result.evidence[0].llm_metadata.keys()
    assert result.evidence[0].published_at == "2004-10-04"
    assert result.evidence[0].urls[0].url == "https://scryfall.com/card/tmp/14/humility"
