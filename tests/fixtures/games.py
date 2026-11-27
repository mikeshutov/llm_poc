from copy import deepcopy

import pytest

from integrations.edhrec.models import EdhrecCommanderPage


@pytest.fixture
def commander_page_factory():
    def create(**overrides):
        payload = {
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
        return EdhrecCommanderPage.model_validate({**deepcopy(payload), **deepcopy(overrides)})

    return create
