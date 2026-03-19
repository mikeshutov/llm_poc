from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Country(BaseModel):
    common_name: str
    official_name: str
    capital: list[str]
    region: str
    subregion: str
    population: int
    currencies: dict[str, Any]
    languages: dict[str, str]
    flag: str
