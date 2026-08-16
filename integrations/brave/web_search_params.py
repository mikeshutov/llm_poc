from typing import Any

import pycountry
from pydantic import BaseModel, Field, field_validator


class WebSearchParams(BaseModel):
    q: str
    count: int = 10
    offset: int = 0
    country: str = "ALL"
    search_lang: str = "en"
    extra_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("country", mode="before")
    @classmethod
    def normalize_country(cls, v: str) -> str:
        try:
            return pycountry.countries.lookup(v).alpha_2
        except LookupError:
            return "ALL"

    def to_api_params(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "count": self.count,
            "offset": self.offset,
            "country": self.country,
            "search_lang": self.search_lang,
            **self.extra_params,
        }
