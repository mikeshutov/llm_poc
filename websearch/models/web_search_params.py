from typing import Any

from pydantic import BaseModel, Field


class WebSearchParams(BaseModel):
    q: str
    count: int = 10
    offset: int = 0
    country: str = "CA"
    search_lang: str = "en"
    extra_params: dict[str, Any] = Field(default_factory=dict)

    def to_api_params(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "count": self.count,
            "offset": self.offset,
            "country": self.country,
            "search_lang": self.search_lang,
            **self.extra_params,
        }
