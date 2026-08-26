from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from common.html_text import html_to_plain_text


class CandidateContent(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str | None = None
    summary: str | None = None
    description: str | None = None
    text: str | None = None
    url: str | None = None
    image_url: str | None = None

    @field_validator("name", "summary", "description", "text", mode="before")
    @classmethod
    def normalize_reranker_text(cls, value: object) -> object:
        return html_to_plain_text(value) if isinstance(value, str) else value

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            value = getattr(self, key)
            return default if value is None else value
        if self.model_extra and key in self.model_extra:
            value = self.model_extra[key]
            return default if value is None else value
        return default

    def __getitem__(self, key: str) -> Any:
        value = self.get(key)
        if value is None and not (self.model_extra and key in self.model_extra) and not hasattr(self, key):
            raise KeyError(key)
        return value

    def __contains__(self, key: str) -> bool:
        if key in self.model_fields_set:
            return True
        return bool(self.model_extra and key in self.model_extra)
