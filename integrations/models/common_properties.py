from typing import Optional

from pydantic import BaseModel, field_validator


class CommonProperties(BaseModel):
    color: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    gender: Optional[str] = None

    @field_validator("color", mode="before")
    @classmethod
    def normalize_color(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in {"none", "non", "n/a", "na", "null", "unknown", "any", "all", "no color"}:
            return None
        return text

    @field_validator("gender", mode="before")
    @classmethod
    def normalize_gender(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in {"none", "non", "unisex", "any", "all"}:
            return None
        if lowered in {"men", "man", "male", "mens"}:
            return "Men"
        if lowered in {"women", "woman", "female", "womens"}:
            return "Women"
        return None
