from typing import Optional

from pydantic import BaseModel, field_validator


class ProductQuery(BaseModel):
    category: Optional[list[str]] = None
    style: Optional[str] = None
    size_label: Optional[str] = None
    size_numeric: Optional[float] = None
    size_unit: Optional[str] = None
    color: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    gender: Optional[str] = None

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value: object) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("color", mode="before")
    @classmethod
    def normalize_color(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.lower() in {"none", "non", "n/a", "na", "null", "unknown", "any", "all", "no color"}:
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
