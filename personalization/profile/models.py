from __future__ import annotations

from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, Field, model_serializer

from personalization.user_attributes.models.user_attribute_models import UserAttribute


class GeoLocation(BaseModel):
    city: str | None = None
    region: str | None = None
    country: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    timezone: str | None = None


class GeoMetadata(BaseModel):
    current_datetime: str
    current_date: str
    current_weekday: str
    timezone: str
    location: GeoLocation | None = None


class UserAttributesSection(BaseModel):
    attributes: list[UserAttribute] = Field(default_factory=list)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "attributes": [
                {
                    key: value
                    for key, value in asdict(attribute).items()
                    if key != "attribute_embedding"
                }
                for attribute in self.attributes
            ]
        }

    @model_serializer(mode="plain")
    def serialize_model(self) -> dict[str, Any]:
        return self.to_prompt_dict()


class UserProfile(BaseModel):
    geometadata: GeoMetadata | None = None
    user_attributes: UserAttributesSection = Field(default_factory=UserAttributesSection)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "geometadata": None if self.geometadata is None else self.geometadata.model_dump(),
            "user_attributes": self.user_attributes.to_prompt_dict(),
        }

    @model_serializer(mode="plain")
    def serialize_model(self) -> dict[str, Any]:
        return self.to_prompt_dict()
