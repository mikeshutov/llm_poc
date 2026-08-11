from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_serializer

from common.serialization import prune_empty_prompt_values
from personalization.user_attributes.models.user_attribute_models import UserAttribute

ATTRIBUTE_PROMPT_EXCLUDED_FIELDS = {
    "attribute_embedding",
    "user_id",
    "confidence",
    "importance",
    "id",
    "group_key",
    "source",
    "is_active",
    "created_at",
    "updated_at",
}

ATTRIBUTE_PROMPT_MANAGEMENT_EXCLUDED_FIELDS = {
    "attribute_embedding",
    "user_id",
    "confidence",
    "importance",
}


class GeoLocation(BaseModel):
    city: str | None = None
    region: str | None = None
    country: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    timezone: str | None = None


class GeoMetadata(BaseModel):
    current_datetime: str
    current_weekday: str
    timezone: str
    location: GeoLocation | None = None


class UserAttributesSection(BaseModel):
    attributes: list[UserAttribute] = Field(default_factory=list)

    @field_validator("attributes", mode="before")
    @classmethod
    def _coerce_attributes(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value

        coerced: list[Any] = []
        for item in value:
            if isinstance(item, UserAttribute):
                coerced.append(item)
                continue
            if is_dataclass(item):
                coerced.append(asdict(item))
                continue
            coerced.append(item)
        return coerced

    def to_prompt_dict(self, *, include_management_fields: bool = False) -> dict[str, Any]:
        excluded_fields = (
            ATTRIBUTE_PROMPT_MANAGEMENT_EXCLUDED_FIELDS
            if include_management_fields
            else ATTRIBUTE_PROMPT_EXCLUDED_FIELDS
        )
        return prune_empty_prompt_values(
            {
                "attributes": [
                    {
                        key: value
                        for key, value in asdict(attribute).items()
                        if key not in excluded_fields
                    }
                    for attribute in self.attributes
                ]
            }
        )

    @model_serializer(mode="plain")
    def serialize_model(self) -> dict[str, Any]:
        return self.to_prompt_dict()


class UserProfile(BaseModel):
    user_id: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    display_name: str | None = None
    email: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    geometadata: GeoMetadata | None = None
    user_attributes: UserAttributesSection = Field(default_factory=UserAttributesSection)

    def to_prompt_dict(self, include_management_fields: bool = False) -> dict[str, Any]:
        return prune_empty_prompt_values(
            {
                "user_id": self.user_id,
                "first_name": self.first_name,
                "last_name": self.last_name,
                "display_name": self.display_name,
                "email": self.email,
                "geometadata": None if self.geometadata is None else self.geometadata.model_dump(),
                "user_attributes": self.user_attributes.to_prompt_dict(
                    include_management_fields=include_management_fields,
                ),
            }
        )

    @model_serializer(mode="plain")
    def serialize_model(self) -> dict[str, Any]:
        return self.to_prompt_dict()
