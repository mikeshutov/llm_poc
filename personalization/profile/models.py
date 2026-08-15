from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator

from common.data import prune_empty_prompt_values
from personalization.tone.models import TonePreferences
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


class PromptGeoLocation(BaseModel):
    city: str | None = None
    region: str | None = None
    country: str | None = None


class PromptGeoMetadata(BaseModel):
    current_datetime: str
    current_weekday: str
    timezone: str
    location: PromptGeoLocation | None = None


class GeoLocation(BaseModel):
    city: str | None = None
    region: str | None = None
    country: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    timezone: str | None = None

    def to_prompt_model(self) -> PromptGeoLocation:
        return PromptGeoLocation(
            city=self.city,
            region=self.region,
            country=self.country,
        )


class GeoMetadata(BaseModel):
    current_datetime: str
    current_weekday: str
    timezone: str
    location: GeoLocation | None = None

    def to_prompt_model(self) -> PromptGeoMetadata:
        return PromptGeoMetadata(
            current_datetime=self.current_datetime,
            current_weekday=self.current_weekday,
            timezone=self.timezone,
            location=None if self.location is None else self.location.to_prompt_model(),
        )


def build_geometadata(
    *,
    timezone: str | None = "America/Toronto",
    location: GeoLocation | None = None,
) -> GeoMetadata:
    resolved_timezone = (timezone or "").strip()
    if not resolved_timezone and location is not None:
        resolved_timezone = (location.timezone or "").strip()
    if not resolved_timezone:
        resolved_timezone = "America/Toronto"

    now = datetime.now(ZoneInfo(resolved_timezone))
    return GeoMetadata(
        current_datetime=now.isoformat(),
        current_weekday=now.strftime("%A"),
        timezone=resolved_timezone,
        location=location,
    )


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

class UserProfile(BaseModel):
    user_id: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    display_name: str | None = None
    email: str | None = None
    tone: TonePreferences | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    geometadata: GeoMetadata | None = None
    user_attributes: UserAttributesSection = Field(default_factory=UserAttributesSection)

    def to_prompt_dict(
        self,
        include_management_fields: bool = False,
        *,
        include_tone: bool = False,
    ) -> dict[str, Any]:
        return prune_empty_prompt_values(
            {
                "first_name": self.first_name,
                "last_name": self.last_name,
                "display_name": self.display_name,
                "email": self.email,
                "tone": None if not include_tone or self.tone is None else self.tone.model_dump(),
                "geometadata": None if self.geometadata is None else prune_empty_prompt_values(self.geometadata.to_prompt_model().model_dump()),
                "user_attributes": self.user_attributes.to_prompt_dict(
                    include_management_fields=include_management_fields,
                ),
            }
        )

