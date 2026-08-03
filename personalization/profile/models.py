from __future__ import annotations

from pydantic import BaseModel, Field

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


class UserProfile(BaseModel):
    geometadata: GeoMetadata | None = None
    user_attributes: UserAttributesSection = Field(default_factory=UserAttributesSection)
