from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JobSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    days_of_week: list[int] = Field(min_length=1)
    run_time: time
    timezone: str

    @field_validator("days_of_week")
    @classmethod
    def validate_days(cls, value: list[int]) -> list[int]:
        normalized = sorted(set(value))
        if not normalized or any(day < 1 or day > 7 for day in normalized):
            raise ValueError("days_of_week must contain ISO weekdays from 1 through 7")
        return normalized

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("timezone is required")
        try:
            ZoneInfo(normalized)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {normalized}") from exc
        return normalized
