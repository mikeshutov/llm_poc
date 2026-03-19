from __future__ import annotations

from pydantic import BaseModel


class WorldTime(BaseModel):
    timezone: str
    datetime: str
    utc_offset: str
    day_of_week: int
    abbreviation: str
