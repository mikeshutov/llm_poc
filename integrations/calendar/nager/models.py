from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PublicHoliday(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    date: str
    local_name: str = Field(alias="localName")
    name: str
    country_code: str = Field(alias="countryCode")
    fixed: Optional[bool] = None
    global_holiday: Optional[bool] = Field(default=None, alias="global")
    counties: Optional[list[str]] = None
    launch_year: Optional[int] = Field(default=None, alias="launchYear")
    types: Optional[list[str]] = None
