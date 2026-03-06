from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class PublicHoliday(BaseModel):
    date: str
    local_name: str
    name: str
    country_code: str
    fixed: Optional[bool] = None
    global_holiday: Optional[bool] = None
    counties: Optional[list[str]] = None
    launch_year: Optional[int] = None
    types: Optional[list[str]] = None
