from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class IpLocation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: str
    country: Optional[str] = None
    country_code: Optional[str] = Field(None, alias="countryCode")
    region: Optional[str] = None
    region_name: Optional[str] = Field(None, alias="regionName")
    city: Optional[str] = None
    zip: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    timezone: Optional[str] = None
    isp: Optional[str] = None
    org: Optional[str] = None
    query: Optional[str] = None
