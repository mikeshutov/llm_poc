from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class DailyWeather(BaseModel):
    date: list[str] = []
    temperature_2m_max: list[Optional[float]] = []
    temperature_2m_min: list[Optional[float]] = []
    precipitation_sum: list[Optional[float]] = []
    windspeed_10m_max: list[Optional[float]] = []


class GeocodedLocation(BaseModel):
    name: str
    country: str
    latitude: float
    longitude: float
    timezone: Optional[str] = None


class MonthlyWeatherSummary(BaseModel):
    city: str
    country: str
    year: int
    month: int
    days_count: int
    avg_temp_max_c: Optional[float] = None
    avg_temp_min_c: Optional[float] = None
    total_precip_mm: Optional[float] = None
    avg_wind_max_kmh: Optional[float] = None
    daily: Optional[DailyWeather] = None
