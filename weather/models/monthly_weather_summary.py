from dataclasses import dataclass

from weather.models.daily_weather import DailyWeather


@dataclass(frozen=True)
class MonthlyWeatherSummary:
    city: str
    country: str
    year: int
    month: int
    days_count: int
    avg_temp_max_c: float | None
    avg_temp_min_c: float | None
    total_precip_mm: float | None
    avg_wind_max_kmh: float | None
    daily: DailyWeather | None = None
