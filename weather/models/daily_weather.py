from dataclasses import dataclass


@dataclass(frozen=True)
class DailyWeather:
    date: list[str]
    temperature_2m_max: list[float | None]
    temperature_2m_min: list[float | None]
    precipitation_sum: list[float | None]
    windspeed_10m_max: list[float | None]
