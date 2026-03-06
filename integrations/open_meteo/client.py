from __future__ import annotations

import calendar
from datetime import date, datetime
from typing import Any

from datetime import timedelta

from integrations.http_client import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.open_meteo.errors import (
    WeatherArchiveError,
    WeatherGeocodingError,
    WeatherNotFoundError,
)
from integrations.open_meteo.models import CurrentWeather, DailyWeather, GeocodedLocation, MonthlyWeatherSummary


class OpenMeteoClient:
    def __init__(
        self,
        base_url_weather: str = "https://archive-api.open-meteo.com/v1",
        base_url_forecast: str = "https://api.open-meteo.com/v1",
        base_url_geo: str = "https://geocoding-api.open-meteo.com/v1",
        timeout_s: float = 20.0,
        ttl: timedelta = DEFAULT_TTL,
        http: HttpClient | None = None,
    ):
        self.base_url_weather = base_url_weather.rstrip("/")
        self.base_url_forecast = base_url_forecast.rstrip("/")
        self.base_url_geo = base_url_geo.rstrip("/")
        self._http = http or HttpClient(timeout_s=timeout_s, ttl=ttl)

    def _get_json(self, url: str, params: dict[str, Any], error_cls: type[Exception]) -> dict[str, Any]:
        try:
            payload = self._http.get(url, params)
        except HttpClientError as e:
            raise error_cls(str(e)) from e
        if not isinstance(payload, dict):
            raise error_cls(f"Unexpected non-object response from {url}")
        return payload

    def geocode_city(self, city: str) -> GeocodedLocation:
        city_norm = (city or "").strip()
        if not city_norm:
            raise WeatherGeocodingError("City must be a non-empty string.")

        url = f"{self.base_url_geo}/search"
        payload = self._get_json(
            url,
            params={
                "name": city_norm,
                "count": 1,
                "language": "en",
                "format": "json",
            },
            error_cls=WeatherGeocodingError,
        )
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            raise WeatherNotFoundError(f"No geocoding result found for city '{city_norm}'.")

        first = results[0] if isinstance(results[0], dict) else None
        if not first:
            raise WeatherGeocodingError("Malformed geocoding response (first result is not an object).")

        try:
            name = str(first.get("name") or city_norm)
            country = str(first.get("country") or "")
            latitude = float(first["latitude"])
            longitude = float(first["longitude"])
            timezone = first.get("timezone")
            timezone_str = str(timezone) if isinstance(timezone, str) else None
        except (KeyError, TypeError, ValueError) as exc:
            raise WeatherGeocodingError(f"Malformed geocoding result payload: {exc}") from exc

        return GeocodedLocation(
            name=name,
            country=country,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone_str,
        )

    def get_historical_month(self, city: str, year: int, month: int) -> MonthlyWeatherSummary:
        city_norm = (city or "").strip()
        if not city_norm:
            raise WeatherArchiveError("City must be a non-empty string.")
        if month < 1 or month > 12:
            raise WeatherArchiveError("Month must be between 1 and 12.")

        current_year = datetime.utcnow().year
        if year < 1940 or year > current_year:
            raise WeatherArchiveError(f"Year must be between 1940 and {current_year}.")

        geocoded = self.geocode_city(city_norm)
        last_day = calendar.monthrange(year, month)[1]
        start_date = date(year, month, 1).isoformat()
        end_date = date(year, month, last_day).isoformat()

        url = f"{self.base_url_weather}/archive"
        payload = self._get_json(
            url,
            params={
                "latitude": geocoded.latitude,
                "longitude": geocoded.longitude,
                "start_date": start_date,
                "end_date": end_date,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
                "timezone": "auto",
            },
            error_cls=WeatherArchiveError,
        )

        daily_payload = payload.get("daily")
        if not isinstance(daily_payload, dict):
            raise WeatherArchiveError("Malformed archive response: missing 'daily' object.")

        dates = daily_payload.get("time")
        temp_max = daily_payload.get("temperature_2m_max")
        temp_min = daily_payload.get("temperature_2m_min")
        precip = daily_payload.get("precipitation_sum")
        wind = daily_payload.get("windspeed_10m_max")

        if not all(isinstance(v, list) for v in [dates, temp_max, temp_min, precip, wind]):
            raise WeatherArchiveError("Malformed archive response: one or more daily arrays are missing.")

        length = len(dates)
        if not all(len(v) == length for v in [temp_max, temp_min, precip, wind]):
            raise WeatherArchiveError("Malformed archive response: daily array lengths do not match.")

        daily = DailyWeather(
            date=[str(v) for v in dates],
            temperature_2m_max=[_to_float_or_none(v) for v in temp_max],
            temperature_2m_min=[_to_float_or_none(v) for v in temp_min],
            precipitation_sum=[_to_float_or_none(v) for v in precip],
            windspeed_10m_max=[_to_float_or_none(v) for v in wind],
        )

        avg_temp_max = _avg(daily.temperature_2m_max)
        avg_temp_min = _avg(daily.temperature_2m_min)
        total_precip = _sum(daily.precipitation_sum)
        avg_wind = _avg(daily.windspeed_10m_max)

        return MonthlyWeatherSummary(
            city=geocoded.name,
            country=geocoded.country,
            year=year,
            month=month,
            days_count=length,
            avg_temp_max_c=avg_temp_max,
            avg_temp_min_c=avg_temp_min,
            total_precip_mm=total_precip,
            avg_wind_max_kmh=avg_wind,
            daily=daily,
        )


    def get_current(self, latitude: float, longitude: float, timezone: str = "auto") -> CurrentWeather:
        data = self._get_json(
            f"{self.base_url_forecast}/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current_weather": "true",
                "timezone": timezone,
            },
            error_cls=WeatherArchiveError,
        )
        cw = data["current_weather"]
        units = data.get("current_weather_units", {
            "temperature": "\u00b0C",
            "windspeed": "km/h",
            "winddirection": "\u00b0",
        })
        return CurrentWeather(
            latitude=data["latitude"],
            longitude=data["longitude"],
            timezone=data.get("timezone", timezone),
            elevation=data.get("elevation", 0.0),
            time=cw["time"],
            temperature=cw["temperature"],
            windspeed=cw["windspeed"],
            winddirection=cw["winddirection"],
            weathercode=cw["weathercode"],
            is_day=bool(cw.get("is_day", 1)),
            units=units,
        )

    def get_current_for_location(self, location_name: str) -> tuple[GeocodedLocation, CurrentWeather] | None:
        try:
            loc = self.geocode_city(location_name)
        except WeatherNotFoundError:
            return None
        weather = self.get_current(loc.latitude, loc.longitude, timezone=loc.timezone or "auto")
        return loc, weather


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 2)


def _sum(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return round(sum(present), 2)
