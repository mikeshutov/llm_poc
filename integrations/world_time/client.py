from __future__ import annotations

from datetime import timedelta
from typing import Any

from common.http import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.world_time.models import WorldTime


class WorldTimeClientError(RuntimeError):
    pass


class WorldTimeClient:
    def __init__(
        self,
        base_url: str = "https://worldtimeapi.org/api",
        timeout_s: float = 10.0,
        ttl: timedelta = DEFAULT_TTL,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = HttpClient(
            timeout_s=timeout_s,
            ttl=ttl,
        )

    def get_time(self, timezone: str) -> WorldTime:
        timezone = (timezone or "").strip()
        if not timezone:
            raise ValueError("Timezone must not be empty.")
        url = f"{self.base_url}/timezone/{timezone}"
        try:
            payload = self._http.get(url)
        except HttpClientError as e:
            raise WorldTimeClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise WorldTimeClientError("Unexpected response from World Time API.")
        return self._parse_world_time(payload)

    def list_timezones(self) -> list[str]:
        url = f"{self.base_url}/timezone"
        try:
            payload = self._http.get(url)
        except HttpClientError as e:
            raise WorldTimeClientError(str(e)) from e
        if not isinstance(payload, list):
            raise WorldTimeClientError("Unexpected response from World Time API.")
        return [tz for tz in payload if isinstance(tz, str)]

    def _parse_world_time(self, data: dict[str, Any]) -> WorldTime:
        return WorldTime(
            timezone=data.get("timezone", ""),
            datetime=data.get("datetime", ""),
            utc_offset=data.get("utc_offset", ""),
            day_of_week=data.get("day_of_week", 0),
            abbreviation=data.get("abbreviation", ""),
        )
