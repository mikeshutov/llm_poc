from __future__ import annotations

from datetime import timedelta
from typing import Any

from integrations.http_client import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.calendar.nager.models import PublicHoliday


class NagerDateClientError(RuntimeError):
    pass


class NagerDateClient:
    def __init__(
        self,
        base_url: str = "https://date.nager.at/api/v3",
        timeout_s: float = 20.0,
        user_agent: str = "POCProductSearch/1.0 (Nager.Date client)",
        ttl: timedelta = DEFAULT_TTL,
        http: HttpClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = http or HttpClient(
            timeout_s=timeout_s,
            headers={"Accept": "application/json", "User-Agent": user_agent},
            ttl=ttl,
        )

    def _get_json(self, path: str) -> dict[str, Any] | list[Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            payload = self._http.get(url)
        except HttpClientError as e:
            raise NagerDateClientError(str(e)) from e
        if not isinstance(payload, (dict, list)):
            raise NagerDateClientError("Unexpected non-JSON response from Nager.Date API.")
        return payload

    def get_public_holidays(self, year: int, country_code: str) -> list[PublicHoliday]:
        if year < 1:
            raise ValueError("Year must be a positive integer.")

        country = (country_code or "").strip().upper()
        if len(country) != 2:
            raise ValueError("Country code must be a 2-letter ISO code.")

        payload = self._get_json(f"publicholidays/{year}/{country}")
        if not isinstance(payload, list):
            raise NagerDateClientError("Malformed Nager.Date response: expected a list of holidays.")

        return [PublicHoliday.model_validate(item) for item in payload if isinstance(item, dict)]
