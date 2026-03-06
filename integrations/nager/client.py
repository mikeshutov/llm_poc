from __future__ import annotations

from datetime import timedelta
from typing import Any

from integrations.http_client import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.nager.models import PublicHoliday


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

        holidays: list[PublicHoliday] = []
        for item in payload:
            if not isinstance(item, dict):
                continue

            holidays.append(
                PublicHoliday(
                    date=str(item.get("date") or ""),
                    local_name=str(item.get("localName") or ""),
                    name=str(item.get("name") or ""),
                    country_code=str(item.get("countryCode") or country),
                    fixed=item.get("fixed") if isinstance(item.get("fixed"), bool) else None,
                    global_holiday=item.get("global") if isinstance(item.get("global"), bool) else None,
                    counties=item.get("counties") if isinstance(item.get("counties"), list) else None,
                    launch_year=item.get("launchYear") if isinstance(item.get("launchYear"), int) else None,
                    types=item.get("types") if isinstance(item.get("types"), list) else None,
                )
            )

        return holidays
