from __future__ import annotations

from datetime import timedelta
from typing import Any

from common.http import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.rest_countries.models import Country


class RestCountriesClientError(RuntimeError):
    pass


class RestCountriesClient:
    def __init__(
        self,
        base_url: str = "https://restcountries.com/v3.1",
        timeout_s: float = 10.0,
        ttl: timedelta = DEFAULT_TTL,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = HttpClient(
            timeout_s=timeout_s,
            ttl=ttl,
        )

    def search(self, name: str) -> list[Country]:
        name = (name or "").strip()
        if not name:
            raise ValueError("Country name must not be empty.")
        url = f"{self.base_url}/name/{name}"
        params: dict[str, Any] = {
            "fields": "name,capital,region,subregion,population,currencies,languages,flags"
        }
        try:
            payload = self._http.get(url, params)
        except HttpClientError as e:
            raise RestCountriesClientError(str(e)) from e
        if not isinstance(payload, list):
            raise RestCountriesClientError("Unexpected response from REST Countries API.")
        return [self._parse_country(item) for item in payload if isinstance(item, dict)]

    def _parse_country(self, data: dict[str, Any]) -> Country:
        name_obj = data.get("name") or {}
        flags_obj = data.get("flags") or {}
        return Country(
            common_name=name_obj.get("common", ""),
            official_name=name_obj.get("official", ""),
            capital=data.get("capital") or [],
            region=data.get("region", ""),
            subregion=data.get("subregion", ""),
            population=data.get("population", 0),
            currencies=data.get("currencies") or {},
            languages=data.get("languages") or {},
            flag=flags_obj.get("emoji", ""),
        )
