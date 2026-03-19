from __future__ import annotations

from datetime import timedelta
from typing import Any

from common.http import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.nasa.models import AstronomyPicture


class NasaClientError(RuntimeError):
    pass


class NasaClient:
    def __init__(
        self,
        base_url: str = "https://api.nasa.gov",
        timeout_s: float = 10.0,
        ttl: timedelta = DEFAULT_TTL,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = HttpClient(
            timeout_s=timeout_s,
            ttl=ttl,
        )

    def get_apod(self, date: str | None = None) -> AstronomyPicture:
        url = f"{self.base_url}/planetary/apod"
        params: dict[str, Any] = {"api_key": "DEMO_KEY"}
        if date:
            params["date"] = date.strip()
        try:
            payload = self._http.get(url, params)
        except HttpClientError as e:
            raise NasaClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise NasaClientError("Unexpected response from NASA APOD API.")
        return AstronomyPicture(
            title=payload.get("title", ""),
            explanation=payload.get("explanation", ""),
            date=payload.get("date", ""),
            url=payload.get("url", ""),
            media_type=payload.get("media_type", ""),
        )
