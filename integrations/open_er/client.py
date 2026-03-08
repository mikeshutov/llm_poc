from __future__ import annotations

from datetime import timedelta

from integrations.http_client import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.open_er.models import ExchangeRates


class OpenErClientError(RuntimeError):
    pass


class OpenErClient:
    def __init__(
        self,
        base_url: str = "https://open.er-api.com/v6",
        timeout_s: float = 10.0,
        user_agent: str = "POCProductSearch/1.0 (OpenER client)",
        ttl: timedelta = DEFAULT_TTL,
        http: HttpClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = http or HttpClient(
            timeout_s=timeout_s,
            headers={"Accept": "application/json", "User-Agent": user_agent},
            ttl=ttl,
        )

    def get_latest(self, base: str = "USD") -> ExchangeRates:
        base_norm = (base or "USD").strip().upper()
        url = f"{self.base_url}/latest/{base_norm}"
        try:
            payload = self._http.get(url, {})
        except HttpClientError as e:
            raise OpenErClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise OpenErClientError("Unexpected response from open.er-api.")
        if payload.get("result") != "success":
            raise OpenErClientError(f"open.er-api error: {payload.get('error-type', 'unknown')}")
        return ExchangeRates.model_validate(payload)
