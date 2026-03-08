from __future__ import annotations

from datetime import timedelta

from integrations.http_client import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.coinbase.models import Currency


class CoinbaseClientError(RuntimeError):
    pass


class CoinbaseClient:
    def __init__(
        self,
        base_url: str = "https://api.coinbase.com/v2",
        timeout_s: float = 10.0,
        user_agent: str = "POCProductSearch/1.0 (Coinbase client)",
        ttl: timedelta = DEFAULT_TTL,
        http: HttpClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = http or HttpClient(
            timeout_s=timeout_s,
            headers={"Accept": "application/json", "User-Agent": user_agent},
            ttl=ttl,
        )

    def get_currencies(self) -> list[Currency]:
        url = f"{self.base_url}/currencies"
        try:
            payload = self._http.get(url, {})
        except HttpClientError as e:
            raise CoinbaseClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise CoinbaseClientError("Unexpected response from Coinbase API.")
        data = payload.get("data") or []
        return [Currency.model_validate(item) for item in data if isinstance(item, dict)]
