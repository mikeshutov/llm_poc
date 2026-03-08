from __future__ import annotations

from datetime import timedelta
from typing import Any

from integrations.http_client import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.coingecko.models import CoinMarket


class CoinGeckoClientError(RuntimeError):
    pass


class CoinGeckoClient:
    def __init__(
        self,
        base_url: str = "https://api.coingecko.com/api/v3",
        timeout_s: float = 15.0,
        user_agent: str = "POCProductSearch/1.0 (CoinGecko client)",
        ttl: timedelta = timedelta(minutes=5),
        http: HttpClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = http or HttpClient(
            timeout_s=timeout_s,
            headers={"Accept": "application/json", "User-Agent": user_agent},
            ttl=ttl,
        )

    def get_markets(
        self,
        vs_currency: str = "usd",
        order: str = "market_cap_desc",
        per_page: int = 20,
        page: int = 1,
    ) -> list[CoinMarket]:
        params: dict[str, Any] = {
            "vs_currency": vs_currency.lower(),
            "order": order,
            "per_page": per_page,
            "page": page,
            "sparkline": "false",
        }
        url = f"{self.base_url}/coins/markets"
        try:
            payload = self._http.get(url, params)
        except HttpClientError as e:
            raise CoinGeckoClientError(str(e)) from e
        if not isinstance(payload, list):
            raise CoinGeckoClientError("Unexpected response from CoinGecko API.")
        return [CoinMarket.model_validate(item) for item in payload if isinstance(item, dict)]
