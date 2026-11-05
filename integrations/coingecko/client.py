from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from common.http import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.coingecko.models import CoinMarket

COINGECKO_API_BASE_URL = "https://api.coingecko.com/api/v3"
COINGECKO_WEBSITE_COIN_URL_TEMPLATE = "https://www.coingecko.com/en/coins/{coin_id}"


class CoinGeckoClientError(RuntimeError):
    pass


class CoinGeckoClient:
    def __init__(
        self,
        base_url: str = COINGECKO_API_BASE_URL,
        timeout_s: float = 15.0,
        ttl: timedelta = timedelta(minutes=5),
    ):
        self.base_url = base_url.rstrip("/")
        self._http = HttpClient(
            timeout_s=timeout_s,  
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

    def get_historical_prices(
        self,
        coin_id: str = "bitcoin",
        vs_currency: str = "usd",
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[tuple[datetime, float]]:
        if start_date is None or end_date is None or start_date >= end_date:
            raise CoinGeckoClientError("start_date and end_date are required, and start_date must precede end_date.")
        start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
        end = datetime.combine(end_date, time.max, tzinfo=timezone.utc)
        params: dict[str, Any] = {
            "vs_currency": vs_currency.lower(),
            "from": int(start.timestamp()),
            "to": int(end.timestamp()),
            "precision": "full",
        }
        url = f"{self.base_url}/coins/{coin_id}/market_chart/range"
        try:
            payload = self._http.get(url, params)
        except HttpClientError as e:
            raise CoinGeckoClientError(str(e)) from e
        if not isinstance(payload, dict) or not isinstance(payload.get("prices"), list):
            raise CoinGeckoClientError("Unexpected response from CoinGecko historical prices API.")
        prices: list[tuple[datetime, float]] = []
        for item in payload["prices"]:
            if isinstance(item, list) and len(item) >= 2:
                try:
                    prices.append((datetime.fromtimestamp(float(item[0]) / 1000, tz=timezone.utc), float(item[1])))
                except (TypeError, ValueError, OverflowError):
                    continue
        return prices
