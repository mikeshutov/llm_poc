from __future__ import annotations

from datetime import timedelta
from typing import Any

from integrations.http_client import HttpClient, HttpClientError
from integrations.binance.models import Ticker24hr


class BinanceClientError(RuntimeError):
    pass


class BinanceClient:
    def __init__(
        self,
        base_url: str = "https://api4.binance.com/api/v3",
        timeout_s: float = 10.0,
        user_agent: str = "POCProductSearch/1.0 (Binance client)",
        ttl: timedelta = timedelta(minutes=1),
        http: HttpClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = http or HttpClient(
            timeout_s=timeout_s,
            headers={"Accept": "application/json", "User-Agent": user_agent},
            ttl=ttl,
        )

    def get_ticker(self, symbol: str) -> Ticker24hr:
        sym = (symbol or "").strip().upper()
        if not sym:
            raise ValueError("Symbol must not be empty.")
        url = f"{self.base_url}/ticker/24hr"
        try:
            payload = self._http.get(url, {"symbol": sym})
        except HttpClientError as e:
            raise BinanceClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise BinanceClientError("Unexpected response from Binance API.")
        return Ticker24hr.model_validate(payload)

    def get_all_tickers(self) -> list[Ticker24hr]:
        url = f"{self.base_url}/ticker/24hr"
        try:
            payload = self._http.get(url, {})
        except HttpClientError as e:
            raise BinanceClientError(str(e)) from e
        if not isinstance(payload, list):
            raise BinanceClientError("Unexpected response from Binance API.")
        return [Ticker24hr.model_validate(item) for item in payload if isinstance(item, dict)]
