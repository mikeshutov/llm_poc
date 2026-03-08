from __future__ import annotations

from datetime import timedelta

from integrations.http_client import HttpClient, HttpClientError
from integrations.finance.coindesk.models import BitcoinPrice, BpiCurrency


class CoindeskClientError(RuntimeError):
    pass


class CoindeskClient:
    def __init__(
        self,
        base_url: str = "https://api.coindesk.com/v1/bpi",
        timeout_s: float = 10.0,
        user_agent: str = "POCProductSearch/1.0 (Coindesk client)",
        ttl: timedelta = timedelta(minutes=5),
        http: HttpClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = http or HttpClient(
            timeout_s=timeout_s,
            headers={"Accept": "application/json", "User-Agent": user_agent},
            ttl=ttl,
        )

    def get_current_price(self) -> BitcoinPrice:
        url = f"{self.base_url}/currentprice.json"
        try:
            payload = self._http.get(url, {})
        except HttpClientError as e:
            raise CoindeskClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise CoindeskClientError("Unexpected response from Coindesk API.")

        raw_bpi = payload.get("bpi") or {}
        bpi = {
            code: BpiCurrency.model_validate(entry)
            for code, entry in raw_bpi.items()
            if isinstance(entry, dict)
        }
        return BitcoinPrice.model_validate({**payload, "bpi": bpi})
