from __future__ import annotations

from datetime import timedelta
from typing import Any

from integrations.http_client import HttpClient, HttpClientError, DEFAULT_TTL
from integrations.finance.frankfurter.models import ExchangeRatesSnapshot, ExchangeRatesSeries


class FrankfurterClientError(RuntimeError):
    pass


class FrankfurterClient:
    def __init__(
        self,
        base_url: str = "https://api.frankfurter.dev/v1",
        timeout_s: float = 20.0,
        user_agent: str = "POCProductSearch/1.0 (Frankfurter client)",
        ttl: timedelta = DEFAULT_TTL,
        http: HttpClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._http = http or HttpClient(
            timeout_s=timeout_s,
            headers={"Accept": "application/json", "User-Agent": user_agent},
            ttl=ttl,
        )

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            payload = self._http.get(url, params or {})
        except HttpClientError as e:
            raise FrankfurterClientError(str(e)) from e
        if not isinstance(payload, dict):
            raise FrankfurterClientError("Unexpected non-object response from Frankfurter API.")
        return payload

    def get_latest_rates(
        self,
        base: str = "EUR",
        symbols: list[str] | None = None,
    ) -> ExchangeRatesSnapshot:
        payload = self._get_json(
            "latest",
            params=self._build_rate_params(base=base, symbols=symbols),
        )
        return self._parse_snapshot(payload, fallback_base=base)

    def get_historical_rates(
        self,
        date: str,
        base: str = "EUR",
        symbols: list[str] | None = None,
    ) -> ExchangeRatesSnapshot:
        date_norm = (date or "").strip()
        if not date_norm:
            raise ValueError("Date must be a non-empty string in YYYY-MM-DD format.")

        payload = self._get_json(
            date_norm,
            params=self._build_rate_params(base=base, symbols=symbols),
        )
        return self._parse_snapshot(payload, fallback_base=base)

    def get_time_series(
        self,
        start_date: str,
        end_date: str | None = None,
        base: str = "EUR",
        symbols: list[str] | None = None,
    ) -> ExchangeRatesSeries:
        start_norm = (start_date or "").strip()
        end_norm = (end_date or "").strip()
        if not start_norm:
            raise ValueError("Start date must be a non-empty string in YYYY-MM-DD format.")

        path = f"{start_norm}..{end_norm}" if end_norm else f"{start_norm}.."
        payload = self._get_json(
            path,
            params=self._build_rate_params(base=base, symbols=symbols),
        )
        return self._parse_series(payload, fallback_base=base)

    def get_currencies(self) -> dict[str, str]:
        payload = self._get_json("currencies")
        currencies: dict[str, str] = {}
        for code, name in payload.items():
            if not isinstance(code, str) or not isinstance(name, str):
                continue
            currencies[code] = name
        return currencies

    def _build_rate_params(
        self,
        *,
        base: str,
        symbols: list[str] | None,
    ) -> dict[str, str]:
        base_norm = self._normalize_currency_code(base, label="Base currency")
        params: dict[str, str] = {"base": base_norm}
        if symbols:
            normalized_symbols = [
                self._normalize_currency_code(symbol, label="Currency symbol")
                for symbol in symbols
            ]
            params["symbols"] = ",".join(normalized_symbols)
        return params

    def _parse_snapshot(
        self,
        payload: dict[str, Any],
        *,
        fallback_base: str,
    ) -> ExchangeRatesSnapshot:
        rates_payload = payload.get("rates")
        if not isinstance(rates_payload, dict):
            raise FrankfurterClientError("Malformed Frankfurter response: missing rates object.")

        date_value = payload.get("date")
        date_str = str(date_value).strip() if date_value else ""
        if not date_str:
            raise FrankfurterClientError("Malformed Frankfurter response: missing date.")

        base_value = payload.get("base")
        base_code = (
            self._normalize_currency_code(str(base_value), label="Base currency")
            if isinstance(base_value, str)
            else self._normalize_currency_code(fallback_base, label="Base currency")
        )

        return ExchangeRatesSnapshot(base=base_code, date=date_str, rates=self._parse_rates_map(rates_payload))

    def _parse_series(
        self,
        payload: dict[str, Any],
        *,
        fallback_base: str,
    ) -> ExchangeRatesSeries:
        rates_payload = payload.get("rates")
        if not isinstance(rates_payload, dict):
            raise FrankfurterClientError(
                "Malformed Frankfurter response: missing time series rates object."
            )

        start_date = str(payload.get("start_date") or "").strip()
        end_date = str(payload.get("end_date") or "").strip()
        if not start_date or not end_date:
            raise FrankfurterClientError(
                "Malformed Frankfurter response: missing start_date or end_date."
            )

        base_value = payload.get("base")
        base_code = (
            self._normalize_currency_code(str(base_value), label="Base currency")
            if isinstance(base_value, str)
            else self._normalize_currency_code(fallback_base, label="Base currency")
        )

        parsed_rates: dict[str, dict[str, float]] = {}
        for date_key, day_rates in rates_payload.items():
            if not isinstance(date_key, str) or not isinstance(day_rates, dict):
                continue
            parsed_rates[date_key] = self._parse_rates_map(day_rates)

        return ExchangeRatesSeries(base=base_code, start_date=start_date, end_date=end_date, rates=parsed_rates)

    def _parse_rates_map(self, rates_payload: dict[str, Any]) -> dict[str, float]:
        parsed: dict[str, float] = {}
        for code, value in rates_payload.items():
            if not isinstance(code, str):
                continue
            try:
                parsed[self._normalize_currency_code(code, label="Currency code")] = float(value)
            except (TypeError, ValueError):
                continue
        return parsed

    def _normalize_currency_code(self, code: str, *, label: str) -> str:
        normalized = (code or "").strip().upper()
        if len(normalized) != 3 or not normalized.isalpha():
            raise ValueError(f"{label} must be a 3-letter ISO currency code.")
        return normalized
