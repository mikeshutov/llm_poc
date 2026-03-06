from __future__ import annotations

from pydantic import BaseModel


class ExchangeRatesSnapshot(BaseModel):
    base: str
    date: str
    rates: dict[str, float] = {}


class ExchangeRatesSeries(BaseModel):
    base: str
    start_date: str
    end_date: str
    rates: dict[str, dict[str, float]] = {}
