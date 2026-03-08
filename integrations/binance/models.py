from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Ticker24hr(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    symbol: str
    price_change: Optional[str] = Field(default=None, alias="priceChange")
    price_change_percent: Optional[str] = Field(default=None, alias="priceChangePercent")
    last_price: Optional[str] = Field(default=None, alias="lastPrice")
    high_price: Optional[str] = Field(default=None, alias="highPrice")
    low_price: Optional[str] = Field(default=None, alias="lowPrice")
    volume: Optional[str] = None
    quote_volume: Optional[str] = Field(default=None, alias="quoteVolume")
    open_time: Optional[int] = Field(default=None, alias="openTime")
    close_time: Optional[int] = Field(default=None, alias="closeTime")
    count: Optional[int] = None
