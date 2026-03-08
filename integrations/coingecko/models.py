from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class CoinMarket(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    symbol: str
    name: str
    current_price: Optional[float] = Field(default=None, alias="current_price")
    market_cap: Optional[float] = Field(default=None, alias="market_cap")
    market_cap_rank: Optional[int] = Field(default=None, alias="market_cap_rank")
    total_volume: Optional[float] = Field(default=None, alias="total_volume")
    high_24h: Optional[float] = Field(default=None, alias="high_24h")
    low_24h: Optional[float] = Field(default=None, alias="low_24h")
    price_change_24h: Optional[float] = Field(default=None, alias="price_change_24h")
    price_change_percentage_24h: Optional[float] = Field(default=None, alias="price_change_percentage_24h")
    image: Optional[str] = None
