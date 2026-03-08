from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class BpiTime(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    updated: Optional[str] = None
    updated_iso: Optional[str] = Field(default=None, alias="updatedISO")


class BpiCurrency(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    code: str
    symbol: Optional[str] = None
    rate: Optional[str] = None
    description: Optional[str] = None
    rate_float: Optional[float] = Field(default=None, alias="rate_float")


class BitcoinPrice(BaseModel):
    time: Optional[BpiTime] = None
    disclaimer: Optional[str] = None
    bpi: dict[str, BpiCurrency] = {}
