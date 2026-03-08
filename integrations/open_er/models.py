from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ExchangeRates(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    result: str
    base_code: str
    time_last_update_utc: Optional[str] = None
    rates: dict[str, float] = {}
