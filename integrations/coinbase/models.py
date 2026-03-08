from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Currency(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    min_size: Optional[str] = Field(default=None, alias="min_size")
