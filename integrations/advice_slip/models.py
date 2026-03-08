from __future__ import annotations

from pydantic import BaseModel


class AdviceSlip(BaseModel):
    id: int
    advice: str
