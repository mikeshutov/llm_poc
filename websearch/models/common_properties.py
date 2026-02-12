from typing import Optional

from pydantic import BaseModel


class CommonProperties(BaseModel):
    color: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    gender: Optional[str] = None
