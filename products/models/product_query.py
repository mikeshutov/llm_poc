from typing import Optional

from pydantic import BaseModel


class ProductQuery(BaseModel):
    style: Optional[str] = None
    size_label: Optional[str] = None
    size_numeric: Optional[float] = None
    size_unit: Optional[str] = None
