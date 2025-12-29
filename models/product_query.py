from typing import Optional
from pydantic import BaseModel

class ProductQuery(BaseModel):
    category: Optional[str] = None
    color: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    style: Optional[str] = None
    size_label: Optional[str] = None
    size_numeric: Optional[float] = None
    size_unit: Optional[str] = None
    gender: Optional[str] = None
    query_text: Optional[str] = None
