from typing import Optional

from pydantic import BaseModel, ConfigDict

from products.models.product_source import ProductSource


class ProductResultModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    category: Optional[str] = None
    color: Optional[str] = None
    style: Optional[str] = None
    gender: Optional[str] = None
    season: Optional[str] = None
    year: Optional[int] = None
    price: Optional[float] = None
    url: Optional[str] = None
    image_url: Optional[str] = None
    score: Optional[float] = None
    source: ProductSource = ProductSource.DB
