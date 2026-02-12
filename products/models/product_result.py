from dataclasses import dataclass
from typing import Optional

from products.models.product_source import ProductSource


@dataclass(frozen=False)
class ProductResult:
    id: str
    name: str
    category: Optional[str]
    color: Optional[str]
    style: Optional[str]
    gender: Optional[str]
    season: Optional[str]
    year: Optional[int]
    price: Optional[float]
    url: Optional[str] = None
    image_url: Optional[str] = None

    # Ranking/debug fields (DB vector distance, RAG similarity, etc.)
    score: Optional[float] = None
    source: ProductSource = ProductSource.DB
