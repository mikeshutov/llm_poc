from dataclasses import dataclass
from typing import Optional

from models.product_source import ProductSource


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

    # Ranking/debug fields (DB vector distance, RAG similarity, etc.)
    score: Optional[float] = None
    source: ProductSource = ProductSource.DB