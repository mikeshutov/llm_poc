from dataclasses import dataclass
from typing import List

from products.models.product_result import ProductResult


@dataclass(frozen=True)
class ProductSearchResults:
    internal_results: List[ProductResult]
    external_results: List[ProductResult]
