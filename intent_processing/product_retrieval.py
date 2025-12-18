import pandas as pd
from typing import Optional
from models.product_query import ProductQuery

from intent_processing.category_resolution import resolve_category

def normalize_filters(pq):
    if pq and pq.category:
        pq.category = resolve_category(pq.category)
    return pq

def find_products(filters: Optional[ProductQuery]):
    result = df.copy()
    if filters.category:
        result = result[result["category"] == filters.category]
    if filters.color:
        result = result[result["color"] == filters.color]

    return result.head(10)