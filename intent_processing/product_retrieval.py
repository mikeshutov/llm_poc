import pandas as pd
from typing import Optional
from models.product_query import ProductQuery
from intent_processing.product_embeddings import embed_text
from intent_processing.db_product_search import search_products
from models.product_result import ProductResult
from products.product_repository import ProductRepository


def find_products(filters: Optional[ProductQuery]) -> list[ProductResult]:
    repo = ProductRepository()
    query_vector = embed_text("".join(p for p in [filters.query_text, filters.style] if p))
    results = repo.search_products(query_embedding=query_vector,filters=filters,limit=10)


    #add web search results here as well maybe create new type for return
    return results
