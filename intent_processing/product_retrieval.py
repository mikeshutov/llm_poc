import pandas as pd
from typing import Optional
from models.product_query import ProductQuery
from intent_processing.product_embeddings import embed_text
from intent_processing.db_product_search import search_products


def find_products(filters: Optional[ProductQuery]) -> pd.DataFrame:
    qvec = embed_text("".join(p for p in [filters.query_text, filters.style] if p))
    df = search_products(filters=filters, query_embedding=qvec, limit=20)

    #add web search results here as well maybe create new type for return
    return df
