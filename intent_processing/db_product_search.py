import os
from typing import Any, Optional

import psycopg
import pandas as pd
from pgvector.psycopg import register_vector

from models.product_query import ProductQuery

DB_URL = os.getenv("DATABASE_URL", "postgresql://app:app@localhost:5432/products")

def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(DB_URL)
    register_vector(conn)
    return conn

# main product search with filters
def search_products(
    filters: Optional[ProductQuery],
    query_embedding: list[float],
    limit: int = 20,
) -> pd.DataFrame:
    where = []
    params: list[Any] = []

    if filters is not None:
        if filters.color:
             where.append("LOWER(color) = LOWER(%s)")
             params.append(filters.color)

        if filters.price_min is not None:
            where.append("price >= %s")
            params.append(filters.price_min)

        if filters.price_max is not None:
            where.append("price <= %s")
            params.append(filters.price_max)

        if filters.gender is not None:
            where.append("gender=%s")
            params.append(filters.gender)

        if filters.style:
            where.append("LOWER(style)=LOWER(%s)")
            params.append(f"%{filters.style}%")

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    # Vector ranking: smaller distance = closer match
    # We also return the distance as a debug column
    sql = f"""
        SELECT
          id, name, category, color, style, gender, season, year, price,
          (embedding <-> (%s)::vector) AS distance
        FROM products
        {where_sql}
        ORDER BY embedding <-> (%s)::vector
        LIMIT %s
    """

    # IMPORTANT: query vector is used twice (SELECT distance and ORDER BY)
    params2 = [query_embedding, *params, query_embedding, limit]

    with get_conn() as conn:
        return pd.read_sql(sql, conn, params=params2)