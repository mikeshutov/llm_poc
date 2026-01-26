import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence

import psycopg
from psycopg.rows import dict_row

from models.product_query import ProductQuery
from models.product_result import ProductResult
from models.product_source import ProductSource
from pgvector.psycopg import register_vector

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://app:app@localhost:5432/products",
)

class ProductRepository:
    def __init__(self):
        conn = psycopg.connect(DB_URL)
        register_vector(conn)
        self._conn = conn

    def search_products(
            self,
            query_embedding: Sequence[float],
            filters: Optional[ProductQuery] = None,
            limit: int = 20,
    ) -> list[ProductResult]:
        sql, params = self._build_search_sql(query_embedding, filters, limit)

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        # Map DB rows -> ProductResult
        results: list[ProductResult] = []
        for r in rows:
            results.append(
                ProductResult(
                    id=str(r["id"]),
                    name=r["name"],
                    category=r.get("category"),
                    color=r.get("color"),
                    style=r.get("style"),
                    gender=r.get("gender"),
                    season=r.get("season"),
                    year=r.get("year"),
                    price=r.get("price"),
                    score=float(r["distance"]) if r.get("distance") is not None else None,
                    source=ProductSource.DB,
                )
            )
        return results

    def _build_search_sql(
        self,
        query_embedding: Sequence[float],
        filters: Optional["ProductQuery"],
        limit: int,
    ) -> tuple[str, list[Any]]:
        where: list[str] = []
        params: list[Any] = []

        if filters:
            if getattr(filters, "color", None):
                where.append("LOWER(color) = LOWER(%s)")
                params.append(filters.color)

            if getattr(filters, "price_min", None) is not None:
                where.append("price >= %s")
                params.append(filters.price_min)

            if getattr(filters, "price_max", None) is not None:
                where.append("price <= %s")
                params.append(filters.price_max)

            if getattr(filters, "gender", None) is not None:
                where.append("gender = %s")
                params.append(filters.gender)

            if getattr(filters, "style", None):
                where.append("LOWER(style) LIKE LOWER(%s)")
                params.append(f"%{filters.style}%")

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        sql = f"""
            SELECT
              id, name, category, color, style, gender, season, year, price,
              (embedding <-> (%s)::vector) AS distance
            FROM products
            {where_sql}
            ORDER BY embedding <-> (%s)::vector
            LIMIT %s
        """

        final_params = [list(query_embedding), *params, list(query_embedding), limit]
        return sql, final_params