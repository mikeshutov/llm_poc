import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence

import psycopg
from psycopg.rows import dict_row

from products.models.product_query import ProductQuery
from websearch.models.common_properties import CommonProperties
from products.models.product_result import ProductResult
from products.models.product_result_model import ProductResultModel
from products.models.product_source import ProductSource
from pgvector.psycopg import register_vector
from intent_processing.product_embeddings import embed_text

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
            common_filters: Optional["CommonProperties"] = None,
            product_filters: Optional[ProductQuery] = None,
            limit: int = 20,
    ) -> list[ProductResult]:
        sql, params = self._build_search_sql(query_embedding, common_filters, product_filters, limit)

        print(sql)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        # Map DB rows -> ProductResult
        results: list[ProductResult] = []
        for r in rows:
            data = {
                "id": str(r["id"]),
                "name": r["name"],
                "category": r.get("category"),
                "color": r.get("color"),
                "style": r.get("style"),
                "gender": r.get("gender"),
                "season": r.get("season"),
                "year": r.get("year"),
                "price": r.get("price"),
                "url": None,
                "image_url": r.get("image_url"),
                "score": float(r["distance"]) if r.get("distance") is not None else None,
                "source": ProductSource.DB,
            }
            validated = ProductResultModel.model_validate(data).model_dump()
            results.append(ProductResult(**validated))
        return results

    def search_products_filtered(
            self,
            query_text: str,
            common_filters: Optional["CommonProperties"] = None,
            product_filters: Optional[ProductQuery] = None,
            limit: int = 20,
    ) -> list[ProductResult]:
        query_embedding = embed_text(query_text or "")
        sql, params = self._build_search_sql(query_embedding, common_filters, product_filters, limit)

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results: list[ProductResult] = []
        for r in rows:
            validated = ProductResultModel.model_validate({
                "id": str(r["id"]),
                "name": r["name"],
                "category": r.get("category"),
                "color": r.get("color"),
                "style": r.get("style"),
                "gender": r.get("gender"),
                "season": r.get("season"),
                "year": r.get("year"),
                "price": r.get("price"),
                "url": None,
                "image_url": r.get("image_url"),
                "score": None,
                "source": ProductSource.DB,
            }).model_dump()
            results.append(ProductResult(**validated))
        return results

    def _build_search_sql(
        self,
        query_embedding: Sequence[float],
        common_filters: Optional["CommonProperties"],
        product_filters: Optional["ProductQuery"],
        limit: int,
    ) -> tuple[str, list[Any]]:
        where: list[str] = []
        params: list[Any] = []

        if common_filters:
            if getattr(common_filters, "color", None):
                where.append("LOWER(color) = LOWER(%s)")
                params.append(common_filters.color)

            if getattr(common_filters, "price_min", None) is not None:
                where.append("price >= %s")
                params.append(common_filters.price_min)

            if getattr(common_filters, "price_max", None) is not None:
                where.append("price <= %s")
                params.append(common_filters.price_max)

            if getattr(common_filters, "gender", None) is not None:
                where.append("gender = %s")
                params.append(common_filters.gender)

        if product_filters:
            if getattr(product_filters, "style", None):
                where.append("LOWER(style) LIKE LOWER(%s)")
                params.append(f"%{product_filters.style}%")

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        sql = f"""
            SELECT
              id, name, category, color, style, gender, season, year, price, image_url,
              (embedding <-> (%s)::vector) AS distance
            FROM products
            {where_sql}
            ORDER BY embedding <-> (%s)::vector
            LIMIT %s
        """

        final_params = [list(query_embedding), *params, list(query_embedding), limit]
        return sql, final_params
