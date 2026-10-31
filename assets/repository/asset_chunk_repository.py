from __future__ import annotations

import os
from typing import Iterable
from uuid import UUID

from psycopg.rows import dict_row

from assets.models import Asset, AssetChunk, AssetContentType
from db.connection import get_connection
from llm.clients.embeddings import embed_text


class AssetChunkRepository:
    """Persists searchable asset chunks and derives each embedding from content."""

    def __init__(self, conn=None):
        self._conn = conn or get_connection()

    @staticmethod
    def _normalized_contents(contents: Iterable[str]) -> list[str]:
        return [content.strip() for content in contents if isinstance(content, str) and content.strip()]

    def replace_content(self, asset_id: UUID, asset_type: str, contents: Iterable[str]) -> list[AssetChunk]:
        normalized = self._normalized_contents(contents)
        if not normalized:
            normalized = [f"Asset type: {asset_type}. Asset ID: {asset_id}"]
        chunks = [
            AssetChunk(
                asset_id=asset_id,
                asset_type=asset_type,
                chunk_index=index,
                content=content,
                embedding=embed_text(content, purpose="asset_chunk"),
            )
            for index, content in enumerate(normalized)
        ]
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM asset_chunks WHERE asset_id = %s", (asset_id,))
            cur.executemany(
                """
                INSERT INTO asset_chunks
                    (asset_id, asset_type, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s, (%s)::vector)
                """,
                [
                    (chunk.asset_id, chunk.asset_type, chunk.chunk_index, chunk.content, chunk.embedding)
                    for chunk in chunks
                ],
            )
            self._conn.commit()
        return chunks

    def ensure_asset_chunk(self, asset: Asset) -> list[AssetChunk]:
        source = "\n".join(
            part
            for part in (
                f"Name: {asset.name}",
                f"Type: {asset.asset_type}",
                f"Origin: {asset.origin}",
                str(asset.metadata) if asset.metadata else "",
            )
            if part
        )
        return self.replace_content(asset.id, asset.asset_type, [source])

    def get_first(self, asset_id: UUID, *, user_id: str | None = None) -> str | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.content
                FROM asset_chunks c
                JOIN assets a ON a.id = c.asset_id
                WHERE c.asset_id = %s
                  AND c.chunk_index = 0
                  AND (a.user_id IS NULL OR (%s IS NOT NULL AND a.user_id = %s))
                """,
                (asset_id, user_id, user_id),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def delete_chunks(self, asset_id: UUID) -> None:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM asset_chunks WHERE asset_id = %s", (asset_id,))
            self._conn.commit()

    def search_assets(
        self,
        query_embedding: list[float],
        *,
        asset_id: UUID | None = None,
        asset_type: str | None = None,
        content_type: AssetContentType | None = None,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        distinct = "DISTINCT ON (a.id)" if not asset_id else ""
        conditions = [
            "cfc.embedding <=> (%s)::vector <= %s",
            "(a.user_id IS NULL OR (%s IS NOT NULL AND a.user_id = %s))",
        ]
        params: list = [query_embedding, query_embedding, float(os.getenv("MAX_CHUNK_DISTANCE", "0.7")), user_id, user_id]
        if asset_id:
            conditions.append("cfc.asset_id = %s")
            params.append(asset_id)
        if asset_type:
            conditions.append("a.asset_type = %s")
            params.append(asset_type)
        if content_type == AssetContentType.IMAGE:
            conditions.append("fa.mime_type LIKE 'image/%%'")
        elif content_type == AssetContentType.TEXT:
            conditions.append("(a.asset_type <> 'file' OR fa.mime_type IS NULL OR fa.mime_type NOT LIKE 'image/%%')")
        params.append(limit)
        order_by = "a.id, distance ASC" if not asset_id else "distance ASC"
        sql = f"""
            SELECT {distinct} a.id AS asset_id, a.asset_type, a.name AS asset_name,
                fa.storage_path, fa.mime_type, hp.url, hp.title, cfc.content,
                cfc.embedding <=> (%s)::vector AS distance
            FROM asset_chunks cfc
            JOIN assets a ON a.id = cfc.asset_id
            LEFT JOIN file_assets fa ON fa.asset_id = a.id
            LEFT JOIN http_pages hp ON hp.asset_id = a.id
            WHERE {" AND ".join(conditions)}
            ORDER BY {order_by}
            LIMIT %s
        """
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return [
                {key: value for key, value in row.items() if key != "distance"}
                for row in cur.fetchall()
            ]
