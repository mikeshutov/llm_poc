from __future__ import annotations

from typing import Iterable
from uuid import UUID

from psycopg.rows import dict_row

from assets.models import Asset, AssetChunk
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
