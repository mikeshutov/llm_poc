from __future__ import annotations

import os
from enum import Enum
from uuid import UUID

from common.config import IMAGE_MIME_PREFIX
from assets.repository.asset_chunk_repository import AssetChunkRepository
from assets.files.models import FileChunkResult
from db.connection import get_connection


class FileTypeFilter(str, Enum):
    text = "text"
    image = "image"


TOP_K = 10
MAX_CHUNK_DISTANCE = float(os.getenv("MAX_CHUNK_DISTANCE", "0.7"))


class FileChunkRepository:
    def __init__(self) -> None:
        self._conn = get_connection()

    def save_chunks(self, file_id: UUID, chunks: list[tuple[int, str, list[float]]]) -> None:
        self.save_content(file_id, "file", [content for _, content, _ in chunks])

    def save_content(self, asset_id: UUID, asset_type: str, contents: list[str]) -> None:
        AssetChunkRepository(self._conn).replace_content(asset_id, asset_type, contents)

    def delete_chunks(self, file_id: UUID) -> None:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM asset_chunks WHERE asset_id = %s", (file_id,))
            self._conn.commit()

    def get_first_chunk(self, asset_id: UUID, *, user_id: str | None = None) -> str | None:
        return AssetChunkRepository(self._conn).get_first(asset_id, user_id=user_id)

    def search_file_via_chunks(
        self,
        query_embedding: list[float],
        file_id: UUID | None = None,
        file_type: FileTypeFilter | None = None,
        user_id: str | None = None,
        limit: int = TOP_K,
    ) -> list[FileChunkResult]:
        distinct = "DISTINCT ON (a.id)" if not file_id else ""

        conditions = ["cfc.embedding <=> (%s)::vector <= %s", "(a.user_id IS NULL OR (%s IS NOT NULL AND a.user_id = %s))"]
        params: list = [query_embedding, query_embedding, MAX_CHUNK_DISTANCE, user_id, user_id]
        if file_id:
            conditions.append("cfc.asset_id = %s")
            params.append(file_id)
        if file_type == FileTypeFilter.image:
            conditions.append(f"fa.mime_type LIKE '{IMAGE_MIME_PREFIX}%%'")
        elif file_type == FileTypeFilter.text:
            conditions.append(f"fa.mime_type NOT LIKE '{IMAGE_MIME_PREFIX}%%'")
        params.append(limit)
        order_by = "a.id, distance ASC" if not file_id else "distance ASC"
        sql = f"""
            SELECT {distinct} a.id AS file_id, a.asset_type, a.name AS file_name,
                fa.storage_path AS file_path, cfc.content,
                cfc.embedding <=> (%s)::vector AS distance
            FROM asset_chunks cfc
            JOIN assets a ON a.id = cfc.asset_id
            JOIN file_assets fa ON fa.asset_id = a.id
            WHERE {" AND ".join(conditions)}
            ORDER BY {order_by}
            LIMIT %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [FileChunkResult(**{row_key: row_value for row_key, row_value in row.items() if row_key != "distance"}) for row in rows]
