from __future__ import annotations

import os
from enum import Enum
from uuid import UUID

from common.file_constants import IMAGE_MIME_PREFIX
from files.models import FileChunkResult
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
        with self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO file_chunks (file_id, chunk_index, content, embedding)
                VALUES (%s, %s, %s, (%s)::vector)
                ON CONFLICT (file_id, chunk_index) DO NOTHING
                """,
                [(file_id, idx, content, embedding) for idx, content, embedding in chunks],
            )
            self._conn.commit()

    def search_file_via_chunks(
        self,
        query_embedding: list[float],
        file_id: UUID | None = None,
        file_type: FileTypeFilter | None = None,
        limit: int = TOP_K,
    ) -> list[FileChunkResult]:
        distinct = "DISTINCT ON (cf.id)" if not file_id else ""

        conditions = ["cfc.embedding <=> (%s)::vector <= %s"]
        params: list = [query_embedding, query_embedding, MAX_CHUNK_DISTANCE]
        if file_id:
            conditions.append("cfc.file_id = %s")
            params.append(file_id)
        if file_type == FileTypeFilter.image:
            conditions.append(f"cf.file_type LIKE '{IMAGE_MIME_PREFIX}%%'")
        elif file_type == FileTypeFilter.text:
            conditions.append(f"cf.file_type NOT LIKE '{IMAGE_MIME_PREFIX}%%'")
        params.append(limit)
        order_by = "cf.id, distance ASC" if not file_id else "distance ASC"
        sql = f"""
            SELECT {distinct} cf.id AS file_id, cf.file_name, cf.file_path, cfc.content,
                cfc.embedding <=> (%s)::vector AS distance
            FROM file_chunks cfc
            JOIN files cf ON cf.id = cfc.file_id
            WHERE {" AND ".join(conditions)}
            ORDER BY {order_by}
            LIMIT %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            #messyish but we can improve this later
            return [FileChunkResult(**{row_key: row_value for row_key, row_value in row.items() if row_key != "distance"}) for row in rows]
