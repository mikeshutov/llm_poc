from __future__ import annotations

from uuid import UUID

from psycopg.types.json import Json

from files.models import File
from db.connection import get_connection


class FileRepository:
    def __init__(self) -> None:
        self._conn = get_connection()

    def get_file_by_id(self, file_id: UUID, *, user_id: str | None = None) -> dict | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT f.id, f.file_path, f.file_name, f.file_type, f.metadata, f.uploaded_at,
                       f.user_id, f.conversation_id,
                       fc.content AS first_chunk
                FROM files f
                LEFT JOIN file_chunks fc ON fc.file_id = f.id AND fc.chunk_index = 0
                WHERE f.id = %s
                  AND (CAST(%s AS text) IS NULL OR f.user_id = %s)
                """,
                (file_id, user_id, user_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def create_file(
        self,
        file_path: str,
        file_name: str,
        file_type: str,
        metadata: dict | None = None,
        *,
        user_id: str | None = None,
        conversation_id: UUID | None = None,
    ) -> File:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO files
                    (file_path, file_name, file_type, metadata, user_id, conversation_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, file_path, file_name, file_type, metadata, uploaded_at, user_id, conversation_id
                """,
                (file_path, file_name, file_type, Json(metadata or {}), user_id, conversation_id),
            )
            row = cur.fetchone()
            self._conn.commit()
            return File(*row.values())
