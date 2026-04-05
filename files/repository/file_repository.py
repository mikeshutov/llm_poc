from __future__ import annotations

from psycopg.types.json import Json

from files.models import File
from db.connection import get_connection


class FileRepository:
    def __init__(self) -> None:
        self._conn = get_connection()

    def create_file(
        self,
        file_path: str,
        file_name: str,
        file_type: str,
        metadata: dict | None = None,
    ) -> File:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO files
                    (file_path, file_name, file_type, metadata)
                VALUES (%s, %s, %s, %s)
                RETURNING id, file_path, file_name, file_type, metadata, uploaded_at
                """,
                (file_path, file_name, file_type, Json(metadata or {})),
            )
            row = cur.fetchone()
            self._conn.commit()
            return File(*row.values())
