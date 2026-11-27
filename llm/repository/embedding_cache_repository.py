from contextlib import closing, nullcontext

import psycopg
from psycopg.rows import dict_row

from db.connection import get_connection


class EmbeddingCacheRepository:
    """Permanent embedding cache; injected connections remain caller-owned."""

    def __init__(self, conn: psycopg.Connection | None = None):
        self._conn = conn

    def get(self, *, model: str, input_hash: str) -> list[float] | None:
        with (
            nullcontext(self._conn) if self._conn is not None else closing(get_connection())
        ) as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT embedding FROM embedding_cache_entries WHERE model = %s AND input_hash = %s",
                (model, input_hash),
            )
            row = cur.fetchone()
        return None if row is None else row["embedding"]

    def put(self, *, model: str, input_hash: str, embedding: list[float]) -> None:
        with (
            nullcontext(self._conn) if self._conn is not None else closing(get_connection())
        ) as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO embedding_cache_entries (model, input_hash, embedding)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (model, input_hash) DO NOTHING""",
                (model, input_hash, embedding),
            )
