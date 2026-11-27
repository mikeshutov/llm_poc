from contextlib import closing

from psycopg.rows import dict_row

from db.connection import get_connection


def get_cached_embedding(*, model: str, input_hash: str) -> list[float] | None:
    with closing(get_connection()) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT embedding FROM embedding_cache_entries WHERE model = %s AND input_hash = %s",
            (model, input_hash),
        )
        row = cur.fetchone()
    return None if row is None else row["embedding"]


def cache_embedding(*, model: str, input_hash: str, embedding: list[float]) -> None:
    with closing(get_connection()) as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO embedding_cache_entries (model, input_hash, embedding)
               VALUES (%s, %s, %s)
               ON CONFLICT (model, input_hash) DO NOTHING""",
            (model, input_hash, embedding),
        )
