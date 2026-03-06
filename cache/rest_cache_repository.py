from __future__ import annotations

import hashlib
import json
import os
from datetime import timedelta
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


DB_URL = os.getenv("DATABASE_URL", "postgresql://app:app@localhost:5432/products")

# should be dynamo PG for now
class RestCacheRepository:
    def __init__(self, conn: psycopg.Connection):
        self._conn = conn

    @staticmethod
    def _params_hash(params: dict[str, Any]) -> str:
        canonical = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def get(self, url: str, params: dict[str, Any]) -> Any | None:
        params_hash = self._params_hash(params)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT response_payload FROM rest_cache
                WHERE url = %s AND params_hash = %s AND expires_at > now()
                """,
                (url, params_hash),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return row["response_payload"]

    def put(self, url: str, params: dict[str, Any], response: Any, ttl: timedelta) -> None:
        params_hash = self._params_hash(params)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rest_cache (url, params_hash, params_payload, response_payload, expires_at)
                VALUES (%s, %s, %s, %s, now() + %s)
                ON CONFLICT (url, params_hash) DO UPDATE
                    SET response_payload = EXCLUDED.response_payload,
                        created_at = now(),
                        expires_at = EXCLUDED.expires_at
                """,
                (url, params_hash, Jsonb(params), Jsonb(response), ttl),
            )
            self._conn.commit()
