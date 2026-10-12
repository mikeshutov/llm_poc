from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from common.signatures import build_signature
from db.connection import get_connection


class ToolCacheRepository:
    """SQL cache store behind a replaceable repository boundary."""

    def __init__(self, conn=None):
        self._conn = conn or get_connection()

    @staticmethod
    def build_key(*, tool_name: str, tool_version: int, tool_input: Any, user_id: str | None = None) -> str:
        key = {
            "tool_name": tool_name,
            "tool_version": tool_version,
            "input": tool_input,
        }
        if user_id is not None:
            key["user_id"] = user_id
        return build_signature(key)

    def get(self, *, cache_key: str) -> Any | None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT output_payload
                FROM tool_cache_entries
                WHERE cache_key = %s AND expires_at > now()
                """,
                (cache_key,),
            )
            row = cur.fetchone()
        return None if row is None else row["output_payload"]

    def put(
        self,
        *,
        cache_key: str,
        tool_name: str,
        tool_version: int,
        input_payload: Any,
        output_payload: Any,
        cache_ttl_seconds: int,
    ) -> None:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=cache_ttl_seconds)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tool_cache_entries (
                    cache_key, tool_name, tool_version, input_payload,
                    output_payload, expires_at
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (cache_key) DO UPDATE SET
                    tool_name = EXCLUDED.tool_name,
                    tool_version = EXCLUDED.tool_version,
                    input_payload = EXCLUDED.input_payload,
                    output_payload = EXCLUDED.output_payload,
                    expires_at = EXCLUDED.expires_at,
                    updated_at = now()
                """,
                (
                    cache_key,
                    tool_name,
                    tool_version,
                    Jsonb(input_payload),
                    Jsonb(output_payload),
                    expires_at,
                ),
            )

    @staticmethod
    def sanitize(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return ToolCacheRepository.sanitize(value.model_dump(mode="json"))
        if isinstance(value, dict):
            return {str(key): ToolCacheRepository.sanitize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [ToolCacheRepository.sanitize(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
