from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from common.data.serialization import sanitize_for_json_storage
from common.signatures import build_signature
from db.connection import get_connection


class ToolCacheRepository:
    """SQL cache store behind a replaceable repository boundary."""

    def __init__(self, conn=None):
        self._conn = conn or get_connection()

    @staticmethod
    def build_cache_key(*, tool_name: str, tool_version: int, tool_input: Any, user_id: str | None = None) -> str:
        key = {
            "tool_name": tool_name,
            "tool_version": tool_version,
            "input": tool_input,
        }
        if user_id is not None:
            key["user_id"] = user_id
        return build_signature(key)

    def get(
        self,
        *,
        tool: Any,
        tool_input: Any,
        user_id: str | None = None,
    ) -> Any | None:
        normalized_tool_input = sanitize_for_json_storage(tool_input or {})
        cache_key = self.build_cache_key(
            tool_name=tool.name,
            tool_version=tool.version,
            tool_input=normalized_tool_input,
            user_id=user_id,
        )
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
        tool: Any,
        tool_input: Any,
        output_payload: Any,
        cache_ttl_seconds: int,
        user_id: str | None = None,
    ) -> None:
        normalized_tool_input = sanitize_for_json_storage(tool_input or {})
        normalized_output = sanitize_for_json_storage(output_payload)
        cache_key = self.build_cache_key(
            tool_name=tool.name,
            tool_version=tool.version,
            tool_input=normalized_tool_input,
            user_id=user_id,
        )
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
                    tool.name,
                    tool.version,
                    Jsonb(normalized_tool_input),
                    Jsonb(normalized_output),
                    expires_at,
                ),
            )
