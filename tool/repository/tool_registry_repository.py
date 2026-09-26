from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from db.connection import get_connection
from llm.clients.embeddings import embed_text
from tool.models import RegisteredTool

MIN_TOOL_SIMILARITY = 0.35


class ToolRegistryRepository:
    def __init__(self, conn: psycopg.Connection | None = None):
        self._conn = conn or get_connection()
        register_vector(self._conn)

    @staticmethod
    def _normalize_row(row: dict) -> dict:
        normalized = dict(row)
        for field_name in ("created_at", "updated_at"):
            value = normalized.get(field_name)
            if value is not None and hasattr(value, "isoformat"):
                normalized[field_name] = value
        return normalized

    def list_all(self) -> list[RegisteredTool]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, name, version, description, result_type,
                       rate_limit_key, retry_policy, rate_limit_policy,
                       is_active, metadata, created_at, updated_at
                FROM tools
                ORDER BY name ASC, created_at ASC
                """
            )
            rows = cur.fetchall()
        return [RegisteredTool(**self._normalize_row(row)) for row in rows]

    def upsert(self, definition: RegisteredTool) -> RegisteredTool:
        tool_embedding = embed_text(definition.description) if definition.description else None
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO tools (
                    name, version, description, tool_embedding, result_type, rate_limit_key,
                    retry_policy, rate_limit_policy, is_active, metadata
                )
                VALUES (%s, %s, %s, (%s)::vector, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (name)
                DO UPDATE SET
                    version = EXCLUDED.version,
                    description = EXCLUDED.description,
                    tool_embedding = EXCLUDED.tool_embedding,
                    result_type = EXCLUDED.result_type,
                    rate_limit_key = EXCLUDED.rate_limit_key,
                    retry_policy = EXCLUDED.retry_policy,
                    rate_limit_policy = EXCLUDED.rate_limit_policy,
                    is_active = EXCLUDED.is_active,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                RETURNING id, name, version, description, result_type,
                          rate_limit_key, retry_policy, rate_limit_policy,
                          is_active, metadata, created_at, updated_at
                """,
                (
                    definition.name,
                    definition.version,
                    definition.description,
                    tool_embedding,
                    definition.result_type,
                    definition.rate_limit_key,
                    Jsonb(asdict(definition.retry_policy)),
                    None if definition.rate_limit_policy is None else Jsonb(asdict(definition.rate_limit_policy)),
                    definition.is_active,
                    Jsonb(definition.metadata),
                ),
            )
            row = cur.fetchone()
        assert row is not None
        return RegisteredTool(**self._normalize_row(row))

    def list_relevant(
        self,
        *,
        query_embedding: list[float],
        is_active: bool = True,
    ) -> list[RegisteredTool]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, name, version, description, result_type,
                       rate_limit_key, retry_policy, rate_limit_policy,
                       is_active, metadata, created_at, updated_at
                FROM tools
                WHERE is_active = %s
                  AND tool_embedding IS NOT NULL
                  AND tool_embedding <=> (%s)::vector <= 1 - %s
                ORDER BY tool_embedding <=> (%s)::vector ASC,
                         name ASC, created_at ASC
                """,
                (is_active, query_embedding, MIN_TOOL_SIMILARITY, query_embedding),
            )
            rows = cur.fetchall()
        return [RegisteredTool(**self._normalize_row(row)) for row in rows]

    def delete_names(self, names: set[str]) -> int:
        if not names:
            return 0
        with self._conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM tools
                WHERE name = ANY(%s)
                """,
                (list(names),),
            )
            return cur.rowcount

    def sync(self, definitions: Iterable[RegisteredTool]) -> list[RegisteredTool]:
        definitions = tuple(definitions)
        definition_names = {definition.name for definition in definitions}

        with self._conn.transaction():
            existing_names = {definition.name for definition in self.list_all()}
            self.delete_names(existing_names - definition_names)
            return [self.upsert(definition) for definition in definitions]
