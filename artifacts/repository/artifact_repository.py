from __future__ import annotations

from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from artifacts.models import Artifact
from db.connection import get_connection


class ArtifactRepository:
    def __init__(self, conn: psycopg.Connection | None = None):
        self._conn = conn or get_connection()

    @staticmethod
    def _require_user_id(user_id: str) -> str:
        resolved_user_id = user_id.strip()
        if not resolved_user_id:
            raise ValueError("user_id is required")
        return resolved_user_id

    @staticmethod
    def _from_row(row: dict[str, Any]) -> Artifact:
        return Artifact.model_validate(row)

    def create_or_get(
        self,
        *,
        user_id: str,
        artifact_type: str,
        artifact_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        resolved_user_id = self._require_user_id(user_id)
        resolved_type = artifact_type.strip()
        resolved_id = artifact_id.strip()
        if not resolved_type or not resolved_id:
            raise ValueError("artifact_type and artifact_id are required")

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO artifacts (user_id, artifact_type, artifact_id, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, artifact_type, artifact_id)
                DO UPDATE SET updated_at = artifacts.updated_at
                RETURNING id, user_id, artifact_type, artifact_id, metadata, created_at
                """,
                (resolved_user_id, resolved_type, resolved_id, Jsonb(metadata or {})),
            )
            row = cur.fetchone()
            assert row is not None
            return self._from_row(row)

    def get(self, artifact_id: UUID, *, user_id: str) -> Artifact | None:
        resolved_user_id = self._require_user_id(user_id)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, user_id, artifact_type, artifact_id, metadata, created_at
                FROM artifacts
                WHERE id = %s AND user_id = %s
                """,
                (artifact_id, resolved_user_id),
            )
            row = cur.fetchone()
            return self._from_row(row) if row else None

    def list_for_user(self, user_id: str, *, artifact_type: str | None = None, limit: int = 100) -> list[Artifact]:
        resolved_user_id = self._require_user_id(user_id)
        sql = """
            SELECT id, user_id, artifact_type, artifact_id, metadata, created_at
            FROM artifacts
            WHERE user_id = %s
        """
        params: list[Any] = [resolved_user_id]
        if artifact_type is not None:
            resolved_type = artifact_type.strip()
            if not resolved_type:
                raise ValueError("artifact_type cannot be blank")
            sql += " AND artifact_type = %s"
            params.append(resolved_type)
        sql += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return [self._from_row(row) for row in cur.fetchall()]
