from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from db.connection import get_connection
from personalization.profile.models import UserProfile


class UserProfileRepository:
    def __init__(self, conn: psycopg.Connection | None = None):
        self._conn = conn or get_connection()

    def _normalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        for field_name in ("created_at", "updated_at"):
            field_value = normalized.get(field_name)
            if field_value is not None and hasattr(field_value, "isoformat"):
                normalized[field_name] = field_value.isoformat()
        return normalized

    def ensure_profile(
        self,
        user_id: str,
        *,
        first_name: str | None = None,
        last_name: str | None = None,
        display_name: str | None = None,
        email: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UserProfile:
        resolved_user_id = user_id.strip()
        if not resolved_user_id:
            raise ValueError("user_id is required")

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO user_profile (user_id, first_name, last_name, display_name, email, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id)
                DO UPDATE SET
                    first_name = COALESCE(EXCLUDED.first_name, user_profile.first_name),
                    last_name = COALESCE(EXCLUDED.last_name, user_profile.last_name),
                    display_name = COALESCE(EXCLUDED.display_name, user_profile.display_name),
                    email = COALESCE(EXCLUDED.email, user_profile.email),
                    metadata = CASE
                        WHEN EXCLUDED.metadata = '{}'::jsonb THEN user_profile.metadata
                        ELSE user_profile.metadata || EXCLUDED.metadata
                    END,
                    updated_at = now()
                RETURNING user_id, first_name, last_name, display_name, email, metadata, created_at, updated_at
                """,
                (resolved_user_id, first_name, last_name, display_name, email, Jsonb(metadata or {})),
            )
            row = cur.fetchone()
            assert row is not None
            return UserProfile(**self._normalize_row(row))

    def get_profile(self, user_id: str) -> UserProfile | None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT user_id, first_name, last_name, display_name, email, metadata, created_at, updated_at
                FROM user_profile
                WHERE user_id = %s
                """,
                (user_id,),
            )
            row = cur.fetchone()
            return UserProfile(**self._normalize_row(row)) if row else None

    def update_profile(
        self,
        user_id: str,
        first_name: str | None = None,
        last_name: str | None = None,
        display_name: str | None = None,
        email: str | None = None,
    ) -> UserProfile | None:
        resolved_user_id = user_id.strip()
        if not resolved_user_id:
            raise ValueError("user_id is required")

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE user_profile
                SET first_name = %s,
                    last_name = %s,
                    display_name = %s,
                    email = %s,
                    updated_at = now()
                WHERE user_id = %s
                RETURNING user_id, first_name, last_name, display_name, email, metadata, created_at, updated_at
                """,
                (first_name, last_name, display_name, email, resolved_user_id),
            )
            row = cur.fetchone()
            return UserProfile(**self._normalize_row(row)) if row else None

    def list_profiles(self, limit: int = 100) -> list[UserProfile]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT user_id, first_name, last_name, display_name, email, metadata, created_at, updated_at
                FROM user_profile
                ORDER BY COALESCE(NULLIF(BTRIM(display_name), ''), NULLIF(BTRIM(user_id), '')) ASC, created_at ASC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
            return [UserProfile(**self._normalize_row(row)) for row in rows]
