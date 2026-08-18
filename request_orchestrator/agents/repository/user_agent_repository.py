from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from db.connection import get_connection
from request_orchestrator.agents.models.user_agent import UserAgent


class UserAgentRepository:
    def __init__(self, conn: psycopg.Connection | None = None):
        self._conn = conn or get_connection()

    def _normalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        for field_name in ("created_at", "updated_at"):
            field_value = normalized.get(field_name)
            if field_value is not None and hasattr(field_value, "isoformat"):
                normalized[field_name] = field_value.isoformat()
        return normalized

    def list_for_user(self, user_id: str, *, is_active: bool | None = True) -> list[UserAgent]:
        resolved_user_id = user_id.strip()
        if not resolved_user_id:
            return []

        sql = """
            SELECT
                id,
                user_id,
                name,
                description,
                allowed_categories,
                planner_instruction,
                planner_rules,
                max_turns,
                is_active,
                metadata,
                created_at,
                updated_at
            FROM user_agent
            WHERE user_id = %s
        """
        params: list[Any] = [resolved_user_id]
        if is_active is not None:
            sql += " AND is_active = %s"
            params.append(is_active)
        sql += " ORDER BY name ASC, created_at ASC"

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [UserAgent(**self._normalize_row(row)) for row in rows]

    def upsert(
        self,
        *,
        user_id: str,
        name: str,
        description: str = "",
        allowed_categories: list[str] | None = None,
        planner_instruction: str,
        planner_rules: str = "",
        max_turns: int = 10,
        is_active: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> UserAgent:
        resolved_user_id = user_id.strip()
        resolved_name = name.strip()
        if not resolved_user_id:
            raise ValueError("user_id is required")
        if not resolved_name:
            raise ValueError("name is required")
        resolved_planner_instruction = planner_instruction.strip()
        if not resolved_planner_instruction:
            raise ValueError("planner_instruction is required")

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO user_agent (
                    user_id,
                    name,
                    description,
                    allowed_categories,
                    planner_instruction,
                    planner_rules,
                    max_turns,
                    is_active,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, name)
                DO UPDATE SET
                    description = EXCLUDED.description,
                    allowed_categories = EXCLUDED.allowed_categories,
                    planner_instruction = EXCLUDED.planner_instruction,
                    planner_rules = EXCLUDED.planner_rules,
                    max_turns = EXCLUDED.max_turns,
                    is_active = EXCLUDED.is_active,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                RETURNING
                    id,
                    user_id,
                    name,
                    description,
                    allowed_categories,
                    planner_instruction,
                    planner_rules,
                    max_turns,
                    is_active,
                    metadata,
                    created_at,
                    updated_at
                """,
                (
                    resolved_user_id,
                    resolved_name,
                    description,
                    allowed_categories or [],
                    resolved_planner_instruction,
                    planner_rules,
                    max_turns,
                    is_active,
                    Jsonb(metadata or {}),
                ),
            )
            row = cur.fetchone()
            assert row is not None
            return UserAgent(**self._normalize_row(row))

    def set_active(self, user_id: str, name: str, *, is_active: bool) -> bool:
        resolved_user_id = user_id.strip()
        resolved_name = name.strip()
        if not resolved_user_id:
            raise ValueError("user_id is required")
        if not resolved_name:
            raise ValueError("name is required")

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE user_agent
                SET is_active = %s,
                    updated_at = now()
                WHERE user_id = %s
                  AND name = %s
                """,
                (is_active, resolved_user_id, resolved_name),
            )
            return cur.rowcount > 0
