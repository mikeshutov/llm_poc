from __future__ import annotations

from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from db.connection import get_connection
from llm.clients.embeddings import embed_text
from request_orchestrator.agent_runner.models.agent_profile import AgentExecutionStrategy
from request_orchestrator.agents.models.user_agent import UserAgent, UserAgentModelConfig

MIN_AGENT_SIMILARITY = 0.35


class UserAgentRepository:
    def __init__(self, conn: psycopg.Connection | None = None):
        self._conn = conn or get_connection()
        register_vector(self._conn)

    def _normalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        for field_name in ("created_at", "updated_at"):
            field_value = normalized.get(field_name)
            if field_value is not None and hasattr(field_value, "isoformat"):
                normalized[field_name] = field_value.isoformat()
        return normalized

    def _list_model_configs_by_agent_id(self, agent_ids: list[Any]) -> dict[Any, list[UserAgentModelConfig]]:
        if not agent_ids:
            return {}

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    user_agent_id,
                    stage,
                    provider,
                    model
                FROM user_agent_model_config
                WHERE user_agent_id = ANY(%s)
                ORDER BY stage ASC
                """,
                (agent_ids,),
            )
            rows = cur.fetchall()

        grouped: dict[Any, list[UserAgentModelConfig]] = {}
        for row in rows:
            grouped.setdefault(row["user_agent_id"], []).append(
                UserAgentModelConfig(
                    stage=row["stage"],
                    provider=row["provider"],
                    model=row["model"],
                )
            )
        return grouped

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
                execution_strategy,
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
        model_configs_by_agent_id = self._list_model_configs_by_agent_id([row["id"] for row in rows])
        return [
            UserAgent(
                **self._normalize_row(row),
                model_configs=model_configs_by_agent_id.get(row["id"], []),
            )
            for row in rows
        ]

    def list_relevant_for_user(
        self,
        user_id: str,
        *,
        query_embedding: list[float],
    ) -> list[UserAgent]:
        resolved_user_id = user_id.strip()
        if not resolved_user_id:
            return []

        sql = """
            SELECT
                id,
                user_id,
                name,
                description,
                execution_strategy,
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
              AND is_active = TRUE
              AND description_embedding IS NOT NULL
              AND description_embedding <=> (%s)::vector <= 1 - %s
            ORDER BY description_embedding <=> (%s)::vector ASC, name ASC, created_at ASC
        """
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                sql,
                (
                    resolved_user_id,
                    query_embedding,
                    MIN_AGENT_SIMILARITY,
                    query_embedding,
                ),
            )
            rows = cur.fetchall()
        model_configs_by_agent_id = self._list_model_configs_by_agent_id([row["id"] for row in rows])
        return [
            UserAgent(
                **self._normalize_row(row),
                model_configs=model_configs_by_agent_id.get(row["id"], []),
            )
            for row in rows
        ]

    def upsert(
        self,
        *,
        user_id: str,
        name: str,
        description: str = "",
        execution_strategy: AgentExecutionStrategy = AgentExecutionStrategy.PLANNER_EXECUTOR_EVALUATOR,
        allowed_categories: list[str] | None = None,
        planner_instruction: str,
        planner_rules: str = "",
        max_turns: int = 10,
        is_active: bool = True,
        model_configs: list[UserAgentModelConfig] | None = None,
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
        resolved_execution_strategy = AgentExecutionStrategy(str(execution_strategy).strip())
        resolved_description = description.strip()
        description_embedding = embed_text(resolved_description) if resolved_description else None
        resolved_model_configs = [] if model_configs is None else [
            UserAgentModelConfig(
                stage=config.stage.strip(),
                provider=config.provider.strip(),
                model=config.model.strip(),
            )
            for config in model_configs
        ]
        required_stages = set(resolved_execution_strategy.required_model_stages())
        configured_stages = {config.stage for config in resolved_model_configs}
        missing_stages = sorted(required_stages - configured_stages)
        if missing_stages:
            raise ValueError(
                f"Missing model configs for execution strategy {resolved_execution_strategy.value!r}: {', '.join(missing_stages)}"
            )

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO user_agent (
                    user_id,
                    name,
                    description,
                    description_embedding,
                    execution_strategy,
                    allowed_categories,
                    planner_instruction,
                    planner_rules,
                    max_turns,
                    is_active,
                    metadata
                )
                VALUES (%s, %s, %s, (%s)::vector, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, name)
                DO UPDATE SET
                    description = EXCLUDED.description,
                    description_embedding = EXCLUDED.description_embedding,
                    execution_strategy = EXCLUDED.execution_strategy,
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
                    description_embedding,
                    execution_strategy,
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
                    resolved_description,
                    description_embedding,
                    resolved_execution_strategy.value,
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
            cur.execute(
                """
                DELETE FROM user_agent_model_config
                WHERE user_agent_id = %s
                """,
                (row["id"],),
            )
            if resolved_model_configs:
                cur.executemany(
                    """
                    INSERT INTO user_agent_model_config (
                        user_agent_id,
                        stage,
                        provider,
                        model
                    )
                    VALUES (%s, %s, %s, %s)
                    """,
                    [
                        (
                            row["id"],
                            config.stage,
                            config.provider,
                            config.model,
                        )
                        for config in resolved_model_configs
                    ],
                )
            return UserAgent(
                **self._normalize_row(row),
                model_configs=resolved_model_configs,
            )

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
