from __future__ import annotations

from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from db.connection import get_connection
from llm.clients.embeddings import embed_text
from request_orchestrator.agent_runner.models.agent_profile import AgentExecutionStrategy
from request_orchestrator.agents.models.agent import Agent, AgentModelConfig, AgentType

MIN_AGENT_SIMILARITY = 0.35


class AgentRepository:
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

    def _list_model_configs_by_agent_id(self, agent_ids: list[Any]) -> dict[Any, list[AgentModelConfig]]:
        if not agent_ids:
            return {}

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    agent_id,
                    stage,
                    provider,
                    model
                FROM agent_model_config
                WHERE agent_id = ANY(%s)
                ORDER BY stage ASC
                """,
                (agent_ids,),
            )
            rows = cur.fetchall()

        grouped: dict[Any, list[AgentModelConfig]] = {}
        for row in rows:
            grouped.setdefault(row["agent_id"], []).append(
                AgentModelConfig(
                    stage=row["stage"],
                    provider=row["provider"],
                    model=row["model"],
                )
            )
        return grouped

    def list_for_user(self, user_id: str, *, is_active: bool | None = True) -> list[Agent]:
        # for now the list call only cares about the users agents post refactor we will load all via this
        resolved_user_id = user_id.strip()
        if not resolved_user_id:
            return []

        sql = """
            SELECT
                id,
                agent_type,
                user_id,
                name,
                version,
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
            FROM agents
            WHERE agent_type = 'user' AND user_id = %s
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
            Agent(
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
    ) -> list[Agent]:
        resolved_user_id = user_id.strip()
        if not resolved_user_id:
            return []

        sql = """
            SELECT
                id,
                agent_type,
                user_id,
                name,
                version,
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
            FROM agents
            WHERE agent_type = 'user' AND user_id = %s
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
            Agent(
                **self._normalize_row(row),
                model_configs=model_configs_by_agent_id.get(row["id"], []),
            )
            for row in rows
        ]

    def _get_agent_by_id(
        self,
        cur: Any,
        *,
        agent_type: AgentType,
        user_id: str | None,
        name: str,
    ) -> Agent | None:
        cur.execute(
            """
            SELECT id, agent_type, user_id, name, version, description,
                   execution_strategy, allowed_categories, planner_instruction,
                   planner_rules, max_turns, is_active, metadata,
                   created_at, updated_at
            FROM agents
            WHERE agent_type = %s
              AND user_id IS NOT DISTINCT FROM %s
              AND name = %s
            """,
            (agent_type.value, user_id, name),
        )
        row = cur.fetchone()
        return None if row is None else Agent(**self._normalize_row(row))

    def _save_model_configs(
        self,
        cur: Any,
        *,
        agent_id: Any,
        model_configs: list[AgentModelConfig] | None,
    ) -> list[AgentModelConfig]:
        if model_configs is not None:
            cur.execute(
                """
                DELETE FROM agent_model_config
                WHERE agent_id = %s
                """,
                (agent_id,),
            )
        if model_configs:
            cur.executemany(
                """
                INSERT INTO agent_model_config (agent_id, stage, provider, model)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (agent_id, stage)
                DO UPDATE SET
                    provider = EXCLUDED.provider,
                    model = EXCLUDED.model,
                    updated_at = now()
                """,
                [
                    (agent_id, config.stage, config.provider, config.model)
                    for config in model_configs
                ],
            )
        cur.execute(
            """
            SELECT stage, provider, model
            FROM agent_model_config
            WHERE agent_id = %s
            ORDER BY stage ASC
            """,
            (agent_id,),
        )
        return [
            AgentModelConfig(
                stage=row["stage"],
                provider=row["provider"],
                model=row["model"],
            )
            for row in cur.fetchall()
        ]

    def _update_agent(
        self,
        cur: Any,
        *,
        agent_id: Any,
        version: int,
        description: str,
        description_embedding: list[float] | None,
        execution_strategy: AgentExecutionStrategy,
        allowed_categories: list[str] | None,
        planner_instruction: str,
        planner_rules: str,
        max_turns: int,
        is_active: bool,
        metadata: dict[str, Any] | None,
        model_configs: list[AgentModelConfig] | None,
    ) -> Agent:
        cur.execute(
            """
            UPDATE agents
            SET description = %s,
                version = %s,
                description_embedding = (%s)::vector,
                execution_strategy = %s,
                allowed_categories = %s,
                planner_instruction = %s,
                planner_rules = %s,
                max_turns = %s,
                is_active = %s,
                metadata = %s,
                updated_at = now()
            WHERE id = %s
            RETURNING id, agent_type, user_id, name, version, description,
                      execution_strategy, allowed_categories, planner_instruction,
                      planner_rules, max_turns, is_active, metadata,
                      created_at, updated_at
            """,
            (
                description,
                version,
                description_embedding,
                execution_strategy.value,
                allowed_categories or [],
                planner_instruction,
                planner_rules,
                max_turns,
                is_active,
                Jsonb(metadata or {}),
                agent_id,
            ),
        )
        row = cur.fetchone()
        assert row is not None
        saved_model_configs = self._save_model_configs(
            cur,
            agent_id=row["id"],
            model_configs=model_configs,
        )
        return Agent(
            **self._normalize_row(row),
            model_configs=saved_model_configs,
        )

    def _insert_agent(
        self,
        cur: Any,
        *,
        agent_type: AgentType,
        user_id: str | None,
        name: str,
        version: int,
        description: str,
        description_embedding: list[float] | None,
        execution_strategy: AgentExecutionStrategy,
        allowed_categories: list[str] | None,
        planner_instruction: str,
        planner_rules: str,
        max_turns: int,
        is_active: bool,
        metadata: dict[str, Any] | None,
        model_configs: list[AgentModelConfig] | None,
    ) -> Agent:
        cur.execute(
            """
            INSERT INTO agents (
                agent_type, user_id, name, version, description, description_embedding,
                execution_strategy, allowed_categories, planner_instruction,
                planner_rules, max_turns, is_active, metadata
            )
            VALUES (%s, %s, %s, %s, %s, (%s)::vector, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, agent_type, user_id, name, version, description,
                      execution_strategy, allowed_categories, planner_instruction,
                      planner_rules, max_turns, is_active, metadata,
                      created_at, updated_at
            """,
            (
                agent_type.value,
                user_id,
                name,
                version,
                description,
                description_embedding,
                execution_strategy.value,
                allowed_categories or [],
                planner_instruction,
                planner_rules,
                max_turns,
                is_active,
                Jsonb(metadata or {}),
            ),
        )
        row = cur.fetchone()
        assert row is not None
        saved_model_configs = self._save_model_configs(
            cur,
            agent_id=row["id"],
            model_configs=model_configs,
        )
        return Agent(
            **self._normalize_row(row),
            model_configs=saved_model_configs,
        )

    def upsert(
        self,
        *,
        user_id: str | None = None,
        agent_type: AgentType = AgentType.USER,
        name: str,
        version: int = 1,
        description: str = "",
        execution_strategy: AgentExecutionStrategy = AgentExecutionStrategy.PLANNER_EXECUTOR_EVALUATOR,
        allowed_categories: list[str] | None = None,
        planner_instruction: str = "",
        planner_rules: str = "",
        max_turns: int = 10,
        is_active: bool = True,
        model_configs: list[AgentModelConfig] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Agent:
        resolved_user_id = user_id.strip() if user_id is not None else None
        resolved_name = name.strip()
        if not resolved_name:
            raise ValueError("name is required")

        resolved_planner_instruction = planner_instruction.strip()
        if not resolved_planner_instruction:
            raise ValueError("planner_instruction is required")
        resolved_execution_strategy = AgentExecutionStrategy(str(execution_strategy).strip())
        resolved_description = description.strip()
        description_embedding = embed_text(resolved_description) if resolved_description else None
        resolved_model_configs = None if model_configs is None else [
            AgentModelConfig(
                stage=config.stage.strip(),
                provider=config.provider.strip(),
                model=config.model.strip(),
            )
            for config in model_configs
        ]

        if version < 1:
            raise ValueError("version must be positive")

        if agent_type == AgentType.SYSTEM:
            if resolved_user_id is not None:
                raise ValueError("system agents cannot have a user_id")
        elif agent_type == AgentType.USER:
            if not resolved_user_id:
                raise ValueError("user_id is required")
            required_stages = set(resolved_execution_strategy.required_model_stages())
            configured_stages = {config.stage for config in resolved_model_configs or []}
            missing_stages = sorted(required_stages - configured_stages)
            if missing_stages:
                raise ValueError(
                    f"Missing model configs for execution strategy {resolved_execution_strategy.value!r}: {', '.join(missing_stages)}"
                )
        else:
            raise ValueError(f"unsupported agent_type: {agent_type!r}")

        with self._conn.cursor(row_factory=dict_row) as cur:
            existing_agent = self._get_agent_by_id(
                cur,
                agent_type=agent_type,
                user_id=resolved_user_id,
                name=resolved_name,
            )
            common_args = {
                "version": version,
                "description": resolved_description,
                "description_embedding": description_embedding,
                "execution_strategy": resolved_execution_strategy,
                "allowed_categories": allowed_categories,
                "planner_instruction": resolved_planner_instruction,
                "planner_rules": planner_rules,
                "max_turns": max_turns,
                "is_active": is_active,
                "metadata": metadata,
                "model_configs": resolved_model_configs,
            }
            if existing_agent is not None:
                if agent_type == AgentType.SYSTEM and existing_agent.version == version:
                    return existing_agent
                return self._update_agent(cur, agent_id=existing_agent.id, **common_args)
            return self._insert_agent(
                cur,
                agent_type=agent_type,
                user_id=resolved_user_id,
                name=resolved_name,
                **common_args,
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
                UPDATE agents
                SET is_active = %s,
                    updated_at = now()
                WHERE agent_type = 'user' AND user_id = %s
                  AND name = %s
                """,
                (is_active, resolved_user_id, resolved_name),
            )
            return cur.rowcount > 0
