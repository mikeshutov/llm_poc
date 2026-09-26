from __future__ import annotations

from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from db.connection import get_connection
from llm.conversation_model_config import (
    CONVERSATION_MODEL_CONFIG_SPECS,
    ConversationModelConfig,
    ConversationModelConfigEntry,
)
from llm.model_config_resolver import resolve_conversation_model_config


class ConversationModelConfigRepository:
    def __init__(self, conn: psycopg.Connection | None = None):
        self._conn = conn or get_connection()

    def ensure_defaults(
        self,
        conversation_id: UUID,
        entries: list[ConversationModelConfigEntry] | None = None,
    ) -> list[ConversationModelConfigEntry]:
        existing_entries = list(entries) if entries is not None else self.list(conversation_id)
        existing_keys = {(entry.agent, entry.stage) for entry in existing_entries}
        default_config = ConversationModelConfig.build_default()
        missing_entries: list[ConversationModelConfigEntry] = []

        for spec in CONVERSATION_MODEL_CONFIG_SPECS:
            if (spec.agent, spec.stage) in existing_keys:
                continue
            provider = default_config.resolve_provider(spec.agent, spec.stage)
            model = default_config.resolve(spec.agent, spec.stage)
            persisted_entry = self.upsert(conversation_id, spec.agent, spec.stage, provider, model)
            missing_entries.append(persisted_entry)

        return [*existing_entries, *missing_entries]

    def list(self, conversation_id: UUID) -> list[ConversationModelConfigEntry]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT conversation_id, agent, stage, provider, model, created_at, updated_at
                FROM conversation_model_config
                WHERE conversation_id = %s
                ORDER BY agent ASC, stage ASC
                """,
                (conversation_id,),
            )
            rows = cur.fetchall()
            return [
                ConversationModelConfigEntry(
                    conversation_id=row["conversation_id"],
                    agent=row["agent"],
                    stage=row["stage"],
                    provider=row.get("provider") or "openai",
                    model=row["model"],
                    created_at=str(row["created_at"]),
                    updated_at=str(row["updated_at"]),
                )
                for row in rows
            ]

    def upsert(
        self,
        conversation_id: UUID,
        agent: str,
        stage: str,
        provider: str,
        model: str,
    ) -> ConversationModelConfigEntry:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO conversation_model_config (conversation_id, agent, stage, provider, model)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (conversation_id, agent, stage)
                DO UPDATE SET
                    provider = EXCLUDED.provider,
                    model = EXCLUDED.model,
                    updated_at = now()
                RETURNING conversation_id, agent, stage, provider, model, created_at, updated_at
                """,
                (conversation_id, agent, stage, provider, model),
            )
            row = cur.fetchone()
            assert row is not None
            return ConversationModelConfigEntry(
                conversation_id=row["conversation_id"],
                agent=row["agent"],
                stage=row["stage"],
                provider=row.get("provider") or provider,
                model=row["model"],
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
            )

    def clear(self, conversation_id: UUID, agent: str, stage: str) -> bool:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                DELETE FROM conversation_model_config
                WHERE conversation_id = %s
                  AND agent = %s
                  AND stage = %s
                """,
                (conversation_id, agent, stage),
            )
            return cur.rowcount > 0

    def clear_all(self, conversation_id: UUID) -> bool:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                DELETE FROM conversation_model_config
                WHERE conversation_id = %s
                """,
                (conversation_id,),
            )
            return cur.rowcount > 0

    def resolve(self, conversation_id: UUID) -> ConversationModelConfig:
        entries = self.list(conversation_id)
        ensured_entries = self.ensure_defaults(conversation_id, entries)
        return resolve_conversation_model_config(ensured_entries)
