from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional, Sequence
from uuid import UUID

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from conversation.models.conversation_models import (
    Conversation,
    ConversationMessage,
    ConversationMetadata,
    ConversationMemory,
    ConversationRoundtrip,
    LlmCallRecord,
    ConversationSummary,
    ConversationMessageKind,
    ConversationMessageVisibility,
    NodeLog,
    RoundtripFeedback,
    RoundtripMemory,
)
from common.data import bounded_json_value
from db.connection import get_connection
from llm.repository.conversation_model_config_repository import ConversationModelConfigRepository
from personalization.tone.models import TonePreferences
from agent_runtime.constants import NodeName
from agent_runtime.models.orchestrator_payload import OrchestratorPayload


class ConversationRepository:
    def __init__(self, conn: psycopg.Connection | None = None):
        self._conn = conn or get_connection()
        register_vector(self._conn)

    def append_message(
        self,
        roundtrip_id: UUID,
        *,
        kind: ConversationMessageKind | str,
        role: str,
        content: str = "",
        visibility: ConversationMessageVisibility | str = ConversationMessageVisibility.VISIBLE,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        parent_message_id: UUID | None = None,
    ) -> ConversationMessage:
        message_embedding = None
        embedding_text = (content or "").strip()
        if embedding_text and str(visibility) == "visible" and str(kind) in {"user", "assistant", "synthesis"}:
            try:
                from llm.clients.embeddings import embed_text
                message_embedding = embed_text(embedding_text)
            except Exception:
                message_embedding = None
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT id FROM conversation_roundtrip WHERE id = %s FOR UPDATE", (roundtrip_id,))
            cur.execute(
                """
                INSERT INTO conversation_message (
                    roundtrip_id, sequence, kind, visibility, role, content,
                    payload, metadata, parent_message_id, embedding
                )
                SELECT %s, COALESCE(MAX(sequence), -1) + 1, %s, %s, %s, %s, %s, %s, %s, (%s)::vector
                FROM conversation_message
                WHERE roundtrip_id = %s
                RETURNING id, roundtrip_id, sequence, kind, visibility, role,
                          content, payload, metadata, parent_message_id, created_at, embedding
                """,
                (
                    roundtrip_id,
                    kind.value if isinstance(kind, ConversationMessageKind) else str(kind),
                    visibility.value if isinstance(visibility, ConversationMessageVisibility) else str(visibility),
                    role,
                    content or "",
                    Jsonb(bounded_json_value(payload or {}, max_bytes=256_000)),
                    Jsonb(bounded_json_value(metadata or {}, max_bytes=16_000)),
                    parent_message_id,
                    message_embedding,
                    roundtrip_id,
                ),
            )
            row = cur.fetchone()
            assert row is not None
            return ConversationMessage(**row)

    def append_node_log(
        self,
        scope_type: str,
        scope_id: UUID,
        *,
        node_name: NodeName | str,
        stage: str = "",
        event_type: str,
        level: str = "info",
        message: str = "",
        payload: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        correlation_id: UUID | None = None,
    ) -> NodeLog:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"{scope_type}:{scope_id}",),
            )
            cur.execute(
                """
                INSERT INTO node_log (
                    scope_type, scope_id, sequence, node_name, stage, event_type, level,
                    message, payload, duration_ms, correlation_id
                )
                SELECT %s, %s, COALESCE(MAX(sequence), -1) + 1, %s, %s, %s, %s,
                       %s, %s, %s, %s
                FROM node_log
                WHERE scope_type = %s AND scope_id = %s
                RETURNING id, scope_type, scope_id, sequence, node_name, stage, event_type,
                          level, message, payload, duration_ms, correlation_id, created_at
                """,
                (
                    scope_type,
                    scope_id,
                    node_name.value if hasattr(node_name, "value") else str(node_name),
                    stage,
                    event_type,
                    level,
                    message[:2_000],
                    Jsonb(bounded_json_value(payload or {}, max_bytes=64_000)),
                    duration_ms,
                    correlation_id,
                    scope_type,
                    scope_id,
                ),
            )
            row = cur.fetchone()
            assert row is not None
            return NodeLog(**row)

    def list_node_logs(
        self,
        scope_type: str,
        scope_id: UUID,
        *,
        node_name: NodeName | str | None = None,
        stage: str | None = None,
        event_type: str | None = None,
        level: str | None = None,
    ) -> list[NodeLog]:
        filters = ["scope_type = %s", "scope_id = %s"]
        if node_name is not None and hasattr(node_name, "value"):
            node_name = node_name.value
        params: list[Any] = [scope_type, scope_id]
        for column, value in (("node_name", node_name), ("stage", stage), ("event_type", event_type), ("level", level)):
            if value:
                filters.append(f"{column} = %s")
                params.append(value)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
                SELECT id, scope_type, scope_id, sequence, node_name, stage, event_type,
                       level, message, payload, duration_ms, correlation_id, created_at
                FROM node_log
                WHERE {' AND '.join(filters)}
                ORDER BY sequence ASC
                """,
                params,
            )
            return [NodeLog(**row) for row in cur.fetchall()]

    def list_messages(
        self,
        roundtrip_id: UUID,
        *,
        include_internal: bool = True,
    ) -> list[ConversationMessage]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            query = """
                SELECT id, roundtrip_id, sequence, kind, visibility, role,
                       content, payload, metadata, parent_message_id, created_at, embedding
                FROM conversation_message
                WHERE roundtrip_id = %s
            """
            params: list[Any] = [roundtrip_id]
            if not include_internal:
                query += " AND visibility = 'visible'"
            query += " ORDER BY sequence ASC"
            cur.execute(query, params)
            return [ConversationMessage(**row) for row in cur.fetchall()]

    def list_messages_for_conversation(
        self,
        conversation_id: UUID,
        *,
        include_internal: bool = True,
    ) -> list[ConversationMessage]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            visibility_clause = "" if include_internal else " AND cm.visibility = 'visible'"
            cur.execute(
                f"""
                SELECT cm.id, cm.roundtrip_id, cm.sequence, cm.kind, cm.visibility,
                       cm.role, cm.content, cm.payload, cm.metadata,
                       cm.parent_message_id, cm.created_at, cm.embedding
                FROM conversation_message cm
                JOIN conversation_roundtrip rt ON rt.id = cm.roundtrip_id
                WHERE rt.conversation_id = %s{visibility_clause}
                ORDER BY rt.message_index ASC, cm.sequence ASC
                """,
                (conversation_id,),
            )
            return [ConversationMessage(**row) for row in cur.fetchall()]

    def _hydrate_roundtrip(self, row: dict[str, Any]) -> ConversationRoundtrip:
        return ConversationRoundtrip(**row)

    def create_conversation(
        self,
        user_id: str,
        metadata: ConversationMetadata | None = None,
    ) -> Conversation:
        metadata = metadata or ConversationMetadata()
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO conversation (user_id, metadata, title, tone_state)
                VALUES (%s, %s, %s, %s)
                RETURNING id, user_id, title, created_at, metadata, tone_state, summary
                """,
                (
                    user_id,
                    Jsonb(metadata.model_dump(mode="json", exclude_none=True)),
                    "Unnamed",
                    Jsonb({}),
                ),
            )
            row = cur.fetchone()
            assert row is not None
            conversation = Conversation(**row)

        ConversationModelConfigRepository(self._conn).ensure_defaults(conversation.id)
        return conversation

    def create_pending_roundtrip(
        self,
        conversation_id: UUID,
        user_prompt: str,
        model: Optional[str] = None,
        roundtrip_summary: Optional[str] = None,
        roundtrip_summary_embedding: Optional[list[float]] = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationRoundtrip:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO conversation_roundtrip (conversation_id, message_index, status, roundtrip_summary, roundtrip_summary_embedding, metadata)
                SELECT %s, COALESCE(MAX(message_index), -1) + 1, 'pending', %s, (%s)::vector, %s
                FROM conversation_roundtrip
                WHERE conversation_id = %s
                RETURNING id, conversation_id, message_index, status, roundtrip_summary, roundtrip_summary_embedding, created_at, metadata
                """,
                (
                    conversation_id,
                    roundtrip_summary,
                    roundtrip_summary_embedding,
                    Jsonb({**(metadata or {}), **({"model": model} if model else {})}),
                    conversation_id,
                ),
            )
            row = cur.fetchone()
            assert row is not None
            roundtrip = ConversationRoundtrip(**row)
        self.append_message(
            roundtrip.id,
            kind=ConversationMessageKind.USER,
            role="user",
            content=user_prompt,
        )
        return roundtrip

    def update_roundtrip(
        self,
        roundtrip_id: UUID,
        response: str,
        payload: OrchestratorPayload,
        roundtrip_summary: Optional[str] = None,
        roundtrip_summary_embedding: Optional[list[float]] = None,
    ) -> ConversationRoundtrip:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE conversation_roundtrip
                SET status = 'completed',
                    roundtrip_summary = COALESCE(%s, roundtrip_summary),
                    roundtrip_summary_embedding = COALESCE((%s)::vector, roundtrip_summary_embedding),
                    updated_at = now()
                WHERE id = %s
                RETURNING id, conversation_id, message_index, status, roundtrip_summary, roundtrip_summary_embedding, created_at, metadata
                """,
                (
                    roundtrip_summary,
                    roundtrip_summary_embedding,
                    roundtrip_id,
                ),
            )
            row = cur.fetchone()
            assert row is not None
            roundtrip = ConversationRoundtrip(**row)
        synthesis_payload = payload.model_dump(mode="json", exclude_none=True)
        model = (roundtrip.metadata or {}).get("model")
        self.append_message(
            roundtrip_id,
            kind=ConversationMessageKind.ASSISTANT,
            role="assistant",
            content=response,
            payload=synthesis_payload,
            metadata={"model": model, "stage": "synthesis"} if model else {"stage": "synthesis"},
        )
        return roundtrip

    def get_roundtrip_feedback(self, roundtrip_id: UUID) -> Optional[RoundtripFeedback]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, roundtrip_id, met_expectation, reason, expected_answer, created_at, model
                FROM roundtrip_feedback
                WHERE roundtrip_id = %s
                """,
                (roundtrip_id,),
            )
            row = cur.fetchone()
            return RoundtripFeedback(**row) if row else None

    def upsert_roundtrip_feedback(
        self,
        roundtrip_id: UUID,
        met_expectation: bool,
        reason: Optional[str] = None,
        expected_answer: Optional[str] = None,
        model: Optional[str] = None,
    ) -> RoundtripFeedback:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO roundtrip_feedback (roundtrip_id, met_expectation, reason, expected_answer, model)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (roundtrip_id)
                DO UPDATE SET
                    met_expectation = EXCLUDED.met_expectation,
                    reason = EXCLUDED.reason,
                    expected_answer = EXCLUDED.expected_answer,
                    model = EXCLUDED.model
                RETURNING id, roundtrip_id, met_expectation, reason, expected_answer, created_at, model
                """,
                (roundtrip_id, met_expectation, reason, expected_answer, model),
            )
            row = cur.fetchone()
            assert row is not None
            return RoundtripFeedback(**row)

    def create_summary(
        self,
        conversation_id: UUID,
        summary: str,
        message_index_cutoff: int,
    ) -> ConversationSummary:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO conversation_summary (conversation_id, summary, message_index_cutoff)
                VALUES (%s, %s, %s)
                RETURNING id, conversation_id, summary, message_index_cutoff, created_at
                """,
                (conversation_id, summary, message_index_cutoff),
            )
            row = cur.fetchone()
            assert row is not None
            return ConversationSummary(**row)

    def update_conversation_summary(
        self,
        conversation_id: UUID,
        summary: str,
        summary_embedding: Optional[list[float]] = None,
    ) -> None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE conversation
                SET summary = %s,
                    summary_embedding = (%s)::vector,
                    updated_at = now()
                WHERE id = %s
                """,
                (summary, summary_embedding, conversation_id),
            )

    def get_summary(
        self,
        conversation_id: UUID,
    ) -> Optional[ConversationSummary]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, conversation_id, summary, message_index_cutoff, created_at
                FROM conversation_summary
                WHERE conversation_id = %s
                """,
                (conversation_id,),
            )
            row = cur.fetchone()
            return ConversationSummary(**row) if row else None

    def get_roundtrip(self, roundtrip_id: UUID) -> Optional[ConversationRoundtrip]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    rt.id,
                    rt.conversation_id,
                    rt.message_index,
                    rt.status,
                    rt.roundtrip_summary,
                    rt.roundtrip_summary_embedding,
                    rt.created_at,
                    rt.metadata
                FROM conversation_roundtrip rt
                WHERE rt.id = %s
                """,
                (roundtrip_id,),
            )
            row = cur.fetchone()
            return self._hydrate_roundtrip(row) if row else None

    def get_roundtrip_for_user(self, roundtrip_id: UUID, user_id: str | None) -> Optional[ConversationRoundtrip]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    rt.id,
                    rt.conversation_id,
                    rt.message_index,
                    rt.status,
                    rt.roundtrip_summary,
                    rt.roundtrip_summary_embedding,
                    rt.created_at,
                    rt.metadata
                FROM conversation_roundtrip rt
                JOIN conversation c ON c.id = rt.conversation_id
                WHERE rt.id = %s
                  AND (CAST(%s AS text) IS NULL OR c.user_id = %s)
                """,
                (roundtrip_id, user_id, user_id),
            )
            row = cur.fetchone()
            return self._hydrate_roundtrip(row) if row else None

    def get_conversation(self, conversation_id: UUID) -> Optional[Conversation]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, user_id, title, created_at, metadata, tone_state, summary, summary_embedding
                FROM conversation
                WHERE id = %s
                """,
                (conversation_id,),
            )
            row = cur.fetchone()
            return Conversation(**row) if row else None

    def list_roundtrips(
        self,
        conversation_id: UUID,
        limit: int = 50,
        after_message_index: Optional[int] = None,
        newest_first: bool = False,
    ) -> list[ConversationRoundtrip]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            order_direction = "DESC" if newest_first else "ASC"
            if after_message_index is None:
                cur.execute(
                    f"""
                    SELECT
                        rt.id,
                        rt.conversation_id,
                        rt.message_index,
                        rt.status,
                        rt.roundtrip_summary,
                        rt.roundtrip_summary_embedding,
                        rt.created_at,
                        rt.metadata
                    FROM conversation_roundtrip rt
                    WHERE rt.conversation_id = %s
                      AND rt.status = 'completed'
                    ORDER BY rt.message_index {order_direction}
                    LIMIT %s
                    """,
                    (conversation_id, limit),
                )
            else:
                cur.execute(
                    f"""
                    SELECT
                        rt.id,
                        rt.conversation_id,
                        rt.message_index,
                        rt.status,
                        rt.roundtrip_summary,
                        rt.roundtrip_summary_embedding,
                        rt.created_at,
                        rt.metadata
                    FROM conversation_roundtrip rt
                    WHERE rt.conversation_id = %s
                      AND rt.message_index > %s
                      AND rt.status = 'completed'
                    ORDER BY rt.message_index {order_direction}
                    LIMIT %s
                    """,
                    (conversation_id, after_message_index, limit),
                )
            rows = cur.fetchall()
            roundtrips = [self._hydrate_roundtrip(r) for r in rows]
            if newest_first:
                roundtrips.reverse()
            return roundtrips

    def list_roundtrips_through_message_index(
        self,
        conversation_id: UUID,
        message_index: int,
    ) -> list[ConversationRoundtrip]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    rt.id,
                    rt.conversation_id,
                    rt.message_index,
                    rt.status,
                    rt.roundtrip_summary,
                    rt.roundtrip_summary_embedding,
                    rt.created_at,
                    rt.metadata
                FROM conversation_roundtrip rt
                WHERE rt.conversation_id = %s
                  AND rt.message_index <= %s
                  AND rt.status = 'completed'
                ORDER BY rt.message_index ASC
                """,
                (conversation_id, message_index),
            )
            rows = cur.fetchall()
            return [self._hydrate_roundtrip(r) for r in rows]

    def get_latest_summary(self, conversation_id: UUID) -> Optional[ConversationSummary]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, conversation_id, summary, message_index_cutoff, created_at
                FROM conversation_summary
                WHERE conversation_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (conversation_id,),
            )
            row = cur.fetchone()
            return ConversationSummary(**row) if row else None

    def list_conversations(self, user_id: str, limit: int = 50) -> list[Conversation]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, user_id, created_at, metadata, title, tone_state, summary, summary_embedding
                FROM conversation
                WHERE user_id = %s
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            rows = cur.fetchall()
            return [Conversation(**r) for r in rows]

    def count_conversations(self, user_id: str) -> int:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS conversation_count
                FROM conversation
                WHERE user_id = %s
                """,
                (user_id,),
            )
            row = cur.fetchone()
            return int(row["conversation_count"]) if row else 0

    def copy_roundtrip_to_conversation(
        self,
        conversation_id: UUID,
        source_roundtrip: ConversationRoundtrip,
    ) -> ConversationRoundtrip:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO conversation_roundtrip (
                    conversation_id,
                    message_index,
                    status,
                    roundtrip_summary,
                    roundtrip_summary_embedding,
                    created_at,
                    updated_at,
                    metadata
                )
                VALUES (%s, %s, 'completed', %s, (%s)::vector, %s, %s, %s)
                RETURNING id, conversation_id, message_index, status, roundtrip_summary, roundtrip_summary_embedding, created_at, metadata
                """,
                (
                    conversation_id,
                    source_roundtrip.message_index,
                    source_roundtrip.roundtrip_summary,
                    source_roundtrip.roundtrip_summary_embedding,
                    source_roundtrip.created_at,
                    source_roundtrip.created_at,
                    Jsonb(source_roundtrip.metadata),
                ),
            )
            row = cur.fetchone()
            assert row is not None
            copied = ConversationRoundtrip(**row)
        message_id_map: dict[UUID, UUID] = {}
        for message in self.list_messages(source_roundtrip.id):
            copied_message = self.append_message(
                copied.id,
                kind=message.kind,
                role=message.role,
                content=message.content,
                visibility=message.visibility,
                payload=message.payload,
                metadata=message.metadata,
                parent_message_id=(
                    message_id_map.get(message.parent_message_id)
                    if message.parent_message_id is not None
                    else None
                ),
            )
            message_id_map[message.id] = copied_message.id
        return self._hydrate_roundtrip({**row, "id": copied.id})

    def delete_roundtrip(self, roundtrip_id: UUID) -> None:
        """Delete a roundtrip and its persisted message/runtime children."""
        with self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM node_log WHERE scope_type = 'roundtrip' AND scope_id = %s",
                (roundtrip_id,),
            )
            cur.execute(
                "DELETE FROM conversation_roundtrip WHERE id = %s",
                (roundtrip_id,),
            )

    def search_conversation_memories(
        self,
        query_embedding: Sequence[float],
        limit: int = 5,
        user_id: str | None = None,
    ) -> list[ConversationMemory]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    id AS conversation_id,
                    summary,
                    updated_at AS last_used_date,
                    (summary_embedding <-> (%s)::vector) AS relevance_score
                FROM conversation
                WHERE summary_embedding IS NOT NULL
                  AND BTRIM(summary) <> ''
                  AND (CAST(%s AS text) IS NULL OR user_id = %s)
                ORDER BY summary_embedding <-> (%s)::vector ASC
                LIMIT %s
                """,
                (list(query_embedding), user_id, user_id, list(query_embedding), limit),
            )
            rows = cur.fetchall()
            return [
                ConversationMemory(
                    conversation_id=row['conversation_id'],
                    summary=row['summary'],
                    last_used_date=str(row['last_used_date']),
                    relevance_score=float(row['relevance_score']),
                )
                for row in rows
            ]

    def search_roundtrip_memories(
        self,
        query_embedding: Sequence[float],
        conversation_ids: Sequence[UUID] | None = None,
        limit: int = 5,
        user_id: str | None = None,
    ) -> list[RoundtripMemory]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            filters = [
                "(CAST(%s AS text) IS NULL OR c.user_id = %s)",
                "rt.roundtrip_summary_embedding IS NOT NULL",
                "BTRIM(COALESCE(rt.roundtrip_summary, '')) <> ''",
            ]
            params: list[Any] = [user_id, user_id]
            if conversation_ids:
                filters.insert(0, "rt.conversation_id = ANY(%s)")
                params.insert(0, list(conversation_ids))
            params.extend([list(query_embedding), max(1, limit)])
            cur.execute(
                f"""
                SELECT
                    rt.conversation_id,
                    rt.id AS roundtrip_id,
                    rt.message_index,
                    rt.roundtrip_summary,
                    rt.created_at,
                    (rt.roundtrip_summary_embedding <-> (%s)::vector) AS relevance_score
                FROM conversation_roundtrip rt
                JOIN conversation c ON c.id = rt.conversation_id
                WHERE {' AND '.join(filters)}
                ORDER BY rt.roundtrip_summary_embedding <-> (%s)::vector ASC
                LIMIT %s
                """,
                [list(query_embedding), *params[:-2], list(query_embedding), params[-1]],
            )
            rows = cur.fetchall()
            memories = []
            for row in rows:
                memories.append(RoundtripMemory(
                    conversation_id=row['conversation_id'],
                    roundtrip_id=row['roundtrip_id'],
                    message_index=row['message_index'],
                    roundtrip_summary=row['roundtrip_summary'],
                    created_at=str(row['created_at']),
                    relevance_score=float(row['relevance_score']),
                    messages=self.list_messages(row["roundtrip_id"], include_internal=True),
                ))
            return memories

    def search_message_memories(
        self,
        query_embedding: Sequence[float],
        conversation_ids: Sequence[UUID] | None = None,
        limit: int = 5,
        user_id: str | None = None,
    ) -> list[RoundtripMemory]:
        """Search individual message embeddings, then return complete roundtrips."""
        with self._conn.cursor(row_factory=dict_row) as cur:
            filters = [
                "cm.embedding IS NOT NULL",
                "(CAST(%s AS text) IS NULL OR c.user_id = %s)",
            ]
            params: list[Any] = [user_id, user_id]
            if conversation_ids:
                filters.append("rt.conversation_id = ANY(%s)")
                params.append(list(conversation_ids))
            params.extend([list(query_embedding), max(1, limit)])
            cur.execute(
                f"""
                SELECT matched_message_id, roundtrip_id, conversation_id,
                       message_index, roundtrip_summary, created_at, relevance_score
                FROM (
                    SELECT
                        cm.id AS matched_message_id,
                        cm.roundtrip_id,
                        rt.conversation_id,
                        rt.message_index,
                        rt.roundtrip_summary,
                        rt.created_at,
                        (cm.embedding <-> (%s)::vector) AS relevance_score,
                        ROW_NUMBER() OVER (
                            PARTITION BY cm.roundtrip_id
                            ORDER BY cm.embedding <-> (%s)::vector ASC
                        ) AS message_rank
                    FROM conversation_message cm
                    JOIN conversation_roundtrip rt ON rt.id = cm.roundtrip_id
                    JOIN conversation c ON c.id = rt.conversation_id
                    WHERE {' AND '.join(filters)}
                ) ranked_messages
                WHERE message_rank = 1
                ORDER BY relevance_score ASC
                LIMIT %s
                """,
                [list(query_embedding), list(query_embedding), *params[:-2], params[-1]],
            )
            rows = cur.fetchall()

        results: list[RoundtripMemory] = []
        for row in rows:
            roundtrip = self._hydrate_roundtrip({
                "id": row["roundtrip_id"],
                "conversation_id": row["conversation_id"],
                "message_index": row["message_index"],
                "roundtrip_summary": row["roundtrip_summary"],
                "created_at": row["created_at"],
            })
            results.append(RoundtripMemory(
                conversation_id=roundtrip.conversation_id,
                roundtrip_id=roundtrip.id,
                message_index=roundtrip.message_index,
                roundtrip_summary=roundtrip.roundtrip_summary or "",
                created_at=str(roundtrip.created_at),
                relevance_score=float(row["relevance_score"]),
                messages=self.list_messages(roundtrip.id, include_internal=True),
                matched_message_id=row["matched_message_id"],
            ))
        return results

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                DELETE
                FROM conversation
                WHERE id = %s
                and user_id = %s
                """,
                (conversation_id, user_id),
            )
            return cur.rowcount > 0

    def get_latest_conversation(self, user_id: str) -> Conversation:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, user_id, title, created_at, metadata, tone_state, summary, summary_embedding
                FROM conversation
                WHERE user_id = %s
                order by updated_at DESC
                """,
                (user_id,),
            )
            row = cur.fetchone()
            return Conversation(**row) if row else None

    def update_tone_state(self, conversation_id: UUID, tone_state: TonePreferences) -> None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE conversation
                SET tone_state = %s, updated_at = now()
                WHERE id = %s;
                """,
                (Jsonb(tone_state.model_dump(mode="json", exclude_none=True)), conversation_id),
            )

    def set_conversation_title(self, conversation_id: str, title: str) -> bool:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE conversation
                SET title = %s
                WHERE id = %s;
                """,
                (title, conversation_id),
            )
            return cur.rowcount > 0

    def record_llm_usage(self, **kwargs) -> LlmCallRecord:
        metadata = dict(kwargs.get("metadata") or {})
        payload = {
            "kind": "llm_call",
            "model_scope": kwargs.get("agent") or "",
            "agent": kwargs.get("agent"),
            "stage": kwargs.get("stage"),
            "callsite": kwargs.get("callsite", ""),
            "model": kwargs.get("model", ""),
            "input_tokens": kwargs.get("input_tokens", 0),
            "output_tokens": kwargs.get("output_tokens", 0),
            "total_tokens": kwargs.get("total_tokens", 0),
            "cached_input_tokens": kwargs.get("cached_input_tokens", 0),
            "computed_input_cost": str(kwargs.get("computed_input_cost", 0)),
            "computed_output_cost": str(kwargs.get("computed_output_cost", 0)),
            "computed_total_cost": str(kwargs.get("computed_total_cost", 0)),
            "metadata": metadata,
        }
        roundtrip_id = kwargs.get("roundtrip_id")
        if roundtrip_id:
            self.append_node_log(
                "roundtrip",
                roundtrip_id,
            node_name=kwargs.get("node_name") or kwargs.get("stage") or kwargs.get("agent") or "llm",
                stage=kwargs.get("stage") or "",
                event_type="llm_call",
                message="LLM call completed",
                payload=payload,
                duration_ms=metadata.get("latency_ms"),
            )
        return LlmCallRecord(
            id=UUID(int=0), conversation_id=kwargs.get("conversation_id"), roundtrip_id=roundtrip_id,
            user_id=kwargs.get("user_id"), agent=kwargs.get("agent"), stage=kwargs.get("stage"),
            callsite=kwargs.get("callsite", ""), model=kwargs.get("model", ""),
            input_tokens=kwargs.get("input_tokens", 0), output_tokens=kwargs.get("output_tokens", 0),
            total_tokens=kwargs.get("total_tokens", 0), cached_input_tokens=kwargs.get("cached_input_tokens", 0),
            input_price_per_million_tokens=kwargs.get("input_price_per_million_tokens", Decimal(0)),
            output_price_per_million_tokens=kwargs.get("output_price_per_million_tokens", Decimal(0)),
            computed_input_cost=kwargs.get("computed_input_cost", Decimal(0)),
            computed_output_cost=kwargs.get("computed_output_cost", Decimal(0)),
            computed_total_cost=kwargs.get("computed_total_cost", Decimal(0)), metadata=metadata,
            created_at="", updated_at="",
        )
