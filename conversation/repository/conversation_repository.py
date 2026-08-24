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
    ConversationEvent,
    ConversationMemory,
    ConversationRoundtrip,
    LlmCallRecord,
    ConversationSummary,
    RoundtripFeedback,
    RoundtripMemory,
    RoundtripPrompt,
)
from db.connection import get_connection
from llm.repository.conversation_model_config_repository import ConversationModelConfigRepository


class ConversationRepository:
    def __init__(self, conn: psycopg.Connection | None = None):
        self._conn = conn or get_connection()
        register_vector(self._conn)

    def create_conversation(self, user_id: str, metadata: Optional[dict[str, Any]] = None) -> Conversation:
        metadata = metadata or {}
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO conversation (user_id, metadata, title, tone_state)
                VALUES (%s, %s, %s, %s)
                RETURNING id, user_id, title, created_at, metadata, tone_state, summary
                """,
                (user_id, Jsonb(metadata), "Unnamed", Jsonb({})),
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
        metadata: Optional[dict[str, Any]] = None,
    ) -> ConversationRoundtrip:
        metadata = metadata or {}
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO conversation_roundtrip (conversation_id, message_index, user_prompt, generated_response, roundtrip_summary, roundtrip_summary_embedding, response_payload, parsed_query, model, metadata)
                SELECT %s, COALESCE(MAX(message_index), -1) + 1, %s, '', %s, (%s)::vector, '{}'::jsonb, '{}'::jsonb, %s, %s
                FROM conversation_roundtrip
                WHERE conversation_id = %s
                RETURNING id, conversation_id, message_index, user_prompt, generated_response, roundtrip_summary, roundtrip_summary_embedding, response_payload, parsed_query, created_at, metadata, model, assistant_follow_up
                """,
                (conversation_id, user_prompt, roundtrip_summary, roundtrip_summary_embedding, model, Jsonb(metadata), conversation_id),
            )
            row = cur.fetchone()
            assert row is not None
            return ConversationRoundtrip(**row)

    def update_roundtrip(
        self,
        roundtrip_id: UUID,
        response: str,
        payload: dict[str, Any],
        roundtrip_summary: Optional[str] = None,
        roundtrip_summary_embedding: Optional[list[float]] = None,
        assistant_follow_up: str | None = None,
    ) -> ConversationRoundtrip:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE conversation_roundtrip
                SET generated_response = %s,
                    roundtrip_summary = COALESCE(%s, roundtrip_summary),
                    roundtrip_summary_embedding = COALESCE((%s)::vector, roundtrip_summary_embedding),
                    assistant_follow_up = COALESCE(%s, assistant_follow_up),
                    response_payload = %s,
                    updated_at = now()
                WHERE id = %s
                RETURNING id, conversation_id, message_index, user_prompt, generated_response, roundtrip_summary, roundtrip_summary_embedding, response_payload, parsed_query, created_at, metadata, model, assistant_follow_up
                """,
                (response, roundtrip_summary, roundtrip_summary_embedding, assistant_follow_up, Jsonb(payload), roundtrip_id),
            )
            row = cur.fetchone()
            assert row is not None
            return ConversationRoundtrip(**row)

    def append_roundtrip(
        self,
        conversation_id: UUID,
        user_prompt: str,
        generated_response: str,
        response_payload: Optional[dict[str, Any]] = None,
        parsed_query: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        model: Optional[str] = None,
        roundtrip_summary: Optional[str] = None,
        roundtrip_summary_embedding: Optional[list[float]] = None,
    ) -> ConversationRoundtrip:
        metadata = metadata or {}
        response_payload = response_payload or {}
        parsed_query = parsed_query or {}
        assistant_follow_up = str(response_payload.get("next_question") or "").strip()
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO conversation_roundtrip (conversation_id, message_index, user_prompt, generated_response, roundtrip_summary, roundtrip_summary_embedding, assistant_follow_up, response_payload, parsed_query, model, metadata)
                SELECT
                    %s,
                    COALESCE(MAX(message_index), -1) + 1,
                    %s,
                    %s,
                    %s,
                    (%s)::vector,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                FROM conversation_roundtrip
                WHERE conversation_id = %s
                RETURNING id, conversation_id, message_index, user_prompt, generated_response, roundtrip_summary, roundtrip_summary_embedding, response_payload, parsed_query, created_at, metadata, model, assistant_follow_up
                """,
                (conversation_id, user_prompt, generated_response, roundtrip_summary, roundtrip_summary_embedding, assistant_follow_up, Jsonb(response_payload), Jsonb(parsed_query), model, Jsonb(metadata), conversation_id),
            )
            row = cur.fetchone()
            assert row is not None

            cur.execute(
                """
                UPDATE conversation
                SET updated_at = now()
                WHERE id = %s;
                """,
                (conversation_id,),
            )
            return ConversationRoundtrip(**row)

    def create_roundtrip_prompt(
        self,
        roundtrip_id: UUID,
        agent: str,
        prompt_step: str,
        prompt: str,
    ) -> RoundtripPrompt:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO roundtrip_prompts (roundtrip_id, agent, prompt_step, prompt)
                VALUES (%s, %s, %s, %s)
                RETURNING id, roundtrip_id, agent, prompt_step, prompt, created_at
                """,
                (roundtrip_id, agent, prompt_step, prompt),
            )
            row = cur.fetchone()
            assert row is not None
            return RoundtripPrompt(**row)

    def create_conversation_event(
        self,
        *,
        conversation_id: UUID,
        roundtrip_id: UUID | None,
        event_type: str,
        source: str,
        agent_name: str = "",
        node_name: str = "",
        iteration: int | None = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> ConversationEvent:
        payload = payload or {}
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO conversation_event (
                    conversation_id,
                    roundtrip_id,
                    event_type,
                    source,
                    agent_name,
                    node_name,
                    iteration,
                    payload
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING
                    id,
                    conversation_id,
                    roundtrip_id,
                    event_type,
                    source,
                    agent_name,
                    node_name,
                    iteration,
                    payload,
                    created_at
                """,
                (
                    conversation_id,
                    roundtrip_id,
                    event_type,
                    source,
                    agent_name,
                    node_name,
                    iteration,
                    Jsonb(payload),
                ),
            )
            row = cur.fetchone()
            assert row is not None
            return ConversationEvent(**row)

    def list_conversation_events_for_roundtrip(self, roundtrip_id: UUID) -> list[ConversationEvent]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    conversation_id,
                    roundtrip_id,
                    event_type,
                    source,
                    agent_name,
                    node_name,
                    iteration,
                    payload,
                    created_at
                FROM conversation_event
                WHERE roundtrip_id = %s
                ORDER BY id ASC
                """,
                (roundtrip_id,),
            )
            rows = cur.fetchall()
            return [ConversationEvent(**row) for row in rows]

    def create_llm_call(
        self,
        *,
        conversation_id: UUID | None,
        roundtrip_id: UUID | None,
        user_id: str | None,
        agent: str | None,
        stage: str | None,
        callsite: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        cached_input_tokens: int,
        input_price_per_million_tokens: Decimal,
        output_price_per_million_tokens: Decimal,
        computed_input_cost: Decimal,
        computed_output_cost: Decimal,
        computed_total_cost: Decimal,
        metadata: Optional[dict[str, Any]] = None,
    ) -> LlmCallRecord:
        metadata = metadata or {}
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO llm_call (
                    conversation_id,
                    roundtrip_id,
                    user_id,
                    agent,
                    stage,
                    callsite,
                    model,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cached_input_tokens,
                    input_price_per_million_tokens,
                    output_price_per_million_tokens,
                    computed_input_cost,
                    computed_output_cost,
                    computed_total_cost,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING
                    id,
                    conversation_id,
                    roundtrip_id,
                    user_id,
                    agent,
                    stage,
                    callsite,
                    model,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cached_input_tokens,
                    input_price_per_million_tokens,
                    output_price_per_million_tokens,
                    computed_input_cost,
                    computed_output_cost,
                    computed_total_cost,
                    metadata,
                    created_at,
                    updated_at
                """,
                (
                    conversation_id,
                    roundtrip_id,
                    user_id,
                    agent,
                    stage,
                    callsite,
                    model,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cached_input_tokens,
                    input_price_per_million_tokens,
                    output_price_per_million_tokens,
                    computed_input_cost,
                    computed_output_cost,
                    computed_total_cost,
                    Jsonb(metadata),
                ),
            )
            row = cur.fetchone()
            assert row is not None
            return LlmCallRecord(
                id=row['id'],
                conversation_id=row['conversation_id'],
                roundtrip_id=row['roundtrip_id'],
                user_id=row.get('user_id'),
                agent=row['agent'],
                stage=row['stage'],
                callsite=row['callsite'],
                model=row['model'],
                input_tokens=row['input_tokens'],
                output_tokens=row['output_tokens'],
                total_tokens=row['total_tokens'],
                cached_input_tokens=row.get('cached_input_tokens', 0),
                input_price_per_million_tokens=row['input_price_per_million_tokens'],
                output_price_per_million_tokens=row['output_price_per_million_tokens'],
                computed_input_cost=row['computed_input_cost'],
                computed_output_cost=row['computed_output_cost'],
                computed_total_cost=row['computed_total_cost'],
                metadata=row['metadata'],
                created_at=str(row['created_at']),
                updated_at=str(row['updated_at']),
            )

    def list_llm_calls_for_roundtrip(self, roundtrip_id: UUID) -> list[LlmCallRecord]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    conversation_id,
                    roundtrip_id,
                    user_id,
                    agent,
                    stage,
                    callsite,
                    model,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cached_input_tokens,
                    input_price_per_million_tokens,
                    output_price_per_million_tokens,
                    computed_input_cost,
                    computed_output_cost,
                    computed_total_cost,
                    metadata,
                    created_at,
                    updated_at
                FROM llm_call
                WHERE roundtrip_id = %s
                ORDER BY created_at ASC
                """,
                (roundtrip_id,),
            )
            rows = cur.fetchall()
            return [
                LlmCallRecord(
                    id=row['id'],
                    conversation_id=row['conversation_id'],
                    roundtrip_id=row['roundtrip_id'],
                    user_id=row.get('user_id'),
                    agent=row['agent'],
                    stage=row['stage'],
                    callsite=row['callsite'],
                    model=row['model'],
                    input_tokens=row['input_tokens'],
                    output_tokens=row['output_tokens'],
                    total_tokens=row['total_tokens'],
                    cached_input_tokens=row.get('cached_input_tokens', 0),
                    input_price_per_million_tokens=row['input_price_per_million_tokens'],
                    output_price_per_million_tokens=row['output_price_per_million_tokens'],
                    computed_input_cost=row['computed_input_cost'],
                    computed_output_cost=row['computed_output_cost'],
                    computed_total_cost=row['computed_total_cost'],
                    metadata=row['metadata'],
                    created_at=str(row['created_at']),
                    updated_at=str(row['updated_at']),
                )
                for row in rows
            ]

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
        tool_summary: str = "",
    ) -> ConversationSummary:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO conversation_summary (conversation_id, summary, tool_summary, message_index_cutoff)
                VALUES (%s, %s, %s, %s)
                RETURNING id, conversation_id, summary, tool_summary, message_index_cutoff, created_at
                """,
                (conversation_id, summary, tool_summary, message_index_cutoff),
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
                SELECT id, conversation_id, summary, tool_summary, message_index_cutoff, created_at
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
                    rt.user_prompt,
                    rt.generated_response,
                    rt.roundtrip_summary,
                    rt.roundtrip_summary_embedding,
                    rt.assistant_follow_up,
                    rt.response_payload,
                    rt.parsed_query,
                    rt.created_at,
                    rt.metadata,
                    rt.model,
                    fb.id AS feedback_id
                FROM conversation_roundtrip rt
                LEFT JOIN roundtrip_feedback fb ON fb.roundtrip_id = rt.id
                WHERE rt.id = %s
                """,
                (roundtrip_id,),
            )
            row = cur.fetchone()
            return ConversationRoundtrip(**row) if row else None

    def get_roundtrip_for_user(self, roundtrip_id: UUID, user_id: str | None) -> Optional[ConversationRoundtrip]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    rt.id,
                    rt.conversation_id,
                    rt.message_index,
                    rt.user_prompt,
                    rt.generated_response,
                    rt.roundtrip_summary,
                    rt.roundtrip_summary_embedding,
                    rt.assistant_follow_up,
                    rt.response_payload,
                    rt.parsed_query,
                    rt.created_at,
                    rt.metadata,
                    rt.model,
                    fb.id AS feedback_id
                FROM conversation_roundtrip rt
                JOIN conversation c ON c.id = rt.conversation_id
                LEFT JOIN roundtrip_feedback fb ON fb.roundtrip_id = rt.id
                WHERE rt.id = %s
                  AND (CAST(%s AS text) IS NULL OR c.user_id = %s)
                """,
                (roundtrip_id, user_id, user_id),
            )
            row = cur.fetchone()
            return ConversationRoundtrip(**row) if row else None

    def get_latest_completed_roundtrip(self, conversation_id: UUID) -> Optional[ConversationRoundtrip]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    conversation_id,
                    message_index,
                    user_prompt,
                    generated_response,
                    roundtrip_summary,
                    roundtrip_summary_embedding,
                    assistant_follow_up,
                    response_payload,
                    parsed_query,
                    created_at,
                    metadata,
                    model
                FROM conversation_roundtrip
                WHERE conversation_id = %s
                  AND BTRIM(COALESCE(generated_response, '')) <> ''
                ORDER BY message_index DESC
                LIMIT 1
                """,
                (conversation_id,),
            )
            row = cur.fetchone()
            return ConversationRoundtrip(**row) if row else None

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
                        rt.user_prompt,
                        rt.generated_response,
                        rt.roundtrip_summary,
                        rt.roundtrip_summary_embedding,
                        rt.assistant_follow_up,
                        rt.response_payload,
                        rt.parsed_query,
                        rt.created_at,
                        rt.metadata,
                        rt.model,
                        fb.id AS feedback_id
                    FROM conversation_roundtrip rt
                    LEFT JOIN roundtrip_feedback fb ON fb.roundtrip_id = rt.id
                    WHERE rt.conversation_id = %s
                      AND BTRIM(COALESCE(rt.generated_response, '')) <> ''
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
                        rt.user_prompt,
                        rt.generated_response,
                        rt.roundtrip_summary,
                        rt.roundtrip_summary_embedding,
                        rt.assistant_follow_up,
                        rt.response_payload,
                        rt.parsed_query,
                        rt.created_at,
                        rt.metadata,
                        rt.model,
                        fb.id AS feedback_id
                    FROM conversation_roundtrip rt
                    LEFT JOIN roundtrip_feedback fb ON fb.roundtrip_id = rt.id
                    WHERE rt.conversation_id = %s
                      AND rt.message_index > %s
                      AND BTRIM(COALESCE(rt.generated_response, '')) <> ''
                    ORDER BY rt.message_index {order_direction}
                    LIMIT %s
                    """,
                    (conversation_id, after_message_index, limit),
                )
            rows = cur.fetchall()
            roundtrips = [ConversationRoundtrip(**r) for r in rows]
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
                    rt.user_prompt,
                    rt.generated_response,
                    rt.roundtrip_summary,
                    rt.roundtrip_summary_embedding,
                    rt.assistant_follow_up,
                    rt.response_payload,
                    rt.parsed_query,
                    rt.created_at,
                    rt.metadata,
                    rt.model,
                    fb.id AS feedback_id
                FROM conversation_roundtrip rt
                LEFT JOIN roundtrip_feedback fb ON fb.roundtrip_id = rt.id
                WHERE rt.conversation_id = %s
                  AND rt.message_index <= %s
                ORDER BY rt.message_index ASC
                """,
                (conversation_id, message_index),
            )
            rows = cur.fetchall()
            return [ConversationRoundtrip(**r) for r in rows]

    def get_latest_summary(self, conversation_id: UUID) -> Optional[ConversationSummary]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, conversation_id, summary, tool_summary, message_index_cutoff, created_at
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
                    user_prompt,
                    generated_response,
                    roundtrip_summary,
                    roundtrip_summary_embedding,
                    assistant_follow_up,
                    response_payload,
                    parsed_query,
                    created_at,
                    updated_at,
                    model,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, (%s)::vector, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, conversation_id, message_index, user_prompt, generated_response, roundtrip_summary, roundtrip_summary_embedding, response_payload, parsed_query, created_at, metadata, model, assistant_follow_up
                """,
                (
                    conversation_id,
                    source_roundtrip.message_index,
                    source_roundtrip.user_prompt,
                    source_roundtrip.generated_response,
                    source_roundtrip.roundtrip_summary,
                    source_roundtrip.roundtrip_summary_embedding,
                    source_roundtrip.assistant_follow_up,
                    Jsonb(source_roundtrip.response_payload or {}),
                    Jsonb(source_roundtrip.parsed_query or {}),
                    source_roundtrip.created_at,
                    source_roundtrip.created_at,
                    source_roundtrip.model,
                    Jsonb(source_roundtrip.metadata or {}),
                ),
            )
            row = cur.fetchone()
            assert row is not None
            return ConversationRoundtrip(**row)

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
        conversation_ids: Sequence[UUID],
        limit: int = 5,
        user_id: str | None = None,
    ) -> list[RoundtripMemory]:
        if not conversation_ids:
            return []

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    rt.conversation_id,
                    rt.id AS roundtrip_id,
                    rt.message_index,
                    rt.user_prompt,
                    rt.generated_response,
                    rt.roundtrip_summary,
                    rt.created_at,
                    (rt.roundtrip_summary_embedding <-> (%s)::vector) AS relevance_score
                FROM conversation_roundtrip rt
                JOIN conversation c ON c.id = rt.conversation_id
                WHERE rt.conversation_id = ANY(%s)
                  AND (CAST(%s AS text) IS NULL OR c.user_id = %s)
                  AND rt.roundtrip_summary_embedding IS NOT NULL
                  AND BTRIM(COALESCE(rt.roundtrip_summary, '')) <> ''
                ORDER BY rt.roundtrip_summary_embedding <-> (%s)::vector ASC
                LIMIT %s
                """,
                (list(query_embedding), list(conversation_ids), user_id, user_id, list(query_embedding), limit),
            )
            rows = cur.fetchall()
            return [
                RoundtripMemory(
                    conversation_id=row['conversation_id'],
                    roundtrip_id=row['roundtrip_id'],
                    message_index=row['message_index'],
                    user_prompt=row['user_prompt'],
                    generated_response=row['generated_response'],
                    roundtrip_summary=row['roundtrip_summary'],
                    created_at=str(row['created_at']),
                    relevance_score=float(row['relevance_score']),
                )
                for row in rows
            ]

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

    def update_tone_state(self, conversation_id: UUID, tone_state: dict[str, Any]) -> None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE conversation
                SET tone_state = %s, updated_at = now()
                WHERE id = %s;
                """,
                (Jsonb(tone_state), conversation_id),
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
