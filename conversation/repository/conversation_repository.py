from __future__ import annotations

from typing import Any, Optional, Sequence
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from conversation.models.conversation_models import (
    Conversation,
    ConversationRoundtrip,
    ConversationSummary,
)


class ConversationRepository:
    def __init__(self, conn: psycopg.Connection):
        self._conn = conn


    def create_conversation(self, user_id: str, metadata: Optional[dict[str, Any]] = None) -> Conversation:
        metadata = metadata or {}
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO conversation (user_id, metadata, title, tone_state)
                VALUES (%s, %s, %s, %s)
                RETURNING id, user_id, title, created_at, metadata, tone_state
                """,
                (user_id, Jsonb(metadata), user_id, Jsonb({})),
            )
            row = cur.fetchone()
            assert row is not None
            return Conversation(**row)

    def append_roundtrip(
        self,
        conversation_id: UUID,
        user_prompt: str,
        generated_response: str,
        response_payload: Optional[dict[str, Any]] = None,
        parsed_query: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ConversationRoundtrip:
        metadata = metadata or {}
        response_payload = response_payload or {}
        parsed_query = parsed_query or {}
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO conversation_roundtrip (conversation_id, message_index, user_prompt, generated_response, response_payload, parsed_query, metadata)
                SELECT
                    %s,
                    COALESCE(MAX(message_index), -1) + 1,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                FROM conversation_roundtrip
                WHERE conversation_id = %s
                RETURNING id, conversation_id, message_index, user_prompt, generated_response, response_payload, parsed_query, created_at, metadata
                """,
                (conversation_id, user_prompt, generated_response, Jsonb(response_payload), Jsonb(parsed_query), Jsonb(metadata), conversation_id),
            )
            row = cur.fetchone()
            assert row is not None

            cur.execute(
                """
                UPDATE conversation
                SET updated_at = now()
                WHERE id = %s;
                """, (conversation_id,)
            )
            return ConversationRoundtrip(**row)

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
                """, (conversation_id,)
            )
            row = cur.fetchone()
            return ConversationSummary(**row) if row else None

    def get_conversation(self, conversation_id: UUID) -> Optional[Conversation]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, user_id, title, created_at, metadata, tone_state
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
    ) -> list[ConversationRoundtrip]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            if after_message_index is None:
                cur.execute(
                    """
                    SELECT id, conversation_id, message_index, user_prompt, generated_response, response_payload, parsed_query, created_at, metadata
                    FROM conversation_roundtrip
                    WHERE conversation_id = %s
                    ORDER BY message_index ASC
                    LIMIT %s
                    """,
                    (conversation_id, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT id, conversation_id, message_index, user_prompt, generated_response, response_payload, parsed_query, created_at, metadata
                    FROM conversation_roundtrip
                    WHERE conversation_id = %s
                      AND message_index > %s
                    ORDER BY message_index ASC
                    LIMIT %s
                    """,
                    (conversation_id, after_message_index, limit),
                )
            rows = cur.fetchall()
            return [ConversationRoundtrip(**r) for r in rows]

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

    # method to get all of our conversations (arbitrarily 50 for now)
    def list_conversations(self, user_id: str, limit: int = 50) -> list[Conversation]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, user_id, created_at, metadata, title, tone_state
                FROM conversation
                WHERE user_id = %s
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            rows = cur.fetchall()
            return [Conversation(**r) for r in rows]

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

    # get the latest conversation for a user in our case we just have a single one
    def get_latest_conversation(self, user_id: str) -> Conversation:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, user_id, title, created_at, metadata, tone_state
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

    # repo function to set conversation title for annonymous conversations
    def set_conversation_title(self, conversation_id: str, title: str) -> bool:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE conversation
                SET title = %s
                WHERE id = %s;
                """, (title, conversation_id,)
            )
            return cur.rowcount > 0
