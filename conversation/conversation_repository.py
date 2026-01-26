from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


@dataclass(frozen=False)
class Conversation:
    id: UUID
    user_id: str
    title: Optional[str]
    created_at: str  # or datetime
    metadata: dict[str, Any]


@dataclass(frozen=False)
class ConversationRoundtrip:
    id: UUID
    conversation_id: UUID
    message_index: int
    user_prompt: str
    generated_response: str
    created_at: str  # or datetime
    metadata: dict[str, Any]


@dataclass(frozen=False)
class ConversationSummary:
    id: UUID
    conversation_id: UUID
    summary: str
    created_at: str  # or datetime
    metadata: dict[str, Any]


class ConversationRepository:
    def __init__(self, conn: psycopg.Connection):
        self._conn = conn


    def create_conversation(self, user_id: str, metadata: Optional[dict[str, Any]] = None) -> Conversation:
        metadata = metadata or {}
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO conversation (user_id, metadata, title)
                VALUES (%s, %s, %s)
                RETURNING id, user_id, title, created_at, metadata
                """,
                (user_id, Jsonb(metadata), user_id),
            )
            row = cur.fetchone()
            assert row is not None
            return Conversation(**row)

    def append_roundtrip(
        self,
        conversation_id: UUID,
        user_prompt: str,
        generated_response: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ConversationRoundtrip:
        metadata = metadata or {}
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO conversation_roundtrip (conversation_id, message_index, user_prompt, generated_response, metadata)
                SELECT
                    %s,
                    COALESCE(MAX(message_index), -1) + 1,
                    %s,
                    %s,
                    %s
                FROM conversation_roundtrip
                WHERE conversation_id = %s
                RETURNING id, conversation_id, message_index, user_prompt, generated_response, created_at, metadata
                """,
                (conversation_id, user_prompt, generated_response, Jsonb(metadata), conversation_id),
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
        metadata: Optional[dict[str, Any]] = None,
    ) -> ConversationSummary:
        metadata = metadata or {}
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO conversation_summary (conversation_id, summary, metadata)
                VALUES (%s, %s, %s::jsonb)
                RETURNING id, conversation_id, summary, created_at, metadata
                """,
                (conversation_id, summary, metadata),
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
                SELECT conversation_id, summary, created_at
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
                SELECT id, user_id, title, created_at, metadata
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
                    SELECT id, conversation_id, message_index, user_prompt, generated_response, created_at, metadata
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
                    SELECT id, conversation_id, message_index, user_prompt, generated_response, created_at, metadata
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
                SELECT id, conversation_id, summary, created_at, metadata
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
                SELECT id, user_id, created_at, metadata, title
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
                SELECT id, user_id, title, created_at, metadata
                FROM conversation
                WHERE user_id = %s
                order by updated_at DESC
                """,
                (user_id,),
            )
            row = cur.fetchone()
            return Conversation(**row) if row else None

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