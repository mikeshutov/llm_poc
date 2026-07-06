from __future__ import annotations

from typing import Optional, Sequence
from uuid import UUID

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from db.connection import get_connection
from memories.models.memory_models import Memory, MemorySearchResult

MEMORY_ORDER_FIELDS = {
    "created_at": "created_at",
    "updated_at": "updated_at",
    "confidence": "confidence",
    "importance": "importance",
}
MEMORY_DUPLICATE_DISTANCE_THRESHOLD = 0.12


class MemoryRepository:
    def __init__(self, conn: psycopg.Connection | None = None):
        self._conn = conn or get_connection()
        register_vector(self._conn)

    def _find_exact_memory(
        self,
        memory_text: str,
        *,
        user_id: Optional[str] = None,
        exclude_memory_id: Optional[UUID] = None,
    ) -> Optional[Memory]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    user_id,
                    memory_text,
                    memory_embedding,
                    source,
                    source_conversation_id,
                    source_roundtrip_id,
                    is_active,
                    created_at,
                    updated_at,
                    confidence,
                    importance
                FROM memories
                WHERE LOWER(BTRIM(memory_text)) = LOWER(BTRIM(%s))
                  AND (%s IS NULL OR user_id = %s)
                  AND (%s IS NULL OR id <> %s)
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (memory_text, user_id, user_id, exclude_memory_id, exclude_memory_id),
            )
            row = cur.fetchone()
            return Memory(**row) if row else None

    def _find_similar_memory(
        self,
        query_embedding: Sequence[float],
        *,
        user_id: Optional[str] = None,
        exclude_memory_id: Optional[UUID] = None,
        distance_threshold: float = MEMORY_DUPLICATE_DISTANCE_THRESHOLD,
    ) -> Optional[MemorySearchResult]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    user_id,
                    memory_text,
                    source,
                    source_conversation_id,
                    source_roundtrip_id,
                    is_active,
                    created_at,
                    updated_at,
                    confidence,
                    importance,
                    (memory_embedding <-> (%s)::vector) AS relevance_score
                FROM memories
                WHERE memory_embedding IS NOT NULL
                  AND BTRIM(memory_text) <> ''
                  AND (%s IS NULL OR user_id = %s)
                  AND (%s IS NULL OR id <> %s)
                ORDER BY memory_embedding <-> (%s)::vector ASC
                LIMIT 1
                """,
                (
                    list(query_embedding),
                    user_id,
                    user_id,
                    exclude_memory_id,
                    exclude_memory_id,
                    list(query_embedding),
                ),
            )
            row = cur.fetchone()
            if not row:
                return None
            result = MemorySearchResult(**row)
            return result if result.relevance_score <= distance_threshold else None

    def _update_memory_record(
        self,
        memory_id: UUID,
        *,
        memory_text: Optional[str] = None,
        memory_embedding: Optional[list[float]] = None,
        source: Optional[str] = None,
        source_conversation_id: Optional[UUID] = None,
        source_roundtrip_id: Optional[UUID] = None,
        is_active: Optional[bool] = None,
        confidence: Optional[float] = None,
        importance: Optional[float] = None,
    ) -> Optional[Memory]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE memories
                SET memory_text = COALESCE(%s, memory_text),
                    memory_embedding = COALESCE((%s)::vector, memory_embedding),
                    source = COALESCE(%s, source),
                    source_conversation_id = COALESCE(%s, source_conversation_id),
                    source_roundtrip_id = COALESCE(%s, source_roundtrip_id),
                    is_active = COALESCE(%s, is_active),
                    confidence = COALESCE(%s, confidence),
                    importance = COALESCE(%s, importance),
                    updated_at = now()
                WHERE id = %s
                RETURNING
                    id,
                    user_id,
                    memory_text,
                    memory_embedding,
                    source,
                    source_conversation_id,
                    source_roundtrip_id,
                    is_active,
                    created_at,
                    updated_at,
                    confidence,
                    importance
                """,
                (
                    memory_text,
                    memory_embedding,
                    source,
                    source_conversation_id,
                    source_roundtrip_id,
                    is_active,
                    confidence,
                    importance,
                    memory_id,
                ),
            )
            row = cur.fetchone()
            return Memory(**row) if row else None

    def _deactivate_memory(self, memory_id: UUID) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE memories
                SET is_active = false,
                    updated_at = now()
                WHERE id = %s
                """,
                (memory_id,),
            )

    def create_memory(
        self,
        memory_text: str,
        user_id: Optional[str] = None,
        memory_embedding: Optional[list[float]] = None,
        source: Optional[str] = None,
        source_conversation_id: Optional[UUID] = None,
        source_roundtrip_id: Optional[UUID] = None,
        is_active: bool = True,
        confidence: Optional[float] = None,
        importance: Optional[float] = None,
    ) -> Memory:
        exact_match = self._find_exact_memory(memory_text, user_id=user_id)
        if exact_match is not None:
            updated_memory = self._update_memory_record(
                exact_match.id,
                memory_text=memory_text,
                memory_embedding=memory_embedding,
                source=source,
                source_conversation_id=source_conversation_id,
                source_roundtrip_id=source_roundtrip_id,
                is_active=is_active,
                confidence=confidence,
                importance=importance,
            )
            assert updated_memory is not None
            return updated_memory

        if memory_embedding is not None:
            similar_match = self._find_similar_memory(
                memory_embedding,
                user_id=user_id,
            )
            if similar_match is not None:
                updated_memory = self._update_memory_record(
                    similar_match.id,
                    memory_text=memory_text,
                    memory_embedding=memory_embedding,
                    source=source,
                    source_conversation_id=source_conversation_id,
                    source_roundtrip_id=source_roundtrip_id,
                    is_active=is_active,
                    confidence=confidence,
                    importance=importance,
                )
                assert updated_memory is not None
                return updated_memory

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO memories (
                    user_id,
                    memory_text,
                    memory_embedding,
                    source,
                    source_conversation_id,
                    source_roundtrip_id,
                    is_active,
                    confidence,
                    importance
                )
                VALUES (%s, %s, (%s)::vector, %s, %s, %s, %s, %s, %s)
                RETURNING
                    id,
                    user_id,
                    memory_text,
                    memory_embedding,
                    source,
                    source_conversation_id,
                    source_roundtrip_id,
                    is_active,
                    created_at,
                    updated_at,
                    confidence,
                    importance
                """,
                (
                    user_id,
                    memory_text,
                    memory_embedding,
                    source,
                    source_conversation_id,
                    source_roundtrip_id,
                    is_active,
                    confidence,
                    importance,
                ),
            )
            row = cur.fetchone()
            assert row is not None
            return Memory(**row)

    def update_memory(
        self,
        memory_id: UUID,
        memory_text: Optional[str] = None,
        memory_embedding: Optional[list[float]] = None,
        source: Optional[str] = None,
        source_conversation_id: Optional[UUID] = None,
        source_roundtrip_id: Optional[UUID] = None,
        is_active: Optional[bool] = None,
        confidence: Optional[float] = None,
        importance: Optional[float] = None,
    ) -> Optional[Memory]:
        if memory_text is not None:
            exact_match = self._find_exact_memory(
                memory_text,
                exclude_memory_id=memory_id,
            )
            if exact_match is not None:
                updated_memory = self._update_memory_record(
                    exact_match.id,
                    memory_text=memory_text,
                    memory_embedding=memory_embedding,
                    source=source,
                    source_conversation_id=source_conversation_id,
                    source_roundtrip_id=source_roundtrip_id,
                    is_active=is_active if is_active is not None else True,
                    confidence=confidence,
                    importance=importance,
                )
                self._deactivate_memory(memory_id)
                return updated_memory

            if memory_embedding is not None:
                similar_match = self._find_similar_memory(
                    memory_embedding,
                    exclude_memory_id=memory_id,
                )
                if similar_match is not None:
                    updated_memory = self._update_memory_record(
                        similar_match.id,
                        memory_text=memory_text,
                        memory_embedding=memory_embedding,
                        source=source,
                        source_conversation_id=source_conversation_id,
                        source_roundtrip_id=source_roundtrip_id,
                        is_active=is_active if is_active is not None else True,
                        confidence=confidence,
                        importance=importance,
                    )
                    self._deactivate_memory(memory_id)
                    return updated_memory

        return self._update_memory_record(
            memory_id,
            memory_text=memory_text,
            memory_embedding=memory_embedding,
            source=source,
            source_conversation_id=source_conversation_id,
            source_roundtrip_id=source_roundtrip_id,
            is_active=is_active,
            confidence=confidence,
            importance=importance,
        )

    def get_memory(self, memory_id: UUID) -> Optional[Memory]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    user_id,
                    memory_text,
                    memory_embedding,
                    source,
                    source_conversation_id,
                    source_roundtrip_id,
                    is_active,
                    created_at,
                    updated_at,
                    confidence,
                    importance
                FROM memories
                WHERE id = %s
                """,
                (memory_id,),
            )
            row = cur.fetchone()
            return Memory(**row) if row else None

    def list_memories(
        self,
        *,
        limit: int = 50,
        order_by: str = "updated_at",
        descending: bool = True,
        user_id: Optional[str] = None,
        is_active: Optional[bool] = None,
        source: Optional[str] = None,
        source_conversation_id: Optional[UUID] = None,
        source_roundtrip_id: Optional[UUID] = None,
    ) -> list[Memory]:
        order_field = MEMORY_ORDER_FIELDS.get(order_by)
        if order_field is None:
            raise ValueError(f"Unsupported memory order field: {order_by}")

        order_direction = "DESC" if descending else "ASC"
        query = f"""
            SELECT
                id,
                user_id,
                memory_text,
                memory_embedding,
                source,
                source_conversation_id,
                source_roundtrip_id,
                is_active,
                created_at,
                updated_at,
                confidence,
                importance
            FROM memories
            WHERE (%s IS NULL OR user_id = %s)
              AND (%s IS NULL OR is_active = %s)
              AND (%s IS NULL OR source = %s)
              AND (%s IS NULL OR source_conversation_id = %s)
              AND (%s IS NULL OR source_roundtrip_id = %s)
            ORDER BY {order_field} {order_direction}
            LIMIT %s
        """

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                query,
                (
                    user_id,
                    user_id,
                    is_active,
                    is_active,
                    source,
                    source,
                    source_conversation_id,
                    source_conversation_id,
                    source_roundtrip_id,
                    source_roundtrip_id,
                    limit,
                ),
            )
            rows = cur.fetchall()
            return [Memory(**row) for row in rows]

    def search_memories(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int = 5,
        user_id: Optional[str] = None,
        is_active: Optional[bool] = True,
        source: Optional[str] = None,
        source_conversation_id: Optional[UUID] = None,
        source_roundtrip_id: Optional[UUID] = None,
    ) -> list[MemorySearchResult]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    user_id,
                    memory_text,
                    source,
                    source_conversation_id,
                    source_roundtrip_id,
                    is_active,
                    created_at,
                    updated_at,
                    confidence,
                    importance,
                    (memory_embedding <-> (%s)::vector) AS relevance_score
                FROM memories
                WHERE memory_embedding IS NOT NULL
                  AND BTRIM(memory_text) <> ''
                  AND (%s IS NULL OR user_id = %s)
                  AND (%s IS NULL OR is_active = %s)
                  AND (%s IS NULL OR source = %s)
                  AND (%s IS NULL OR source_conversation_id = %s)
                  AND (%s IS NULL OR source_roundtrip_id = %s)
                ORDER BY memory_embedding <-> (%s)::vector ASC
                LIMIT %s
                """,
                (
                    list(query_embedding),
                    user_id,
                    user_id,
                    is_active,
                    is_active,
                    source,
                    source,
                    source_conversation_id,
                    source_conversation_id,
                    source_roundtrip_id,
                    source_roundtrip_id,
                    list(query_embedding),
                    limit,
                ),
            )
            rows = cur.fetchall()
            return [MemorySearchResult(**row) for row in rows]
