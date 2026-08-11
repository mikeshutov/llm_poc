from __future__ import annotations

from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from db.connection import get_connection
from personalization.tone.models import TonePreferences, ToneRecord, ToneType


class ToneRepository:
    def __init__(self, conn: psycopg.Connection | None = None):
        self._conn = conn or get_connection()

    def _normalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        for field_name in ("created_at", "updated_at"):
            field_value = normalized.get(field_name)
            if field_value is not None and hasattr(field_value, "isoformat"):
                normalized[field_name] = field_value.isoformat()
        return normalized

    def get_tone(
        self,
        *,
        user_id: str,
        tone_type: ToneType = "profile",
        conversation_id: UUID | None = None,
    ) -> ToneRecord | None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    user_id,
                    tone_type,
                    conversation_id,
                    verbosity,
                    formality,
                    directness,
                    humor,
                    technical_depth,
                    created_at,
                    updated_at
                FROM tone
                WHERE user_id = %s
                  AND tone_type = %s
                  AND conversation_id IS NOT DISTINCT FROM %s
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (user_id, tone_type, conversation_id),
            )
            row = cur.fetchone()
            return ToneRecord(**self._normalize_row(row)) if row else None

    def upsert_tone(
        self,
        *,
        user_id: str,
        tone: TonePreferences,
        tone_type: ToneType = "profile",
        conversation_id: UUID | None = None,
    ) -> ToneRecord:
        existing = self.get_tone(
            user_id=user_id,
            tone_type=tone_type,
            conversation_id=conversation_id,
        )

        with self._conn.cursor(row_factory=dict_row) as cur:
            if existing is None:
                cur.execute(
                    """
                    INSERT INTO tone (
                        user_id,
                        tone_type,
                        conversation_id,
                        verbosity,
                        formality,
                        directness,
                        humor,
                        technical_depth
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING
                        id,
                        user_id,
                        tone_type,
                        conversation_id,
                        verbosity,
                        formality,
                        directness,
                        humor,
                        technical_depth,
                        created_at,
                        updated_at
                    """,
                    (
                        user_id,
                        tone_type,
                        conversation_id,
                        tone.verbosity,
                        tone.formality,
                        tone.directness,
                        tone.humor,
                        tone.technical_depth,
                    ),
                )
            else:
                cur.execute(
                    """
                    UPDATE tone
                    SET verbosity = %s,
                        formality = %s,
                        directness = %s,
                        humor = %s,
                        technical_depth = %s,
                        updated_at = now()
                    WHERE id = %s
                    RETURNING
                        id,
                        user_id,
                        tone_type,
                        conversation_id,
                        verbosity,
                        formality,
                        directness,
                        humor,
                        technical_depth,
                        created_at,
                        updated_at
                    """,
                    (
                        tone.verbosity,
                        tone.formality,
                        tone.directness,
                        tone.humor,
                        tone.technical_depth,
                        existing.id,
                    ),
                )
            row = cur.fetchone()
            assert row is not None
            return ToneRecord(**self._normalize_row(row))
