from __future__ import annotations

from typing import Sequence
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from db.connection import get_connection
from retrieval.evidence import EvidenceUrl, EvidenceView


class EvidenceRepository:
    def __init__(self, conn: psycopg.Connection | None = None):
        self._conn = conn or get_connection()

    def get_by_ids(
        self,
        evidence_ids: Sequence[str],
        *,
        user_id: str | None = None,
    ) -> dict[str, EvidenceView]:
        parsed_ids: list[UUID] = []
        seen_ids: set[UUID] = set()
        for evidence_id in evidence_ids:
            if not isinstance(evidence_id, str) or not evidence_id.strip():
                continue
            try:
                parsed_id = UUID(evidence_id.strip())
            except ValueError:
                continue
            if parsed_id in seen_ids:
                continue
            seen_ids.add(parsed_id)
            parsed_ids.append(parsed_id)
        if not parsed_ids:
            return {}

        evidence_by_id: dict[str, EvidenceView] = {}
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT cm.payload
                FROM conversation_message cm
                JOIN conversation_roundtrip rt ON rt.id = cm.roundtrip_id
                JOIN conversation c ON c.id = rt.conversation_id
                WHERE cm.kind = 'tool_result'
                  AND (CAST(%s AS text) IS NULL OR c.user_id = %s)
                """,
                (user_id, user_id),
            )
            rows = cur.fetchall()
        for row in rows:
            payload = row.get("payload") if isinstance(row, dict) else None
            for raw in payload.get("evidence", []) if isinstance(payload, dict) else []:
                    try:
                        evidence = EvidenceView.model_validate(raw)
                    except Exception:
                        continue
                    if evidence.id in parsed_ids and str(evidence.id) not in evidence_by_id:
                        evidence_by_id[str(evidence.id)] = evidence
        return evidence_by_id
