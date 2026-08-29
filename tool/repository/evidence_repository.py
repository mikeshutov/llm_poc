from __future__ import annotations

from typing import Sequence
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from db.connection import get_connection
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceView


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

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT evidence_views.*
                FROM evidence_views
                WHERE evidence_views.id = ANY(%s)
                  AND (CAST(%s AS text) IS NULL OR evidence_views.user_id = %s)
                """,
                (parsed_ids, user_id, user_id),
            )
            rows = cur.fetchall()

        evidence_by_id: dict[str, EvidenceView] = {}
        for row in rows:
            evidence_id = row.get("id")
            if not isinstance(evidence_id, UUID):
                continue
            normalized_id = str(evidence_id)
            if normalized_id in evidence_by_id:
                continue
            evidence_by_id[normalized_id] = EvidenceView(
                id=evidence_id,
                tool_call_id=row["tool_call_id"],
                item_id=row["item_id"],
                tool_name=row["tool_name"],
                title=row["title"],
                summary=row["summary"],
                urls=[EvidenceUrl.model_validate(url) for url in row["urls"]],
                image_url=row["image_url"],
                published_at=row["published_at"],
                source=row["source"],
                entity_type=row["entity_type"],
                location_name=row["location_name"],
                hash=row["hash"],
                llm_metadata=row["llm_metadata"],
                raw_payload=row["raw_payload"],
            )
        return evidence_by_id
