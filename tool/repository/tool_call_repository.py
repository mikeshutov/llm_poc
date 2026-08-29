from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any
from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel

from db.connection import get_connection
from request_orchestrator.models.plan import Plan
from request_orchestrator.models.plan import PlanStep
from request_orchestrator.models.evidence import EvidenceUrl, EvidenceView, ToolResult
from common.signatures import build_signature
from tool.repository.models import ToolCall


class ToolCallRepository:
    def __init__(self):
        self._conn = get_connection()

    def append_tool_call(
        self,
        roundtrip_id: UUID,
        plan: Plan | None,
        step: PlanStep,
        *,
        input_payload: Any | None = None,
        output_payload: Any | None = None,
        evidence: list[EvidenceView] | None = None,
        status: str | None = None,
        error_message: str | None = None,
        duration_ms: int | None = None,
    ) -> UUID:
        plan_id = plan.db_id if plan else None
        result = output_payload
        resolved_status = status or ("completed" if result is not None else "pending")
        sanitized_input_payload = self._sanitize_for_storage(step.args if input_payload is None else input_payload)
        output = self._sanitize_for_storage(result) if result is not None else None
        request_hash = build_signature({"tool_name": step.tool, "input": sanitized_input_payload})

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO tool_calls (
                    roundtrip_id,
                    plan_id,
                    plan_step_id,
                    step_index,
                    tool_name,
                    request_hash,
                    status,
                    input_payload,
                    output_payload,
                    error_message,
                    duration_ms,
                    goal
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    roundtrip_id,
                    plan_id,
                    step.db_id,
                    step.step_index,
                    step.tool,
                    request_hash,
                    resolved_status,
                    Jsonb(sanitized_input_payload),
                    Jsonb(output) if output is not None else None,
                    error_message,
                    duration_ms,
                    step.plan,
                ),
            )
            row = cur.fetchone()
            assert row is not None
            tool_call_id = row["id"]
            self._append_evidence_views(cur, roundtrip_id, tool_call_id, evidence or [])
            return tool_call_id

    def has_request_hash(self, roundtrip_id: UUID, request_hash: str) -> bool:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM tool_calls WHERE roundtrip_id = %s AND request_hash = %s LIMIT 1",
                (roundtrip_id, request_hash),
            )
            return cur.fetchone() is not None

    def _append_evidence_views(
        self,
        cur,
        roundtrip_id: UUID,
        tool_call_id: UUID,
        evidence_views: list[EvidenceView],
    ) -> None:
        if not evidence_views:
            return
        cur.execute(
            """
            SELECT c.user_id
            FROM conversation_roundtrip rt
            JOIN conversation c ON c.id = rt.conversation_id
            WHERE rt.id = %s
            """,
            (roundtrip_id,),
        )
        roundtrip_row = cur.fetchone()
        user_id = roundtrip_row["user_id"] if roundtrip_row else None
        for evidence_view in evidence_views:
            evidence_view.hash = build_signature(
                {
                    "tool_name": evidence_view.tool_name,
                    "item_id": evidence_view.item_id,
                    "title": evidence_view.title,
                    "summary": evidence_view.summary,
                    "urls": [url.model_dump() for url in evidence_view.urls],
                    "published_at": evidence_view.published_at,
                    "source": evidence_view.source,
                    "llm_metadata": evidence_view.llm_metadata,
                }
            )
        candidate_hashes = list({evidence_view.hash for evidence_view in evidence_views if evidence_view.hash})
        if not candidate_hashes:
            return
        cur.execute(
            """
            SELECT evidence_views.hash
            FROM evidence_views
            JOIN tool_calls ON tool_calls.id = evidence_views.tool_call_id
            WHERE tool_calls.roundtrip_id = %s AND evidence_views.hash = ANY(%s)
            """,
            (roundtrip_id, candidate_hashes),
        )
        existing_hashes = {row["hash"] for row in cur.fetchall()}
        unique_evidence_views: list[EvidenceView] = []
        for evidence_view in evidence_views:
            if evidence_view.hash in existing_hashes:
                continue
            existing_hashes.add(evidence_view.hash)
            unique_evidence_views.append(evidence_view)
        if not unique_evidence_views:
            return
        cur.executemany(
            """
            INSERT INTO evidence_views (
                id, tool_call_id, user_id, item_id, tool_name, title,
                summary, urls, image_url, published_at, source, entity_type,
                location_name, hash, llm_metadata, raw_payload
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            [
                (
                    evidence_view.id,
                    tool_call_id,
                    user_id,
                    evidence_view.item_id,
                    evidence_view.tool_name,
                    evidence_view.title,
                    evidence_view.summary,
                    Jsonb(self._sanitize_for_storage([url.model_dump() for url in evidence_view.urls])),
                    evidence_view.image_url,
                    evidence_view.published_at,
                    evidence_view.source,
                    evidence_view.entity_type,
                    evidence_view.location_name,
                    evidence_view.hash,
                    Jsonb(self._sanitize_for_storage(evidence_view.llm_metadata)),
                    Jsonb(self._sanitize_for_storage(evidence_view.raw_payload))
                    if evidence_view.raw_payload is not None
                    else None,
                )
                for evidence_view in unique_evidence_views
            ],
        )

    def update_tool_call_summary(self, tool_call_id: UUID, summary: str) -> None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "UPDATE tool_calls SET summary = %s WHERE id = %s",
                (summary, tool_call_id),
            )

    def get_tool_call(self, tool_call_id: UUID) -> ToolCall | None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM tool_calls WHERE id = %s",
                (tool_call_id,),
            )
            row = cur.fetchone()
            return ToolCall(**row) if row else None

    def get_tool_calls(self, tool_call_ids: list[UUID]) -> list[ToolCall]:
        if not tool_call_ids:
            return []
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM tool_calls WHERE id = ANY(%s)", (tool_call_ids,))
            rows = cur.fetchall()
        by_id = {row["id"]: ToolCall(**row) for row in rows}
        return [by_id[tool_call_id] for tool_call_id in tool_call_ids if tool_call_id in by_id]

    def get_tool_results(self, tool_call_ids: list[UUID]) -> list[ToolResult]:
        if not tool_call_ids:
            return []
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM tool_calls WHERE id = ANY(%s)",
                (tool_call_ids,),
            )
            tool_call_rows = cur.fetchall()
            cur.execute(
                "SELECT * FROM evidence_views WHERE tool_call_id = ANY(%s) ORDER BY created_at ASC",
                (tool_call_ids,),
            )
            evidence_rows = cur.fetchall()

        evidence_by_tool_call_id: dict[UUID, list[EvidenceView]] = {}
        for row in evidence_rows:
            evidence_by_tool_call_id.setdefault(row["tool_call_id"], []).append(
                EvidenceView(
                    id=row["id"],
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
            )

        tool_call_by_id = {row["id"]: ToolCall(**row) for row in tool_call_rows}
        results: list[ToolResult] = []
        for tool_call_id in tool_call_ids:
            tool_call = tool_call_by_id.get(tool_call_id)
            if tool_call is None:
                continue
            payload = tool_call.output_payload if isinstance(tool_call.output_payload, dict) else {}
            result = ToolResult.model_validate(payload)
            results.append(
                result.model_copy(
                    update={
                        "tool_call_id": tool_call.id,
                        "plan_step_id": tool_call.plan_step_id,
                        "tool_name": tool_call.tool_name,
                        "evidence": evidence_by_tool_call_id.get(tool_call.id, []),
                    }
                )
            )
        return results

    def copy_tool_calls(self, roundtrip_id_map: dict[UUID, UUID]) -> None:
        if not roundtrip_id_map:
            return

        tool_calls_by_roundtrip = self.get_tool_calls_by_roundtrips(list(roundtrip_id_map.keys()))
        with self._conn.cursor(row_factory=dict_row) as cur:
            for source_roundtrip_id, copied_roundtrip_id in roundtrip_id_map.items():
                for tool_call in tool_calls_by_roundtrip.get(source_roundtrip_id, []):
                    cur.execute(
                        """
                        INSERT INTO tool_calls (
                            roundtrip_id,
                            plan_id,
                            plan_step_id,
                            step_index,
                            tool_name,
                            request_hash,
                            status,
                            input_payload,
                            output_payload,
                            error_message,
                            duration_ms,
                            goal,
                            summary,
                            created_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            copied_roundtrip_id,
                            None,
                            None,
                            tool_call.step_index,
                            tool_call.tool_name,
                            tool_call.request_hash,
                            tool_call.status,
                            Jsonb(self._sanitize_for_storage(tool_call.input_payload or {})),
                            Jsonb(self._sanitize_for_storage(tool_call.output_payload)) if tool_call.output_payload is not None else None,
                            tool_call.error_message,
                            tool_call.duration_ms,
                            tool_call.goal,
                            tool_call.summary,
                            tool_call.created_at,
                        ),
                    )
                    copied_tool_call = cur.fetchone()
                    assert copied_tool_call is not None
                    cur.execute(
                        """
                        INSERT INTO evidence_views (
                            tool_call_id, user_id, item_id, tool_name, title,
                            summary, urls, image_url, published_at, source, entity_type,
                            location_name, hash, llm_metadata, raw_payload, created_at
                        )
                        SELECT
                            %s, c.user_id, evidence_views.item_id, evidence_views.tool_name, evidence_views.title,
                            evidence_views.summary, evidence_views.urls, evidence_views.image_url,
                            evidence_views.published_at, evidence_views.source, evidence_views.entity_type,
                            evidence_views.location_name, evidence_views.hash, evidence_views.llm_metadata,
                            evidence_views.raw_payload, evidence_views.created_at
                        FROM evidence_views
                        JOIN conversation_roundtrip rt ON rt.id = %s
                        JOIN conversation c ON c.id = rt.conversation_id
                        WHERE evidence_views.tool_call_id = %s
                        """,
                        (copied_tool_call["id"], copied_roundtrip_id, tool_call.id),
                    )

    def _sanitize_for_storage(self, value: Any) -> Any:
        if isinstance(value, BaseModel):
            return self._sanitize_for_storage(value.model_dump())
        if is_dataclass(value):
            return self._sanitize_for_storage(asdict(value))
        if isinstance(value, dict):
            return {
                key: self._sanitize_for_storage(item)
                for key, item in value.items()
                if not str(key).endswith("_embedding")
            }
        if isinstance(value, (list, tuple, set)):
            return [self._sanitize_for_storage(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    #TODO{: figure out how we handle cases where the data is stale. But for now its probably fine.}
    def get_tool_calls_by_roundtrips(self, roundtrip_ids: list[UUID]) -> dict[UUID, list[ToolCall]]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT * FROM tool_calls
                WHERE roundtrip_id = ANY(%s)
                ORDER BY step_index ASC
                """,
                (roundtrip_ids,),
            )
            rows = cur.fetchall()

        result: dict[UUID, list[ToolCall]] = {rid: [] for rid in roundtrip_ids}
        for row in rows:
            tc = ToolCall(**row)
            result[tc.roundtrip_id].append(tc)
        return result
