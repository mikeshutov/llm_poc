from __future__ import annotations

from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from agents.general.agentstate.model import IterationState
from agents.general.models.plan import PlanStep
from tool.repository.models import ToolCall
from db.connection import get_connection


class ToolCallRepository:
    def __init__(self):
        self._conn = get_connection()

    def append_tool_call(
        self,
        roundtrip_id: UUID,
        iteration: IterationState,
        step: PlanStep,
    ) -> None:
        plan = iteration.plan
        plan_id = plan.db_id if plan else None
        result = iteration.results.get(step.id)  # keyed by string ref "E1"
        status = "completed" if result is not None else "pending"
        output = result if isinstance(result, dict) else {"result": str(result)} if result is not None else None

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO tool_calls (
                    roundtrip_id,
                    plan_id,
                    plan_step_id,
                    step_index,
                    tool_name,
                    status,
                    input_payload,
                    output_payload,
                    goal
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    roundtrip_id,
                    plan_id,
                    step.db_id,
                    step.step_index,
                    step.tool,
                    status,
                    Jsonb(step.args or {}),
                    Jsonb(output) if output is not None else None,
                    step.plan,
                ),
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
