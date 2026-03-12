from __future__ import annotations

from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from agent.agentstate.model import IterationState
from db.connection import get_connection


class ToolCallRepository:
    def __init__(self):
        self._conn = get_connection()

    def append_tool_calls(
        self,
        roundtrip_id: UUID,
        iteration_trace: list[IterationState],
    ) -> None:
        if not iteration_trace:
            return

        call_index = 0
        with self._conn.cursor(row_factory=dict_row) as cur:
            for iteration in iteration_trace:
                plan = iteration.plan
                plan_id = plan.db_id if plan else None

                for step in (plan.steps if plan else []):
                    result = iteration.results.get(step.id)
                    status = "completed" if result is not None else "pending"
                    output = result if isinstance(result, dict) else {"result": str(result)} if result is not None else None

                    cur.execute(
                        """
                        INSERT INTO tool_calls (
                            roundtrip_id,
                            plan_id,
                            plan_step_id,
                            step_index,
                            call_index,
                            tool_name,
                            status,
                            input_payload,
                            output_payload,
                            goal
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            roundtrip_id,
                            plan_id,
                            step.id,
                            step.step_index,
                            call_index,
                            step.tool,
                            status,
                            Jsonb(step.args or {}),
                            Jsonb(output) if output is not None else None,
                            step.plan,
                        ),
                    )
                    call_index += 1
