from __future__ import annotations

from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


class ToolCallRepository:
    def __init__(self, conn: psycopg.Connection):
        self._conn = conn

    def append_tool_calls(
        self,
        roundtrip_id: UUID,
        traces: list[dict[str, Any]],
    ) -> None:
        if not traces:
            return
        with self._conn.cursor(row_factory=dict_row) as cur:
            for trace in traces:
                cur.execute(
                    """
                    INSERT INTO tool_calls (
                        roundtrip_id,
                        call_index,
                        tool_name,
                        status,
                        reason,
                        input_payload,
                        output_payload,
                        error_message,
                        duration_ms,
                        goal,
                        done
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        roundtrip_id,
                        trace.get("call_index"),
                        trace.get("tool_name"),
                        trace.get("status"),
                        trace.get("reason"),
                        Jsonb(trace.get("input_payload") or {}),
                        Jsonb(trace.get("output_payload")) if trace.get("output_payload") is not None else None,
                        trace.get("error_message"),
                        trace.get("duration_ms"),
                        trace.get("goal"),
                        trace.get("done"),
                    ),
                )
