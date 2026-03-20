from __future__ import annotations

from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from agents.general.models.plan import Plan, PlanStatus
from db.connection import get_connection


class PlanRepository:
    def __init__(self):
        self._conn = get_connection()

    def save_plan(self, roundtrip_id: UUID, plan: Plan) -> UUID:
        steps_payload = [step.model_dump(mode="json") for step in plan.steps]
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO plans (roundtrip_id, steps, current_step_index, status)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (
                    roundtrip_id,
                    Jsonb(steps_payload),
                    plan.current_step_index,
                    plan.status.value,
                ),
            )
            return cur.fetchone()["id"]

    def update_status(self, plan_id: UUID, status: PlanStatus, current_step_index: int | None = None) -> None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE plans
                SET status = %s,
                    current_step_index = COALESCE(%s, current_step_index),
                    updated_at = now()
                WHERE id = %s
                """,
                (status.value, current_step_index, plan_id),
            )
