from __future__ import annotations

from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from db.connection import get_connection
from request_orchestrator.models.plan import Plan, PlanKind, PlanStatus


class PlanRepository:
    def __init__(self, conn: psycopg.Connection | None = None):
        self._conn = conn or get_connection()

    def save_plan(
        self,
        *,
        plan: Plan,
        roundtrip_id: UUID | None = None,
        user_id: str | None = None,
        job_id: UUID | None = None,
        version: int = 1,
        prompt_hash: str | None = None,
        plan_kind: PlanKind = PlanKind.INTERACTIVE,
    ) -> UUID:
        """Store one plan snapshot for either an interactive roundtrip or a job."""
        if (roundtrip_id is None) == (job_id is None):
            raise ValueError("exactly one of roundtrip_id or job_id is required")
        if not isinstance(plan_kind, PlanKind):
            raise ValueError("unsupported plan kind")
        if version < 1:
            raise ValueError("version must be positive")
        if job_id is not None and not user_id:
            raise ValueError("user_id is required for job plans")

        steps_payload = [step.model_dump(mode="json") for step in plan.steps]
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO plans (
                    roundtrip_id, user_id, job_id, steps, current_step_index,
                    status, version, prompt_hash, plan_kind
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    roundtrip_id,
                    user_id,
                    job_id,
                    Jsonb(steps_payload),
                    plan.current_step_index,
                    plan.status.value,
                    version,
                    prompt_hash,
                    plan_kind.value,
                ),
            )
            return cur.fetchone()["id"]

    def save_interactive_plan(self, roundtrip_id: UUID, plan: Plan) -> UUID:
        return self.save_plan(
            plan=plan,
            roundtrip_id=roundtrip_id,
            plan_kind=PlanKind.INTERACTIVE,
        )

    def get_current_job_plan(self, *, job_id: UUID, user_id: str) -> Plan | None:
        """Load the current plan after verifying job ownership."""
        if not user_id.strip():
            raise ValueError("user_id is required")
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT plans.id, plans.steps, plans.current_step_index, plans.status
                FROM jobs
                JOIN plans ON plans.id = jobs.current_plan_id
                WHERE jobs.id = %s AND jobs.user_id = %s
                """,
                (job_id, user_id.strip()),
            )
            row = cur.fetchone()
        if row is None:
            return None
        plan = Plan.model_validate({
            "steps": row["steps"],
            "current_step_index": row["current_step_index"],
            "status": row["status"],
        })
        plan.db_id = row["id"]
        return plan

    def save_job_plan(
        self,
        *,
        job_id: UUID,
        user_id: str,
        plan: Plan,
        version: int,
        prompt_hash: str,
    ) -> UUID:
        return self.save_plan(
            plan=plan,
            user_id=user_id,
            job_id=job_id,
            version=version,
            prompt_hash=prompt_hash,
            plan_kind=PlanKind.JOB,
        )

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
