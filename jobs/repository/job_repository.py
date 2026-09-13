from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from db.connection import get_connection
from jobs.models import Job, JobPlanStatus, JobRun, JobRunStatus, JobSchedule
from request_orchestrator.models.plan import Plan


class JobRepository:
    def __init__(self, conn: psycopg.Connection | None = None):
        self._conn = conn or get_connection()

    @staticmethod
    def _job_from_row(row: dict[str, Any]) -> Job:
        payload = dict(row)
        payload["schedule"] = JobSchedule(
            days_of_week=payload.pop("days_of_week"),
            run_time=payload.pop("run_time"),
            timezone=payload.pop("timezone"),
        )
        payload["plan"] = Plan.model_validate(payload["plan"])
        return Job.model_validate(payload)

    @staticmethod
    def _run_from_row(row: dict[str, Any]) -> JobRun:
        return JobRun.model_validate(row)

    @staticmethod
    def _require_user_id(user_id: str) -> str:
        resolved_user_id = user_id.strip()
        if not resolved_user_id:
            raise ValueError("user_id is required")
        return resolved_user_id

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        return sha256(prompt.encode("utf-8")).hexdigest()

    def create_job(
        self,
        *,
        user_id: str,
        name: str,
        prompt: str,
        plan: Plan,
        schedule: JobSchedule,
        next_execution_at: datetime,
        enabled: bool = False,
        plan_version: int = 1,
    ) -> Job:
        resolved_user_id = self._require_user_id(user_id)
        if not name.strip():
            raise ValueError("name is required")
        if not prompt.strip():
            raise ValueError("prompt is required")
        if plan_version < 1:
            raise ValueError("plan_version must be positive")
        if enabled and not plan.steps:
            raise ValueError("cannot enable a job without a valid plan")

        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO jobs (
                    user_id, name, prompt, plan, plan_version, plan_generated_at,
                    plan_status, plan_prompt_hash, enabled, days_of_week, run_time, timezone,
                    next_execution_at
                )
                VALUES (%s, %s, %s, %s, %s, now(), %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, user_id, name, prompt, plan, plan_version,
                          plan_generated_at, plan_status, plan_prompt_hash,
                          enabled, days_of_week, run_time,
                          timezone, next_execution_at, created_at, updated_at
                """,
                (
                    resolved_user_id,
                    name.strip(),
                    prompt.strip(),
                    Jsonb(plan.model_dump(mode="json")),
                    plan_version,
                    JobPlanStatus.READY.value if plan.steps else JobPlanStatus.FAILED.value,
                    self._prompt_hash(prompt.strip()),
                    enabled,
                    schedule.days_of_week,
                    schedule.run_time,
                    schedule.timezone,
                    next_execution_at,
                ),
            )
            row = cur.fetchone()
            assert row is not None
            return self._job_from_row(row)

    def get_job(self, job_id: UUID, *, user_id: str) -> Job | None:
        resolved_user_id = self._require_user_id(user_id)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, user_id, name, prompt, plan, plan_version,
                       plan_generated_at, plan_status, plan_prompt_hash,
                       enabled, days_of_week, run_time,
                       timezone, next_execution_at, created_at, updated_at
                FROM jobs
                WHERE id = %s AND user_id = %s
                """,
                (job_id, resolved_user_id),
            )
            row = cur.fetchone()
            return self._job_from_row(row) if row else None

    def list_jobs(self, user_id: str, *, enabled: bool | None = None) -> list[Job]:
        resolved_user_id = self._require_user_id(user_id)
        sql = """
            SELECT id, user_id, name, prompt, plan, plan_version,
                   plan_generated_at, plan_status, plan_prompt_hash,
                   enabled, days_of_week, run_time,
                   timezone, next_execution_at, created_at, updated_at
            FROM jobs WHERE user_id = %s
        """
        params: list[Any] = [resolved_user_id]
        if enabled is not None:
            sql += " AND enabled = %s"
            params.append(enabled)
        sql += " ORDER BY name ASC, created_at ASC"
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return [self._job_from_row(row) for row in cur.fetchall()]

    def list_due_jobs(self, *, now: datetime) -> list[Job]:
        """List jobs eligible for the scheduler's current enqueue pass."""
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, user_id, name, prompt, plan, plan_version,
                       plan_generated_at, plan_status, plan_prompt_hash,
                       enabled, days_of_week, run_time, timezone,
                       next_execution_at, created_at, updated_at
                FROM jobs
                WHERE enabled = TRUE
                  AND plan_status = 'ready'
                  AND next_execution_at <= %s
                ORDER BY next_execution_at ASC, id ASC
                """,
                (now,),
            )
            return [self._job_from_row(row) for row in cur.fetchall()]

    def update_job(
        self,
        job_id: UUID,
        *,
        user_id: str,
        name: str,
        prompt: str,
        schedule: JobSchedule,
        next_execution_at: datetime,
        enabled: bool,
    ) -> Job | None:
        resolved_user_id = self._require_user_id(user_id)
        if not name.strip() or not prompt.strip():
            raise ValueError("name and prompt are required")
        current_job = self.get_job(job_id, user_id=resolved_user_id)
        if current_job is None:
            return None
        resolved_prompt = prompt.strip()
        prompt_changed = self._prompt_hash(resolved_prompt) != current_job.plan_prompt_hash
        if enabled and (
            prompt_changed
            or current_job.plan_status != JobPlanStatus.READY
            or not current_job.plan.steps
        ):
            raise ValueError("cannot enable a job until its plan is regenerated")
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE jobs
                SET name = %s, prompt = %s, enabled = %s,
                    plan_status = CASE WHEN %s THEN 'stale' ELSE plan_status END,
                    days_of_week = %s, run_time = %s, timezone = %s,
                    next_execution_at = %s,
                    updated_at = now()
                WHERE id = %s AND user_id = %s
                RETURNING id, user_id, name, prompt, plan, plan_version,
                          plan_generated_at, plan_status, plan_prompt_hash,
                          enabled, days_of_week, run_time,
                          timezone, next_execution_at, created_at, updated_at
                """,
                (name.strip(), resolved_prompt, enabled, prompt_changed, schedule.days_of_week,
                 schedule.run_time, schedule.timezone, next_execution_at, job_id, resolved_user_id),
            )
            row = cur.fetchone()
            return self._job_from_row(row) if row else None

    def replace_plan(self, job_id: UUID, *, user_id: str, plan: Plan) -> Job | None:
        resolved_user_id = self._require_user_id(user_id)
        if not plan.steps:
            raise ValueError("plan must contain at least one step")
        current_job = self.get_job(job_id, user_id=resolved_user_id)
        if current_job is None:
            return None
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE jobs
                SET plan = %s, plan_version = plan_version + 1,
                    plan_status = 'ready', plan_prompt_hash = %s,
                    plan_generated_at = now(), updated_at = now()
                WHERE id = %s AND user_id = %s
                RETURNING id, user_id, name, prompt, plan, plan_version,
                          plan_generated_at, plan_status, plan_prompt_hash,
                          enabled, days_of_week, run_time,
                          timezone, next_execution_at, created_at, updated_at
                """,
                (Jsonb(plan.model_dump(mode="json")), self._prompt_hash(current_job.prompt), job_id, resolved_user_id),
            )
            row = cur.fetchone()
            return self._job_from_row(row) if row else None

    def set_enabled(self, job_id: UUID, *, user_id: str, enabled: bool) -> bool:
        resolved_user_id = self._require_user_id(user_id)
        current_job = self.get_job(job_id, user_id=resolved_user_id)
        if current_job is None:
            return False
        if enabled and (
            current_job.plan_status != JobPlanStatus.READY
            or not current_job.plan.steps
            or current_job.plan_prompt_hash != self._prompt_hash(current_job.prompt)
        ):
            raise ValueError("cannot enable a job until its plan is regenerated")
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET enabled = %s, updated_at = now() WHERE id = %s AND user_id = %s",
                (enabled, job_id, resolved_user_id),
            )
            return cur.rowcount > 0

    def _create_job_run(
        self,
        job_id: UUID,
        *,
        user_id: str,
        scheduled_for: datetime,
        require_enabled: bool,
    ) -> JobRun:
        resolved_user_id = self._require_user_id(user_id)
        if scheduled_for.tzinfo is None:
            raise ValueError("scheduled_for must be timezone-aware")
        deduplication_key = f"{job_id}:{scheduled_for.astimezone(timezone.utc).isoformat()}"
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO job_runs (job_id, scheduled_for, deduplication_key)
                SELECT id, %s, %s FROM jobs
                WHERE id = %s AND user_id = %s
                  AND plan_status = 'ready'
                  AND (%s = FALSE OR enabled = TRUE)
                ON CONFLICT (job_id, scheduled_for) DO UPDATE
                    SET job_id = EXCLUDED.job_id
                RETURNING id, job_id, scheduled_for, status, queue_message_id,
                          deduplication_key, created_at, updated_at
                """,
                (scheduled_for, deduplication_key, job_id, resolved_user_id, require_enabled),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError("job is not found, disabled, or has no ready plan")
            return self._run_from_row(row)

    def create_job_run(self, job_id: UUID, *, user_id: str, scheduled_for: datetime) -> JobRun:
        """Create an occurrence for the scheduler; the job must be enabled."""
        return self._create_job_run(
            job_id,
            user_id=user_id,
            scheduled_for=scheduled_for,
            require_enabled=True,
        )

    def create_manual_job_run(self, job_id: UUID, *, user_id: str, scheduled_for: datetime) -> JobRun:
        """Create an explicit manual occurrence; disabled jobs are allowed."""
        return self._create_job_run(
            job_id,
            user_id=user_id,
            scheduled_for=scheduled_for,
            require_enabled=False,
        )

    def claim_due_job_run(
        self,
        *,
        job_id: UUID,
        scheduled_for: datetime,
        next_execution_at: datetime,
    ) -> JobRun | None:
        """Atomically advance a due job and create its enqueueable run."""
        if scheduled_for.tzinfo is None or next_execution_at.tzinfo is None:
            raise ValueError("scheduled timestamps must be timezone-aware")
        if next_execution_at <= scheduled_for:
            raise ValueError("next_execution_at must be after scheduled_for")
        deduplication_key = f"{job_id}:{scheduled_for.astimezone(timezone.utc).isoformat()}"
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                WITH claimed AS (
                    UPDATE jobs
                    SET next_execution_at = %s, updated_at = now()
                    WHERE id = %s
                      AND enabled = TRUE
                      AND plan_status = 'ready'
                      AND next_execution_at = %s
                    RETURNING id
                )
                INSERT INTO job_runs (job_id, scheduled_for, deduplication_key)
                SELECT id, %s, %s FROM claimed
                ON CONFLICT (job_id, scheduled_for) DO UPDATE
                    SET job_id = EXCLUDED.job_id
                RETURNING id, job_id, scheduled_for, status, queue_message_id,
                          deduplication_key, created_at, updated_at
                """,
                (next_execution_at, job_id, scheduled_for, scheduled_for, deduplication_key),
            )
            row = cur.fetchone()
            return self._run_from_row(row) if row else None

    def update_run_status(
        self,
        run_id: UUID,
        *,
        user_id: str,
        status: JobRunStatus,
        queue_message_id: str | None = None,
    ) -> JobRun | None:
        resolved_user_id = self._require_user_id(user_id)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE job_runs
                SET status = %s, queue_message_id = COALESCE(%s, queue_message_id), updated_at = now()
                WHERE id = %s
                  AND EXISTS (
                      SELECT 1 FROM jobs
                      WHERE jobs.id = job_runs.job_id AND jobs.user_id = %s
                  )
                RETURNING id, job_id, scheduled_for, status, queue_message_id,
                          deduplication_key, created_at, updated_at
                """,
                (status.value, queue_message_id, run_id, resolved_user_id),
            )
            row = cur.fetchone()
            return self._run_from_row(row) if row else None
