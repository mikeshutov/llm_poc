from __future__ import annotations

from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from artifacts.models import Artifact
from artifacts.repository.artifact_repository import ArtifactRepository
from db.connection import get_connection
from jobs.models import JobResult, JobResultStatus


class JobResultRepository:
    def __init__(self, conn: psycopg.Connection | None = None):
        self._conn = conn or get_connection()
        self._artifact_repo = ArtifactRepository(self._conn)

    @staticmethod
    def _result_from_row(row: dict[str, Any]) -> JobResult:
        return JobResult.model_validate(row)

    @staticmethod
    def _artifact_from_row(row: dict[str, Any]) -> Artifact:
        payload = dict(row)
        payload.pop("job_result_id", None)
        return Artifact.model_validate(payload)

    @staticmethod
    def _require_user_id(user_id: str) -> str:
        resolved_user_id = user_id.strip()
        if not resolved_user_id:
            raise ValueError("user_id is required")
        return resolved_user_id

    def create_pending_result(
        self,
        *,
        user_id: str,
        job_id: UUID,
        job_run_id: UUID,
        metadata: dict[str, Any] | None = None,
    ) -> JobResult:
        resolved_user_id = self._require_user_id(user_id)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO job_results (job_id, job_run_id, metadata)
                SELECT %s, job_runs.id, %s
                FROM job_runs
                JOIN jobs ON jobs.id = job_runs.job_id
                WHERE job_runs.id = %s AND job_runs.job_id = %s AND jobs.user_id = %s
                RETURNING id, job_id, job_run_id, status, text, error_text,
                          metadata, created_at, completed_at
                """,
                (job_id, Jsonb(metadata or {}), job_run_id, job_id, resolved_user_id),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError("job run not found for user")
            return self._result_from_row(row)

    def complete_result(
        self,
        result_id: UUID,
        *,
        user_id: str,
        text: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> JobResult | None:
        self._require_user_id(user_id)
        return self._update_result(result_id, user_id=user_id, status=JobResultStatus.SUCCEEDED, text=text, metadata=metadata)

    def fail_result(
        self,
        result_id: UUID,
        *,
        user_id: str,
        error_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> JobResult | None:
        self._require_user_id(user_id)
        return self._update_result(result_id, user_id=user_id, status=JobResultStatus.FAILED, error_text=error_text, metadata=metadata)

    def _update_result(
        self,
        result_id: UUID,
        *,
        user_id: str,
        status: JobResultStatus,
        text: str | None = None,
        error_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> JobResult | None:
        resolved_user_id = self._require_user_id(user_id)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                UPDATE job_results
                SET status = %s, text = COALESCE(%s, text),
                    error_text = %s,
                    metadata = COALESCE(%s, metadata), completed_at = now()
                WHERE id = %s
                  AND status = 'pending'
                  AND EXISTS (
                      SELECT 1 FROM jobs
                      WHERE jobs.id = job_results.job_id AND jobs.user_id = %s
                  )
                RETURNING id, job_id, job_run_id, status, text, error_text,
                          metadata, created_at, completed_at
                """,
                (status.value, text, error_text, Jsonb(metadata) if metadata is not None else None, result_id, resolved_user_id),
            )
            row = cur.fetchone()
            return self._result_from_row(row) if row else None

    def list_for_job(self, job_id: UUID, *, user_id: str, limit: int = 100) -> list[JobResult]:
        resolved_user_id = self._require_user_id(user_id)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, job_id, job_run_id, status, text, error_text,
                       metadata, created_at, completed_at
                FROM job_results
                WHERE job_id = %s AND EXISTS (
                    SELECT 1 FROM jobs WHERE jobs.id = job_results.job_id AND jobs.user_id = %s
                )
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (job_id, resolved_user_id, limit),
            )
            return [self._result_from_row(row) for row in cur.fetchall()]

    def add_artifact(
        self,
        result_id: UUID,
        *,
        user_id: str,
        artifact_type: str,
        artifact_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        resolved_user_id = self._require_user_id(user_id)
        if not artifact_type.strip() or not artifact_id.strip():
            raise ValueError("artifact_type and artifact_id are required")
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT 1
                FROM job_results
                JOIN jobs ON jobs.id = job_results.job_id
                WHERE job_results.id = %s AND jobs.user_id = %s
                """,
                (result_id, resolved_user_id),
            )
            if cur.fetchone() is None:
                raise ValueError("job result not found for user")
            artifact = self._artifact_repo.create_or_get(
                user_id=resolved_user_id,
                artifact_type=artifact_type,
                artifact_id=artifact_id,
                metadata=metadata,
            )
            cur.execute(
                """
                INSERT INTO job_result_artifacts (job_result_id, artifact_id)
                SELECT %s, %s
                RETURNING job_result_id
                """,
                (result_id, artifact.id),
            )
            row = cur.fetchone()
            return artifact

    def list_artifacts(self, result_id: UUID, *, user_id: str) -> list[Artifact]:
        resolved_user_id = self._require_user_id(user_id)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT a.id, a.user_id, a.artifact_type, a.artifact_id, a.metadata, a.created_at
                FROM job_result_artifacts jra
                JOIN artifacts a ON a.id = jra.artifact_id
                WHERE jra.job_result_id = %s AND EXISTS (
                    SELECT 1
                    FROM job_results
                    JOIN jobs ON jobs.id = job_results.job_id
                    WHERE job_results.id = jra.job_result_id
                      AND jobs.user_id = %s
                )
                ORDER BY a.created_at ASC
                """,
                (result_id, resolved_user_id),
            )
            return [self._artifact_from_row(row) for row in cur.fetchall()]
