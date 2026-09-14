from __future__ import annotations

from uuid import UUID

from jobs.orchestration import JobOrchestrator
from request_orchestrator.models.plan import Plan
from jobs.repository.repo_factory import get_job_repo


def generate_job_plan(
    *,
    job_id: UUID,
    user_id: str,
) -> tuple[Plan | None, str | None]:
    """Preplan for jobs this will be used for execution."""
    repository = get_job_repo()
    job = repository.get_job(job_id, user_id=user_id)
    if job is None:
        raise LookupError("job not found")

    result = JobOrchestrator().generate_plan(user_id=user_id, prompt=job.prompt)
    plan, error = result
    if plan is not None:
        repository.replace_plan(job_id, user_id=user_id, plan=plan)
    return plan, error
