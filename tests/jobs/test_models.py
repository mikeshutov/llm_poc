from datetime import time
from uuid import uuid4

import pytest
from pydantic import ValidationError

from assets.models import AssetReference
from jobs.models import JobSchedule
from jobs.repository.job_repository import JobRepository
from jobs.repository.job_result_repository import JobResultRepository
from agent_runtime.models.plan import Plan, PlanStep, PlanningResult


def test_job_schedule_normalizes_and_validates_weekdays() -> None:
    schedule = JobSchedule(
        days_of_week=[5, 1, 5],
        run_time=time(9, 30),
        timezone="America/Toronto",
    )

    assert schedule.days_of_week == [1, 5]


def test_job_schedule_rejects_invalid_timezone_and_weekday() -> None:
    with pytest.raises(ValidationError):
        JobSchedule(days_of_week=[0], run_time=time(9), timezone="America/Toronto")
    with pytest.raises(ValidationError):
        JobSchedule(days_of_week=[1], run_time=time(9), timezone="Not/A_Timezone")


def test_asset_reference_normalizes_generic_reference() -> None:
    reference = AssetReference(
        id=uuid4(),
        user_id="user-123",
        asset_type=" report ",
        asset_id=" report-123 ",
        created_at="2026-01-01T00:00:00Z",
    )

    assert reference.asset_type == "report"
    assert reference.asset_id == "report-123"


def test_plan_can_be_round_tripped_as_job_payload() -> None:
    plan = Plan(
        steps=[PlanStep(id="step-1", plan="Find products", hypothesis="The database contains matching products.", tool="find_products")],
    )

    restored = Plan.model_validate(plan.model_dump(mode="json"))

    assert restored.steps[0].tool == "find_products"
    assert restored.steps[0].step_index == 0
    assert restored.steps[0].hypothesis == "The database contains matching products."


def test_planning_result_accepts_step_hypothesis() -> None:
    result = PlanningResult.model_validate({
        "steps": [{
            "id": "step-1",
            "plan": "Find products",
            "hypothesis": "The database contains matching products.",
            "tool": "find_products",
        }],
    })

    assert result.steps[0].hypothesis == "The database contains matching products."


def test_job_repositories_require_user_scope() -> None:
    with pytest.raises(ValueError, match="user_id is required"):
        JobRepository._require_user_id(" ")
    with pytest.raises(ValueError, match="user_id is required"):
        JobResultRepository._require_user_id("")
