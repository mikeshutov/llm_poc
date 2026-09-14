from types import SimpleNamespace
from uuid import uuid4

from jobs.orchestration import job_orchestrator
from jobs.service import generate_job_plan
from request_orchestrator.models.plan import PlanStep, PlanningResult


def _planning_result(*, steps: list[PlanStep], status: str = "ready") -> PlanningResult:
    return PlanningResult(steps=steps, status=status)


def test_job_orchestrator_returns_generated_plan(monkeypatch) -> None:
    step = PlanStep(id="step-1", plan="Search", tool="search", args={})
    monkeypatch.setattr(job_orchestrator, "build_user_profile", lambda user_id: SimpleNamespace(user_id=user_id))
    monkeypatch.setattr(job_orchestrator.AgentState, "new", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        job_orchestrator,
        "build_planner_prompt",
        lambda state: SimpleNamespace(build=lambda: "prompt", to_log_input_object=lambda: {}),
    )
    monkeypatch.setattr(
        job_orchestrator,
        "invoke_planner",
        lambda *args, **kwargs: (_planning_result(steps=[step]), None),
    )

    plan, error = job_orchestrator.JobOrchestrator().generate_plan(
        user_id="user-1",
        prompt="Find something",
    )

    assert error is None
    assert plan is not None
    assert plan.steps[0].tool == "search"


def test_job_orchestrator_returns_error_for_empty_plan(monkeypatch) -> None:
    monkeypatch.setattr(job_orchestrator, "build_user_profile", lambda user_id: SimpleNamespace(user_id=user_id))
    monkeypatch.setattr(job_orchestrator.AgentState, "new", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        job_orchestrator,
        "build_planner_prompt",
        lambda state: SimpleNamespace(build=lambda: "prompt", to_log_input_object=lambda: {}),
    )
    monkeypatch.setattr(
        job_orchestrator,
        "invoke_planner",
        lambda *args, **kwargs: (_planning_result(steps=[]), None),
    )

    plan, error = job_orchestrator.JobOrchestrator().generate_plan(
        user_id="user-1",
        prompt="Find something",
    )

    assert plan is None
    assert error == "planner did not produce a usable plan"


def test_generate_job_plan_persists_only_successful_plan(monkeypatch) -> None:
    job_id = uuid4()
    replacement = []
    job = SimpleNamespace(prompt="Find something")
    repository = SimpleNamespace(
        get_job=lambda requested_id, *, user_id: job if requested_id == job_id and user_id == "user-1" else None,
        replace_plan=lambda requested_id, *, user_id, plan: replacement.append((requested_id, user_id, plan)),
    )
    plan = object()

    class FakeOrchestrator:
        def generate_plan(self, *, user_id, prompt):
            assert user_id == "user-1"
            assert prompt == job.prompt
            return plan, None

    monkeypatch.setattr("jobs.service.get_job_repo", lambda: repository)
    monkeypatch.setattr("jobs.service.JobOrchestrator", FakeOrchestrator)

    result_plan, error = generate_job_plan(job_id=job_id, user_id="user-1")

    assert result_plan is plan
    assert error is None
    assert replacement == [(job_id, "user-1", plan)]
