from types import SimpleNamespace

import pytest

from jobs.orchestration import job_orchestrator


@pytest.fixture
def job_planner_factory(monkeypatch, planning_result_factory):
    """Stub the planner boundary while exercising real job orchestration."""
    def create(*, steps=None, **overrides):
        result = planning_result_factory(steps=steps, **overrides)
        monkeypatch.setattr(job_orchestrator, "build_user_profile", lambda user_id: SimpleNamespace(user_id=user_id))
        monkeypatch.setattr(job_orchestrator.AgentState, "new", lambda *args, **kwargs: object())
        monkeypatch.setattr(
            job_orchestrator, "build_planner_prompt",
            lambda state: (SimpleNamespace(build_messages=lambda: [{"role": "user", "content": "prompt"}]), []),
        )
        monkeypatch.setattr(job_orchestrator, "invoke_planner", lambda *args, **kwargs: (result, None))
        return result

    return create
