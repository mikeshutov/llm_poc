from copy import deepcopy

import pytest

from agent_runtime.models.plan import Plan, PlanStep, PlanningResult


@pytest.fixture
def plan_step_factory():
    def create(**overrides):
        return PlanStep(**deepcopy({
            "id": "E1", "plan": "Search", "tool": "web_search",
            "args": {}, **overrides,
        }))

    return create


@pytest.fixture
def plan_factory(plan_step_factory):
    def create(*, steps=None, **overrides):
        return Plan(
            steps=[plan_step_factory()] if steps is None else deepcopy(steps),
            **overrides,
        )

    return create


@pytest.fixture
def default_plan(plan_factory):
    return plan_factory()


@pytest.fixture
def planning_result_factory(plan_step_factory):
    def create(*, steps=None, **overrides):
        return PlanningResult(
            steps=[plan_step_factory()] if steps is None else deepcopy(steps),
            **overrides,
        )

    return create
