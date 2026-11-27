from __future__ import annotations


from agents.main.request import TOP_LEVEL_PROFILE
from agent_runtime.routing.main_router import router as main_router
from agent_runtime.constants import EVALUATE_EDGE


def test_main_router_ignores_needs_replan_and_evaluates(agent_state_factory, default_plan, set_plan_results):
    state = agent_state_factory(profile=TOP_LEVEL_PROFILE)
    set_plan_results(state, plan=default_plan, results={"E1": {"ok": True}}, needs_replan=True)

    assert main_router(state) == EVALUATE_EDGE


def test_main_router_evaluates_when_plan_does_not_need_replan(
    agent_state_factory,
    default_plan,
    set_plan_results,
):
    state = agent_state_factory(profile=TOP_LEVEL_PROFILE)
    set_plan_results(state, plan=default_plan, results={"E1": {"ok": True}}, needs_replan=False)

    assert main_router(state) == EVALUATE_EDGE
