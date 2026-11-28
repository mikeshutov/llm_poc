from dataclasses import dataclass, replace
from uuid import uuid4

import pytest

from agent_runtime.models.agent_execution_context import AgentExecutionContext
from agent_runtime.models.agent_profile import AgentExecutionStrategy, AgentKind, AgentProfile
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.main_state import MainState
from agents.models.agent import Agent
from agents.delegate.profile import DELEGATE_PROFILE
from retrieval.evidence import ToolResult


@dataclass(frozen=True)
class _NamedTool:
    name: str


@pytest.fixture
def agent_profile_factory():
    def create(**overrides):
        return AgentProfile(**{"name": "test_agent", "scope": "main_agent", **overrides})

    return create


@pytest.fixture
def user_agent_profile_factory(agent_profile_factory):
    def create(**overrides):
        return agent_profile_factory(**{
            "name": "assistant", "kind": AgentKind.USER_AGENT, **overrides,
        })

    return create


@pytest.fixture
def execution_context_factory():
    def create(*, user_id=None, **overrides):
        context = AgentExecutionContext.new(**overrides)
        if user_id is not None:
            context.user_profile = context.user_profile.model_copy(update={"user_id": user_id}, deep=True)
        return context

    return create


@pytest.fixture
def agent_state_factory(agent_profile_factory):
    def create(*, profile=None, profile_overrides=None, plan=None, task="", llm=None, user_id=None, **kwargs):
        if profile is None:
            profile = agent_profile_factory(
                static_tools=[_NamedTool(step.tool) for step in plan.steps] if plan else [],
            )
        if profile_overrides:
            profile = replace(profile, **profile_overrides)
        if user_id is not None:
            context = kwargs.get("execution_context") or AgentExecutionContext.new()
            kwargs["execution_context"] = replace(
                context,
                user_profile=context.user_profile.model_copy(update={"user_id": user_id}, deep=True),
            )
        state = AgentState.new(
            agent_profile=profile, task=task, llm=object() if llm is None else llm, **kwargs,
        )
        if plan is not None:
            state.node_states.planner.plan = plan
            state.node_states.planner.plan_count = 1
        return state

    return create


@pytest.fixture
def react_agent_state_factory(agent_state_factory):
    def create(*, max_turns=3, task="Find the answer", profile_overrides=None, **kwargs):
        return agent_state_factory(
            task=task,
            profile_overrides={
                "name": "react_agent", "execution_strategy": AgentExecutionStrategy.REACT,
                "max_turns": max_turns, **(profile_overrides or {}),
            },
            **kwargs,
        )

    return create


@pytest.fixture
def db_agent_factory():
    def create(**overrides):
        return Agent(**{
            "id": uuid4(), "name": "researcher", "planner_instruction": "Plan the request.",
            **overrides,
        })

    return create


@pytest.fixture
def main_state_factory():
    def create(*, task="", agent_profiles=(), llm=None, **kwargs):
        return MainState.new(
            task=task, agent_profiles=list(agent_profiles),
            llm=object() if llm is None else llm, **kwargs,
        )

    return create


@pytest.fixture
def set_plan_results():
    """Attach a plan and its results, keyed by local step ID, without persistence."""
    def update(state, *, plan=None, results=None, plan_count=1, needs_replan=False):
        if plan is not None:
            state.node_states.planner.plan = plan.model_copy(deep=True)
            state.node_states.planner.plan_count = max(0, plan_count)
        state.node_states.planner.needs_replan = needs_replan
        steps = {step.id: step for step in (plan.steps if plan is not None else [])}
        tool_results = []
        for step_id, value in (results or {}).items():
            step = steps[step_id]
            result = value if isinstance(value, ToolResult) else ToolResult(result=value)
            tool_results.append(result.model_copy(update={
                "tool_call_id": result.tool_call_id or uuid4(),
                "plan_step_id": result.plan_step_id or step.db_id,
                "tool_name": result.tool_name or step.tool,
            }))
        state.direct_tool_results[:] = tool_results
        return state

    return update


@pytest.fixture
def delegated_agent_state_factory(agent_state_factory, agent_profile_factory, execution_context_factory,
                                  plan_factory, plan_step_factory):
    def create(*, child=None, child_profile=None, task="Delegated task", evidence_ids=(),
               execution_context=None, parent_profile=None):
        if child is None:
            if child_profile is None:
                child_profile = DELEGATE_PROFILE
            child = agent_state_factory(
                profile=child_profile,
                execution_context=execution_context or execution_context_factory(conversation_id="conversation-test"),
            )
        parent = agent_state_factory(
            profile=parent_profile or agent_profile_factory(name="coordinator"),
            execution_context=child.execution_context,
            plan=plan_factory(steps=[plan_step_factory(
                tool=child.agent_profile.name, plan="Delegate the assigned task",
                args={"task": task, "evidence_ids": list(evidence_ids)},
            )]),
        )
        parent.available_agent_states = {child.agent_profile.name: child}
        return parent

    return create
