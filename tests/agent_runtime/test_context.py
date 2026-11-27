from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Barrier
from uuid import uuid4

import pytest

from agent_runtime.context import (
    bind_agent_context,
    get_current_agent_name,
    get_current_agent_state,
    get_current_conversation_id,
    get_current_roundtrip_id,
    get_current_user_id,
)
from agent_runtime.orchestrator import run_agent


def test_no_active_agent_has_no_ambient_context():
    assert get_current_agent_state() is None
    assert get_current_agent_name() is None
    assert get_current_conversation_id() is None
    assert get_current_roundtrip_id() is None
    assert get_current_user_id() is None


def test_context_readers_follow_live_agent_state(agent_state_factory):
    state = agent_state_factory()
    with bind_agent_context(state):
        assert get_current_agent_state() is state
        assert get_current_roundtrip_id() is None
        state.execution_context.conversation_id = "conversation-2"
        state.execution_context.roundtrip_id = uuid4()
        state.execution_context.user_profile.user_id = "user-2"
        state.agent_profile.name = "updated-agent"

        assert get_current_conversation_id() == state.execution_context.conversation_id
        assert get_current_roundtrip_id() == str(state.execution_context.roundtrip_id)
        assert get_current_user_id() == state.execution_context.user_profile.user_id
        assert get_current_agent_name() == state.agent_profile.name

    assert get_current_agent_state() is None


@pytest.mark.parametrize("fails", [False, True])
def test_nested_agent_scope_restores_parent(agent_profile_factory, agent_state_factory, fails):
    parent = agent_state_factory(profile=agent_profile_factory(name="parent", scope="main_agent"))
    child = agent_state_factory(profile=agent_profile_factory(name="child", scope="main_agent"))
    with bind_agent_context(parent):
        try:
            with bind_agent_context(child):
                assert get_current_agent_state() is child
                assert get_current_agent_name() == "child"
                if fails:
                    raise RuntimeError("Child failed")
        except RuntimeError:
            assert fails

        assert get_current_agent_state() is parent
        assert get_current_agent_name() == "parent"

    assert get_current_agent_state() is None


def test_parallel_agents_keep_their_own_state(agent_profile_factory, agent_state_factory):
    parent = agent_state_factory()
    children = [
        agent_state_factory(profile=agent_profile_factory(name=name, scope="main_agent"))
        for name in ["alice", "bob"]
    ]
    barrier = Barrier(2)

    def run(child):
        assert get_current_agent_state() is parent
        with bind_agent_context(child):
            barrier.wait(timeout=2)
            assert get_current_agent_state() is child
            assert get_current_agent_name() == child.agent_profile.name
        assert get_current_agent_state() is parent

    with bind_agent_context(parent), ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(copy_context().run, run, child) for child in children]
        for future in futures:
            future.result()
        assert get_current_agent_state() is parent

    assert get_current_agent_state() is None


@pytest.mark.parametrize("fails", [False, True])
def test_request_binds_its_agent_before_orchestration_and_restores_context(
    main_state_factory,
    monkeypatch,
    fails,
):
    main_state = main_state_factory(task="Find an answer", agent_profiles=[], llm=object())

    def orchestrate(state):
        active = get_current_agent_state()
        assert active is state.main_agent_state
        assert active.execution_context is state.execution_context
        assert active.inputs.task == state.task
        if fails:
            raise RuntimeError("Orchestration failed")
        return state.result

    monkeypatch.setattr("agent_runtime.orchestrator._ORCHESTRATOR.run", orchestrate)
    if fails:
        with pytest.raises(RuntimeError, match="Orchestration failed"):
            run_agent(main_state)
    else:
        assert run_agent(main_state) is main_state.result

    assert get_current_agent_state() is None
