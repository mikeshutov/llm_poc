from types import SimpleNamespace

from agents.discovery import load_agents as load_agents_module
from agents.document_writing.profile import DOCUMENT_WRITING_PROFILE
from agents.discovery.load_agents import load_agents
from agents.research.profile import RESEARCH_PROFILE
from agents.coding.profile import CODING_PROFILE
from agents.profile_management.profile import PROFILE_MANAGEMENT_PROFILE
from agent_runtime.models.agent_execution_context import AgentExecutionContext
from agent_runtime.models.main_state import MainState


def test_load_agents_adds_system_agents_before_user_agent_discovery(monkeypatch) -> None:
    monkeypatch.setattr(load_agents_module, "embed_text", lambda _: [1.0])

    class FakeRepository:
        def list_for_user(self, user_id):
            return []

    monkeypatch.setattr(load_agents_module, "get_agent_repo", lambda: FakeRepository())

    state = MainState.new(
        task="research a topic and write the findings",
        execution_context=AgentExecutionContext.new(),
        llm=object(),
        agent_profiles=[PROFILE_MANAGEMENT_PROFILE],
    )

    load_agents(state)

    assert state.agent_profiles == [
        PROFILE_MANAGEMENT_PROFILE,
        DOCUMENT_WRITING_PROFILE,
        RESEARCH_PROFILE,
        CODING_PROFILE,
    ]
    assert set(state.agent_states) == {
        "profile_management",
        "document_writing",
        "research",
        "coding",
    }
