from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from capabilities.discovery.capability_discovery import capability_discovery
from capabilities.discovery.models import AvailableAgent, AvailableTool, CapabilityDiscoveryResult
from agent_runtime.models.agent_profile import AgentKind
from agents.models.agent import AgentType
from capabilities.registry.global_registry import GLOBAL_TOOL_REGISTRY

discover_module = importlib.import_module("capabilities.discovery.capability_discovery")


def test_runtime_discovery_registers_database_system_and_user_agents(monkeypatch, db_agent_factory, agent_state_factory) -> None:
    system_agent = db_agent_factory(
        agent_type=AgentType.SYSTEM, name="database_system",
        planner_instruction="Run the database system agent.",
    )
    user_agent = db_agent_factory(
        agent_type=AgentType.USER, user_id="alice", name="runtime_helper",
        planner_instruction="Run Alice's dynamically discovered agent.",
    )

    class Repository:
        def list_system_agents(self):
            return [system_agent]

        def list_relevant_for_user(self, user_id, *, query_embedding):
            assert user_id == "alice"
            return [user_agent]

    monkeypatch.setattr(discover_module, "get_agent_repo", Repository)
    monkeypatch.setattr(discover_module, "embed_text", lambda query: [1.0])
    monkeypatch.setattr(discover_module, "_discover_persisted_tools", lambda *args: [])
    monkeypatch.setattr(discover_module, "_rank_candidates", lambda query, embedding, candidates: [
        (1.0, candidate) for name, text, candidate in candidates
        if name in {system_agent.name, user_agent.name}
    ])
    state = agent_state_factory(task="Find a helper")
    state.execution_context.user_profile.user_id = "alice"

    discovered = capability_discovery(state)

    assert {agent.agent for agent in discovered.agents} == {system_agent.name, user_agent.name}
    assert state.available_agent_states[system_agent.name].agent_profile.kind == AgentKind.BUILTIN
    assert state.available_agent_states[user_agent.name].agent_profile.kind == AgentKind.USER_AGENT
    assert GLOBAL_TOOL_REGISTRY.get(system_agent.name).name == system_agent.name
    assert GLOBAL_TOOL_REGISTRY.get(user_agent.name, user_id="alice").name == user_agent.name


def test_discovery_projects_tools_and_agents_to_common_capability_shape() -> None:
    result = CapabilityDiscoveryResult(
        tools=[AvailableTool(name="web_search", category="web")],
        agents=[AvailableAgent(agent="research", description="Research topics")],
    )

    capabilities = result.as_available_tools()

    assert [capability.name for capability in capabilities] == ["web_search", "research"]
    assert capabilities[0].category == "web"
    assert capabilities[1].category == "agents"


def test_capability_discovery_uses_persisted_tool_embeddings(monkeypatch, agent_state_factory) -> None:
    def fake_embed(text: str) -> list[float]:
        return [1.0, 0.0] if any(term in text.lower() for term in ("food", "meal")) else [0.0, 1.0]

    monkeypatch.setattr(discover_module, "embed_text", fake_embed)
    class FakeToolRepository:
        def list_relevant(self, *, query_embedding, limit):
            assert query_embedding == [1.0, 0.0]
            assert limit == discover_module.RETRIEVAL_K
            return [SimpleNamespace(name="search_meals", description="Search meal recipes.")]

    monkeypatch.setattr(discover_module, "get_tool_registry_repo", lambda: FakeToolRepository())
    state = agent_state_factory(task="food recommendations")

    result = capability_discovery(state)

    assert result.tools
    assert any(tool.name == "search_meals" for tool in result.tools)


def test_capability_discovery_falls_back_without_embeddings(monkeypatch, agent_state_factory) -> None:
    monkeypatch.setattr(discover_module, "embed_text", lambda _: (_ for _ in ()).throw(RuntimeError("offline")))
    state = agent_state_factory(task="Find books")

    result = capability_discovery(state)

    assert result.tools
    assert len(result.tools) <= discover_module.TOP_K


def test_capability_discovery_limits_ranked_results_to_five(monkeypatch, agent_state_factory) -> None:
    monkeypatch.setattr(discover_module, "embed_text", lambda _: [1.0, 0.0])
    state = agent_state_factory(task="Find anything")

    result = capability_discovery(state)

    assert len(result.tools) <= 5
    assert len(result.agents) <= 5


def test_capability_discovery_excludes_static_tools_and_profile_agent(monkeypatch, agent_state_factory) -> None:
    monkeypatch.setattr(discover_module, "embed_text", lambda _: [1.0, 0.0])
    state = agent_state_factory(task="Find anything")

    result = capability_discovery(state)

    assert "discover_capabilities" not in {tool.name for tool in result.tools}
    assert "user_management" not in {tool.name for tool in result.tools}
    assert "profile_management" not in {agent.agent for agent in result.agents}


@pytest.mark.parametrize("persisted_matches", [True, False])
@pytest.mark.parametrize("allowed_tool_names", [{"math_calculate"}, set()])
def test_discovery_respects_profile_permissions_in_ranked_and_fallback_results(
    monkeypatch, persisted_matches, allowed_tool_names, agent_state_factory,
):
    monkeypatch.setattr(discover_module, "embed_text", lambda _: [1.0])
    rows = [
        SimpleNamespace(name=name, description=name, metadata={})
        for name in ["file_delete", "math_calculate"]
    ]
    monkeypatch.setattr(discover_module, "get_tool_registry_repo", lambda: SimpleNamespace(
        list_relevant=lambda **kwargs: rows if persisted_matches else [],
    ))

    def rank(query, candidates, semantic_rank):
        assert {name for name, _, _ in candidates} <= allowed_tool_names
        return [(1.0, value) for _, _, value in candidates]

    monkeypatch.setattr(discover_module, "_rerank_candidates", rank)
    state = agent_state_factory(task="Calculate or delete a file")

    result = capability_discovery(state, include_agents=False, allowed_tool_names=allowed_tool_names)

    assert {tool.name for tool in result.tools} == allowed_tool_names


def test_bm25_prefers_exact_lexical_match() -> None:
    candidates = [
        ("weather", "get weather forecast", object()),
        ("books", "search books and authors", object()),
    ]

    ranked = discover_module._bm25_rank("authors", candidates)

    assert ranked[0][2] == "books"


def test_rrf_combines_ranked_lists_with_weights() -> None:
    ranked = discover_module._rrf_rank(
        ["semantic-only", "shared", "lexical-only"],
        [
            (1.0, [(1.0, 0, "shared"), (0.5, 1, "semantic-only")]),
            (2.0, [(1.0, 0, "lexical-only"), (0.5, 1, "shared")]),
        ],
    )

    assert ranked[0] == "shared"
    assert set(ranked) == {"semantic-only", "shared", "lexical-only"}
