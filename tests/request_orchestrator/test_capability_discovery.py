from __future__ import annotations

import importlib

from request_orchestrator.models.main_state import MainState
from request_orchestrator.shared.discovery.capability_discovery import capability_discovery

discover_module = importlib.import_module("request_orchestrator.shared.discovery.capability_discovery")


def test_capability_discovery_ranks_tools_by_embedding(monkeypatch) -> None:
    def fake_embed(text: str) -> list[float]:
        return [1.0, 0.0] if any(term in text.lower() for term in ("food", "meal")) else [0.0, 1.0]

    monkeypatch.setattr(discover_module, "embed_text", fake_embed)
    state = MainState.new(task="food recommendations", agent_profiles=[])

    result = capability_discovery(state)

    assert result.tools
    assert any(tool.name == "search_meals" for tool in result.tools)


def test_capability_discovery_falls_back_without_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(discover_module, "embed_text", lambda _: (_ for _ in ()).throw(RuntimeError("offline")))
    state = MainState.new(task="Find books", agent_profiles=[])

    result = capability_discovery(state)



def test_capability_discovery_limits_ranked_results_to_five(monkeypatch) -> None:
    monkeypatch.setattr(discover_module, "embed_text", lambda _: [1.0, 0.0])
    state = MainState.new(task="Find anything", agent_profiles=[])

    result = capability_discovery(state)

    assert len(result.tools) <= 5
    assert len(result.agents) <= 5


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
