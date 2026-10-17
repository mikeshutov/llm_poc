from __future__ import annotations

from uuid import uuid4

from request_orchestrator.agent_runner.models.agent_profile import AgentProfile
from request_orchestrator.models.agent_prompt import EvidenceStep
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.evidence import EvidenceView
from request_orchestrator.shared.evaluator.completion_gate import _build_premise, try_completion_evaluation
from serviceclients.completion_evaluator.models import CompletionEvaluatorResult


class FakeCompletionEvaluatorClient:
    def __init__(self, result: CompletionEvaluatorResult) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    def evaluate(self, premise: str, hypothesis: str) -> CompletionEvaluatorResult:
        self.calls.append((premise, hypothesis))
        return self.result


def _state() -> AgentState:
    return AgentState(
        agent_profile=AgentProfile(name="test", scope="test"),
        llm=object(),
    )


def _evidence_steps() -> list[EvidenceStep]:
    return [
        EvidenceStep(
            type="search",
            evidence=[EvidenceView(id=uuid4(), summary="Toronto weather is sunny.")],
        )
    ]


def test_completion_gate_returns_satisfied_for_high_confidence_entailment() -> None:
    client = FakeCompletionEvaluatorClient(
        CompletionEvaluatorResult(contradiction=0.02, neutral=0.03, entailment=0.95)
    )

    result = try_completion_evaluation(_state(), _evidence_steps(), client=client)

    assert result is True
    assert client.calls


def test_completion_gate_uses_compact_named_records_and_tool_hypothesis() -> None:
    evidence = EvidenceView(
        id=uuid4(),
        title="Chicken soup",
        summary="A warm soup recipe.",
        llm_metadata={"servings": 4, "diet": "vegetarian"},
    )
    steps = [
        EvidenceStep(
            type="meal_results",
            hypothesis="Find a vegetarian soup recipe.",
            evidence=[evidence],
        )
    ]
    client = FakeCompletionEvaluatorClient(
        CompletionEvaluatorResult(contradiction=0.02, neutral=0.03, entailment=0.95)
    )

    assert _build_premise(steps) == (
        "Chicken soup with summary, servings, diet"
    )
    assert try_completion_evaluation(_state(), steps, client=client) is True
    assert client.calls[0][1] == "Find a vegetarian soup recipe."


def test_completion_gate_includes_values_only_for_evaluation_fields() -> None:
    evidence = EvidenceView(
        id=uuid4(),
        title="Kenyan Beef Curry",
        summary="A spicy curry.",
        llm_metadata={"area": "Kenyan", "category": "Beef", "ingredients": ["beef", "onion"]},
    )
    steps = [
        EvidenceStep(
            type="meal_results",
            evaluation_fields=["ingredients"],
            evidence=[evidence],
        )
    ]

    assert _build_premise(steps) == (
        'Kenyan Beef Curry with summary, area, category, ingredients=["beef", "onion"]'
    )


def test_completion_gate_returns_none_for_uncertain_result() -> None:
    client = FakeCompletionEvaluatorClient(
        CompletionEvaluatorResult(contradiction=0.1, neutral=0.8, entailment=0.1)
    )

    result = try_completion_evaluation(_state(), _evidence_steps(), client=client)

    assert result is False


def test_completion_gate_returns_none_without_evidence() -> None:
    client = FakeCompletionEvaluatorClient(
        CompletionEvaluatorResult(contradiction=0.0, neutral=0.0, entailment=1.0)
    )

    result = try_completion_evaluation(_state(), [], client=client)

    assert result is False
    assert client.calls == []
