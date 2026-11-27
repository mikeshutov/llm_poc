from __future__ import annotations

from uuid import uuid4

from agent_runtime.models.agent_prompt import EvidenceStep
from agent_runtime.evaluation.completion import _build_premise, try_completion_evaluation
from serviceclients.completion_evaluator.models import CompletionEvaluatorResult


def test_completion_gate_returns_satisfied_for_high_confidence_entailment(
    evidence_step_factory,
    agent_state_factory,
    completion_client_factory,
) -> None:
    client = completion_client_factory(
        CompletionEvaluatorResult(contradiction=0.02, neutral=0.03, entailment=0.95)
    )

    result = try_completion_evaluation(agent_state_factory(), [evidence_step_factory()], client=client)

    assert result is True
    assert client.calls


def test_completion_gate_uses_compact_named_records_and_tool_hypothesis(
    evidence_factory,
    agent_state_factory,
    completion_client_factory,
) -> None:
    evidence = evidence_factory(
        id=uuid4(),
        summary="A warm soup recipe.",
        metadata={"servings": 4, "diet": "vegetarian"},
        data={
            "title": "Chicken soup",
        })
    steps = [
        EvidenceStep(
            type="meal_results",
            hypothesis="Find a vegetarian soup recipe.",
            evidence=[evidence],
        )
    ]
    client = completion_client_factory(
        CompletionEvaluatorResult(contradiction=0.02, neutral=0.03, entailment=0.95)
    )

    assert _build_premise(steps) == (
        "Chicken soup with servings, diet"
    )
    assert try_completion_evaluation(agent_state_factory(), steps, client=client) is True
    assert client.calls[0][1] == "Find a vegetarian soup recipe."


def test_completion_gate_includes_values_only_for_evaluation_fields(evidence_factory) -> None:
    evidence = evidence_factory(
        id=uuid4(),
        summary="A spicy curry.",
        metadata={"area": "Kenyan", "category": "Beef", "ingredients": ["beef", "onion"]},
        data={
            "title": "Kenyan Beef Curry",
        })
    steps = [
        EvidenceStep(
            type="meal_results",
            evaluation_fields=["ingredients"],
            evidence=[evidence],
        )
    ]

    assert _build_premise(steps) == (
        'Kenyan Beef Curry with area, category, ingredients=["beef", "onion"]'
    )


def test_completion_gate_returns_none_for_uncertain_result(
    evidence_step_factory,
    agent_state_factory,
    completion_client_factory,
) -> None:
    client = completion_client_factory(
        CompletionEvaluatorResult(contradiction=0.1, neutral=0.8, entailment=0.1)
    )

    result = try_completion_evaluation(agent_state_factory(), [evidence_step_factory()], client=client)

    assert result is False


def test_completion_gate_returns_none_without_evidence(
    agent_state_factory,
    completion_client_factory,
) -> None:
    client = completion_client_factory(
        CompletionEvaluatorResult(contradiction=0.0, neutral=0.0, entailment=1.0)
    )

    result = try_completion_evaluation(agent_state_factory(), [], client=client)

    assert result is False
    assert client.calls == []


