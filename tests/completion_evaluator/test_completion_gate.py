from __future__ import annotations

from uuid import uuid4

from request_orchestrator.agent_runner.models.agent_profile import AgentProfile
from request_orchestrator.models.agent_prompt import EvidenceStep
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.evidence import EvidenceView
from request_orchestrator.shared.evaluator.completion_gate import try_completion_evaluation
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
