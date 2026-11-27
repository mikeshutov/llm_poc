import pytest

from serviceclients.completion_evaluator.models import CompletionEvaluatorResult


class FakeCompletionEvaluatorClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def evaluate(self, premise, hypothesis):
        self.calls.append((premise, hypothesis))
        return self.result


@pytest.fixture
def completion_client_factory():
    def create(result=None, *, contradiction=0.0, neutral=0.0, entailment=1.0):
        if result is None:
            result = CompletionEvaluatorResult(
                contradiction=contradiction, neutral=neutral, entailment=entailment,
            )
        return FakeCompletionEvaluatorClient(result)

    return create
