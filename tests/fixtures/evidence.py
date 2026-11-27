from copy import deepcopy

import pytest

from agent_runtime.models.agent_prompt import EvidenceStep
from retrieval.evidence import EvidenceView


@pytest.fixture
def evidence_factory():
    def create(**overrides):
        return EvidenceView(**deepcopy(overrides))

    return create


@pytest.fixture
def evidence_step_factory(evidence_factory):
    def create(*, evidence=None, **overrides):
        return EvidenceStep(
            **{"type": "search", **overrides},
            evidence=(
                [evidence_factory(summary="Toronto weather is sunny.")]
                if evidence is None else deepcopy(evidence)
            ),
        )

    return create
