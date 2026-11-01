from __future__ import annotations

from uuid import UUID

from agent_runtime.models.agent_prompt import EvidenceStep
from retrieval.evidence import EvidenceUrl, EvidenceView
from agent_runtime.models.synthesized_result import SynthesisResult, SynthesisResultBlock
from agent_runtime.synthesis.synthesis import _build_result_blocks


def test_synthesis_drops_citations_not_present_in_supplied_evidence() -> None:
    supplied_id = UUID("e5cf297f-8f55-55a7-b1b2-7fb389482919")
    result, used_ids = _build_result_blocks(
        SynthesisResult(
            result=[SynthesisResultBlock(
                content="answer",
                evidence_ids=[str(supplied_id), "not-supplied"],
            )]
        ),
        [EvidenceStep(
            type="search",
            evidence=[EvidenceView(id=supplied_id, title="Supported result")],
        )],
    )

    assert result[0].evidence_ids == [str(supplied_id)]
    assert used_ids == [str(supplied_id)]


def test_synthesis_does_not_fallback_to_all_evidence_after_invalid_citations() -> None:
    supplied_id = UUID("e5cf297f-8f55-55a7-b1b2-7fb389482919")
    result, used_ids = _build_result_blocks(
        SynthesisResult(
            result=[SynthesisResultBlock(
                content="answer",
                evidence_ids=["not-supplied"],
            )]
        ),
        [EvidenceStep(
            type="search",
            evidence=[EvidenceView(id=supplied_id, title="Supported result")],
        )],
    )

    assert result[0].evidence_ids == []
    assert used_ids == []


def test_synthesis_fallback_deduplicates_evidence_with_same_url() -> None:
    first_id = UUID("e5cf297f-8f55-55a7-b1b2-7fb389482919")
    second_id = UUID("c38c296c-3e94-56a7-86e1-dfe071c82fcc")
    result, used_ids = _build_result_blocks(
        SynthesisResult(result=[SynthesisResultBlock(content="answer")]),
        [EvidenceStep(
            type="search",
            evidence=[
                EvidenceView(
                    id=first_id,
                    title="First result",
                    urls=[EvidenceUrl(url="https://example.com/page")],
                ),
                EvidenceView(
                    id=second_id,
                    title="Duplicate result",
                    urls=[EvidenceUrl(url="https://example.com/page")],
                ),
            ],
        )],
    )

    assert result[0].evidence_ids == [str(first_id)]
    assert used_ids == [str(first_id)]


def test_synthesis_deduplicates_explicit_citations_with_same_url() -> None:
    first_id = UUID("e5cf297f-8f55-55a7-b1b2-7fb389482919")
    second_id = UUID("c38c296c-3e94-56a7-86e1-dfe071c82fcc")
    result, used_ids = _build_result_blocks(
        SynthesisResult(result=[SynthesisResultBlock(
            content="answer",
            evidence_ids=[str(first_id), str(second_id)],
        )]),
        [EvidenceStep(
            type="search",
            evidence=[
                EvidenceView(id=first_id, urls=[EvidenceUrl(url="https://example.com/page")]),
                EvidenceView(id=second_id, urls=[EvidenceUrl(url="https://example.com/page")]),
            ],
        )],
    )

    assert result[0].evidence_ids == [str(first_id)]
    assert used_ids == [str(first_id)]
