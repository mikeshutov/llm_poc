from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = ModuleType("yfinance")

if "pycountry" not in sys.modules:
    pycountry_module = ModuleType("pycountry")
    pycountry_module.countries = SimpleNamespace(
        lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper())
    )
    sys.modules["pycountry"] = pycountry_module

from agent_runtime.models.agent_result import AgentResult
from retrieval.evidence import EvidenceView, ToolResult
from agent_runtime.models.orchestrator_result import OrchestratorResult
from agent_runtime.models.orchestrator_payload import OrchestratorPayloadToolSummary
from agent_runtime.models.relevant_evidence import RelevantEvidenceByTool
from agent_runtime.models.synthesized_result import SynthesisResultBlock
from capabilities.registry.constants import TOOL_NAME_LOOKUP_EVIDENCE


def test_persistence_caps_evidence_without_invalid_truncation_marker() -> None:
    tool_call_id = uuid4()
    evidence = [
        EvidenceView(source="research", data={"title": str(index)})
        for index in range(101)
    ]
    tool_result = ToolResult(
        tool_call_id=tool_call_id,
        tool_name="delegate",
        evidence=evidence,
    )
    result = OrchestratorResult(
        agent_result=AgentResult(tool_call_ids=[tool_call_id]),
        result_blocks=[SynthesisResultBlock(
            content="findings",
            evidence_ids=[evidence[-1].id],
        )],
    )

    with patch(
        "agent_runtime.models.orchestrator_result.message_repository.MessageToolRepository"
    ) as repository:
        repository.return_value.get_tool_results.return_value = [tool_result]
        payload, _ = result.to_persistence_models()

    assert len(payload.evidence_by_id) == 100
    assert evidence[-1].id in payload.evidence_by_id
    assert "_truncated_items" not in payload.evidence_by_id


def test_orchestrator_payload_derives_tool_summary_from_evidence() -> None:
    tool_call_id = uuid4()
    tool_results = [
        ToolResult(
            tool_call_id=tool_call_id,
            tool_name="get_current_weather",
            evidence=[
                EvidenceView(
        type="weather",
                    summary="21 C in Toronto.",
        data={
            "item_id": "Toronto",
            "title": "Weather Result",
        })
            ],
        )
    ]

    fake_repo = SimpleNamespace(get_tool_results=lambda tool_call_ids: tool_results)

    with patch("capabilities.registry.repository.message_repository.MessageToolRepository", return_value=fake_repo):
        payload, _ = OrchestratorResult(
            agent_result=AgentResult(tool_call_ids=[tool_call_id]),
            result_blocks=[
                SynthesisResultBlock(
                    content="It is 21 C in Toronto.",
                    evidence_ids=[],
                )
            ],
            answer=["It is 21 C in Toronto."],
        ).to_persistence_models()

    assert payload.tool_summary.model_dump(mode="json") == {
        "evidence_produced": {
            "get_current_weather": [str(tool_results[0].evidence[0].id)],
        }
    }


def test_tool_summary_builds_from_canonical_evidence() -> None:
    product_evidence = EvidenceView(source="find_products", type="product_results")
    ignored_evidence = EvidenceView()

    tool_summary = OrchestratorPayloadToolSummary.build(
        {
            str(product_evidence.id): product_evidence,
            "duplicate": product_evidence,
            str(ignored_evidence.id): ignored_evidence,
        }
    )

    assert tool_summary.model_dump(mode="json") == {
        "evidence_produced": {"find_products": [str(product_evidence.id)]}
    }


def test_orchestrator_result_groups_relevant_evidence_by_tool() -> None:
    tool_call_id = uuid4()
    evidence_id = uuid4()
    tool_results = [
        ToolResult(
            tool_call_id=tool_call_id,
            tool_name="search_products",
            evidence=[
                EvidenceView(
                    id=evidence_id,
        data={
            "title": "Product",
        })
            ],
        )
    ]
    fake_repo = SimpleNamespace(get_tool_results=lambda tool_call_ids: tool_results)

    with patch("capabilities.registry.repository.message_repository.MessageToolRepository", return_value=fake_repo):
        _, relevant_evidence = OrchestratorResult(
            agent_result=AgentResult(
                tool_call_ids=[tool_call_id],
                relevant_evidence_ids=[evidence_id],
            )
        ).to_persistence_models()

    assert relevant_evidence.model_dump(mode="json") == {
        "search_products": [str(evidence_id)]
    }


def test_relevant_evidence_by_tool_builds_from_canonical_evidence() -> None:
    product_evidence = EvidenceView(source="find_products", type="product_results")
    ignored_evidence = EvidenceView()

    relevant_evidence = RelevantEvidenceByTool.build(
        [product_evidence.id, product_evidence.id, ignored_evidence.id, uuid4()],
        {
            str(product_evidence.id): product_evidence,
            str(ignored_evidence.id): ignored_evidence,
        },
    )

    assert relevant_evidence.model_dump(mode="json") == {
        "find_products": [str(product_evidence.id)]
    }


def test_excluded_tool_evidence_is_not_persisted() -> None:
    lookup_tool_call_id = uuid4()
    product_tool_call_id = uuid4()
    lookup_evidence = EvidenceView(
        data={
            "title": "Historical product",
        })
    product_evidence = EvidenceView(
        data={
            "title": "Current product",
        })
    tool_results = [
        ToolResult(
            tool_call_id=lookup_tool_call_id,
            tool_name=TOOL_NAME_LOOKUP_EVIDENCE,
            evidence=[lookup_evidence],
        ),
        ToolResult(
            tool_call_id=product_tool_call_id,
            tool_name="find_products",
            evidence=[product_evidence],
        ),
    ]
    fake_repo = SimpleNamespace(get_tool_results=lambda tool_call_ids: tool_results)

    with patch("capabilities.registry.repository.message_repository.MessageToolRepository", return_value=fake_repo):
        payload, relevant_evidence = OrchestratorResult(
            agent_result=AgentResult(
                tool_call_ids=[lookup_tool_call_id, product_tool_call_id],
                relevant_evidence_ids=[lookup_evidence.id, product_evidence.id],
            ),
            result_blocks=[
                SynthesisResultBlock(
                    content="Current product is available.",
                    evidence_ids=[str(lookup_evidence.id), str(product_evidence.id)],
                )
            ],
        ).to_persistence_models()

    assert list(payload.evidence_by_id) == [str(product_evidence.id)]
    assert payload.result[0].evidence_ids == [str(product_evidence.id)]
    assert payload.used_evidence_ids == [str(product_evidence.id)]
    assert payload.tool_summary.model_dump(mode="json") == {
        "evidence_produced": {"find_products": [str(product_evidence.id)]}
    }
    assert relevant_evidence.model_dump(mode="json") == {
        "find_products": [str(product_evidence.id)]
    }




