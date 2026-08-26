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

from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from request_orchestrator.models.orchestrator_result import OrchestratorResult
from request_orchestrator.models.synthesized_result import SynthesisResultBlock


def test_orchestrator_payload_derives_tool_summary_from_evidence() -> None:
    tool_call_id = uuid4()
    tool_results = [
        ToolResult(
            tool_call_id=tool_call_id,
            tool_name="get_current_weather",
            evidence=[
                EvidenceView(
                    tool_call_id=tool_call_id,
                    item_id="Toronto",
                    entity_type="weather",
                    title="Weather Result",
                    summary="21 C in Toronto.",
                )
            ],
        )
    ]

    fake_repo = SimpleNamespace(get_tool_results=lambda tool_call_ids: tool_results)

    with patch("tool.repository.tool_call_repository.ToolCallRepository", return_value=fake_repo):
        payload = OrchestratorResult(
            agent_result=AgentResult(tool_call_ids=[tool_call_id]),
            result_blocks=[
                SynthesisResultBlock(
                    content="It is 21 C in Toronto.",
                    evidence_ids=[],
                )
            ],
            answer=["It is 21 C in Toronto."],
        ).to_payload_model()

    assert payload.tool_summary.model_dump() == {
        "evidence_produced": [
            {
                "entity_type": "weather",
                "entity_id": "Toronto",
            }
        ]
    }
