from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

if 'yfinance' not in sys.modules:
    sys.modules['yfinance'] = ModuleType('yfinance')

if 'pycountry' not in sys.modules:
    pycountry_module = ModuleType('pycountry')
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules['pycountry'] = pycountry_module

from request_orchestrator.models.agent_result import AgentResult
from rendering.messages.chat import _build_answer_payload


def test_build_answer_payload_includes_roundtrip_latency_ms() -> None:
    payload = _build_answer_payload(
        AgentResult(
            answer=['done'],
            roundtrip_summary='summary',
            roundtrip_latency_ms=1234,
        )
    )

    assert payload['roundtrip_latency_ms'] == 1234
