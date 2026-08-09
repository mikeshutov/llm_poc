from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest

from llm.usage import extract_llm_usage, record_llm_call


class FakeRepo:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create_llm_call(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs


class FakeLangChainResponse:
    def __init__(self) -> None:
        self.content = '{}'
        self.usage_metadata = {
            'input_tokens': 120,
            'output_tokens': 30,
            'total_tokens': 150,
        }
        self.response_metadata = {'model_name': 'gpt-5.4-mini'}


class FakeOpenAIResponse:
    def __init__(self) -> None:
        self.usage = SimpleNamespace(prompt_tokens=210, completion_tokens=45, total_tokens=255)


def test_extract_llm_usage_from_langchain_response() -> None:
    usage = extract_llm_usage(FakeLangChainResponse())

    assert usage is not None
    assert usage.input_tokens == 120
    assert usage.output_tokens == 30
    assert usage.total_tokens == 150


def test_extract_llm_usage_from_openai_response() -> None:
    usage = extract_llm_usage(FakeOpenAIResponse())

    assert usage is not None
    assert usage.input_tokens == 210
    assert usage.output_tokens == 45
    assert usage.total_tokens == 255


def test_record_llm_call_uses_base_pricing_for_snapshot_model_name() -> None:
    fake_repo = FakeRepo()

    with patch('llm.usage.get_conversation_repo', return_value=fake_repo):
        record_llm_call(
            raw_response=FakeLangChainResponse(),
            model_name='gpt-5.4-mini-2026-03-17',
            conversation_id=None,
            roundtrip_id=None,
            agent='main_agent',
            stage='request_analysis',
            callsite='request_analysis.analyze_request',
        )

    assert len(fake_repo.calls) == 1
    assert fake_repo.calls[0]['input_price_per_million_tokens'] == Decimal('0.75')
    assert fake_repo.calls[0]['output_price_per_million_tokens'] == Decimal('4.50')


def test_record_llm_call_computes_costs_and_persists() -> None:
    fake_repo = FakeRepo()
    conversation_id = uuid4()
    roundtrip_id = uuid4()

    with patch('llm.usage.get_conversation_repo', return_value=fake_repo):
        record_llm_call(
            raw_response=FakeLangChainResponse(),
            model_name='gpt-5.4-mini',
            conversation_id=str(conversation_id),
            roundtrip_id=str(roundtrip_id),
            agent='main_agent',
            stage='request_analysis',
            callsite='request_analysis.analyze_request',
            metadata={'kind': 'test'},
        )

    assert len(fake_repo.calls) == 1
    stored = fake_repo.calls[0]
    assert stored['conversation_id'] == conversation_id
    assert stored['roundtrip_id'] == roundtrip_id
    assert stored['model'] == 'gpt-5.4-mini'
    assert stored['input_tokens'] == 120
    assert stored['output_tokens'] == 30
    assert stored['total_tokens'] == 150
    assert stored['input_price_per_million_tokens'] == Decimal('0.75')
    assert stored['output_price_per_million_tokens'] == Decimal('4.50')
    assert stored['computed_input_cost'] == Decimal('0.00009')
    assert stored['computed_output_cost'] == Decimal('0.000135')
    assert stored['computed_total_cost'] == Decimal('0.000225')


def test_record_llm_call_raises_when_model_has_no_pricing() -> None:
    fake_repo = FakeRepo()

    with patch('llm.usage.get_conversation_repo', return_value=fake_repo):
        response = FakeLangChainResponse()
        response.response_metadata = {'model_name': 'not-a-real-model'}
        with pytest.raises(KeyError):
            record_llm_call(
                raw_response=response,
                model_name='not-a-real-model',
                conversation_id=None,
                roundtrip_id=None,
                agent='main_agent',
                stage='request_analysis',
                callsite='request_analysis.analyze_request',
            )
