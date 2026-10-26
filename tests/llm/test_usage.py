from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest

from llm.usage import build_llm_usage_payload, extract_llm_usage, record_llm_call, serialize_llm_call_record
from common.data import bounded_json_value


def test_bounded_json_value_limits_large_nested_payloads() -> None:
    value = bounded_json_value({"items": [{"text": "x" * 100_000} for _ in range(1000)]}, max_bytes=2_000)

    assert len(str(value).encode("utf-8")) <= 2_000
    assert value.get("_truncated") is True


class FakeRepo:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_llm_usage(self, **kwargs):
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
        self.response_metadata = {'model_name': 'gpt-5.6-luna'}


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
            model_name='gpt-5.6-luna-2026-03-17',
            conversation_id=None,
            roundtrip_id=None,
            user_id=None,
            agent='main_agent',
            stage='planner',
            callsite='shared_planner.run_planner',
        )

    assert len(fake_repo.calls) == 1
    assert fake_repo.calls[0]['input_price_per_million_tokens'] == Decimal('0.20')
    assert fake_repo.calls[0]['output_price_per_million_tokens'] == Decimal('1.20')


def test_record_llm_call_computes_costs_and_persists() -> None:
    fake_repo = FakeRepo()
    conversation_id = uuid4()
    roundtrip_id = uuid4()

    with patch('llm.usage.get_conversation_repo', return_value=fake_repo):
        record_llm_call(
            raw_response=FakeLangChainResponse(),
            model_name='gpt-5.6-luna',
            conversation_id=str(conversation_id),
            roundtrip_id=str(roundtrip_id),
            user_id='anonymous',
            agent='main_agent',
            stage='planner',
            callsite='shared_planner.run_planner',
            metadata={'kind': 'test'},
        )

    assert len(fake_repo.calls) == 1
    stored = fake_repo.calls[0]
    assert stored['conversation_id'] == conversation_id
    assert stored['roundtrip_id'] == roundtrip_id
    assert stored['model'] == 'gpt-5.6-luna'
    assert stored['input_tokens'] == 120
    assert stored['output_tokens'] == 30
    assert stored['total_tokens'] == 150
    assert stored['input_price_per_million_tokens'] == Decimal('0.20')
    assert stored['output_price_per_million_tokens'] == Decimal('1.20')
    assert stored['computed_input_cost'] == Decimal('0.000024')
    assert stored['computed_output_cost'] == Decimal('0.000036')
    assert stored['computed_total_cost'] == Decimal('0.000060')
    assert stored['metadata'] == {'kind': 'test'}


def test_record_llm_call_charges_cached_luna_tokens_at_the_cached_rate() -> None:
    fake_repo = FakeRepo()
    response = FakeLangChainResponse()
    response.usage_metadata['input_tokens'] = 1_000
    response.usage_metadata['output_tokens'] = 100
    response.usage_metadata['total_tokens'] = 1_100
    response.usage_metadata['input_token_details'] = {'cache_read': 400}

    with patch('llm.usage.get_conversation_repo', return_value=fake_repo):
        record_llm_call(
            raw_response=response,
            model_name='gpt-5.6-luna',
            conversation_id=None,
            roundtrip_id=None,
            user_id=None,
            agent='main_agent',
            stage='planner',
            callsite='shared_planner.run_planner',
        )

    stored = fake_repo.calls[0]
    assert stored['cached_input_tokens'] == 400
    assert stored['computed_input_cost'] == Decimal('0.000128')
    assert stored['computed_output_cost'] == Decimal('0.00012')
    assert stored['computed_total_cost'] == Decimal('0.000248')


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
                user_id=None,
                agent='main_agent',
                stage='planner',
                callsite='shared_planner.run_planner',
            )


def test_record_llm_call_persists_compact_input_and_output_metadata() -> None:
    fake_repo = FakeRepo()

    with patch('llm.usage.get_conversation_repo', return_value=fake_repo):
        record_llm_call(
            raw_response=FakeLangChainResponse(),
            model_name='gpt-5.6-luna',
            conversation_id=None,
            roundtrip_id=None,
            user_id=None,
            agent='main_agent',
            stage='planner',
            callsite='shared_planner.run_planner',
            input_object={'prompt': 'hello', 'items': [1, 2]},
            output_object={'raw_content': '{}'},
            metadata={'kind': 'test'},
        )

    stored = fake_repo.calls[0]
    assert 'input_object' not in stored['metadata']
    assert 'output_object' not in stored['metadata']
    assert stored['metadata']['input']['bytes'] > 0
    assert stored['metadata']['output']['preview'] == '{}'
    assert stored['metadata']['kind'] == 'test'


def test_record_llm_call_persists_latency_ms_in_metadata() -> None:
    fake_repo = FakeRepo()

    with patch('llm.usage.get_conversation_repo', return_value=fake_repo):
        record_llm_call(
            raw_response=FakeLangChainResponse(),
            model_name='gpt-5.6-luna',
            conversation_id=None,
            roundtrip_id=None,
            user_id=None,
            agent='main_agent',
            stage='planner',
            callsite='shared_planner.run_planner',
            latency_ms=321,
        )

    stored = fake_repo.calls[0]
    assert stored['metadata']['latency_ms'] == 321


def test_serialize_llm_call_record_omits_full_input_and_output_objects() -> None:
    serialized = serialize_llm_call_record(
        {
            'agent': 'main_agent',
            'stage': 'planner',
            'callsite': 'shared_planner.run_planner',
            'model': 'gpt-5.6-luna',
            'input_tokens': 120,
            'output_tokens': 30,
            'total_tokens': 150,
            'cached_input_tokens': 0,
            'input_price_per_million_tokens': '1.00',
            'output_price_per_million_tokens': '6.00',
            'computed_input_cost': '0.00012',
            'computed_output_cost': '0.00018',
            'computed_total_cost': '0.00030',
            'metadata': {
                'latency_ms': 321,
                'input_object': {'prompt': 'hello'},
                'output_object': {'raw_content': '{}'},
                'kind': 'test',
            },
        }
    )

    assert 'input_object' not in serialized
    assert 'output_object' not in serialized
    assert serialized['latency_ms'] == 321
    assert serialized['metadata'] == {'kind': 'test'}


def test_build_llm_usage_payload_sums_latency_ms() -> None:
    payload = build_llm_usage_payload(
        [
            {
                'agent': 'main_agent',
                'stage': 'planner',
                'callsite': 'shared_planner.run_planner',
                'model': 'gpt-5.6-luna',
                'input_tokens': 120,
                'output_tokens': 30,
                'total_tokens': 150,
                'cached_input_tokens': 0,
                'input_price_per_million_tokens': '1.00',
                'output_price_per_million_tokens': '6.00',
                'computed_input_cost': '0.00012',
                'computed_output_cost': '0.00018',
                'computed_total_cost': '0.00030',
                'metadata': {'latency_ms': 100},
            },
            {
                'agent': 'main_agent',
                'stage': 'synthesis',
                'callsite': 'shared_synthesis.run_synthesis',
                'model': 'gpt-5.6-terra',
                'input_tokens': 80,
                'output_tokens': 20,
                'total_tokens': 100,
                'cached_input_tokens': 0,
                'input_price_per_million_tokens': '2.50',
                'output_price_per_million_tokens': '15.00',
                'computed_input_cost': '0.0002',
                'computed_output_cost': '0.0003',
                'computed_total_cost': '0.0005',
                'metadata': {'latency_ms': 250},
            },
        ]
    )

    assert payload['summary']['total_latency_ms'] == 350
    assert payload['calls'][0]['latency_ms'] == 100
    assert payload['calls'][1]['latency_ms'] == 250
