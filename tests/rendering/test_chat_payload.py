from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from uuid import uuid4
from unittest.mock import patch

if 'yfinance' not in sys.modules:
    sys.modules['yfinance'] = ModuleType('yfinance')

if 'pycountry' not in sys.modules:
    pycountry_module = ModuleType('pycountry')
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules['pycountry'] = pycountry_module

from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.evidence import HydratedEvidence, ToolResult
from request_orchestrator.models.orchestrator_result import OrchestratorResult
from request_orchestrator.models.synthesized_result import SynthesisResultBlock
from conversation.models.conversation_models import ConversationEvent
from common.logging import fetch_agent_logs_for_roundtrip
from rendering.debug import _build_log_payload, _build_llm_call_payload, _ordered_agent_log_sections, _split_orchestrator_entries_for_agents
from rendering.messages.chat import _build_answer_payload
from rendering.rendering import fetch_llm_usage_for_roundtrip


def test_build_answer_payload_omits_roundtrip_latency_ms() -> None:
    payload = _build_answer_payload(
        OrchestratorResult(
            agent_result=AgentResult(),
            answer=['done'],
        )
    )

    assert 'roundtrip_latency_ms' not in payload
    assert 'response' not in payload


def test_build_answer_payload_preserves_result_block_evidence_ids() -> None:
    payload = _build_answer_payload(
        OrchestratorResult(
            agent_result=AgentResult(
                tool_results=[
                    ToolResult(
                        step_id="P1E1",
                        tool_name="generic_web_search",
                        hydrated_evidence=[
                            HydratedEvidence(
                                title="Example",
                                summary="Summary",
                                source="generic_web_search",
                            )
                        ],
                    )
                ],
            ),
            result_blocks=[
                SynthesisResultBlock(
                    content="Paragraph with evidence.",
                    evidence_ids=["P1E1R1"],
                )
            ],
            answer=["Paragraph with evidence."],
        )
    )

    assert payload["result"] == [
        {
            "content": "Paragraph with evidence.",
            "evidence_ids": ["P1E1R1"],
        }
    ]
    assert payload["used_evidence_ids"] == ["P1E1R1"]
    assert "P1E1R1" in payload["hydrated_evidence_by_id"]


def test_build_answer_payload_normalizes_result_block_evidence_ids_to_namespaced_keys() -> None:
    payload = _build_answer_payload(
        OrchestratorResult(
            agent_result=AgentResult(
                tool_results=[
                    ToolResult(
                        step_id="main_agent:P1E1",
                        tool_name="generic_web_search",
                        hydrated_evidence=[
                            HydratedEvidence(
                                title="Example",
                                summary="Summary",
                                source="generic_web_search",
                            )
                        ],
                    )
                ],
            ),
            result_blocks=[
                SynthesisResultBlock(
                    content="Paragraph with evidence.",
                    evidence_ids=["P1E1R1"],
                )
            ],
            answer=["Paragraph with evidence."],
        )
    )

    assert payload["result"] == [
        {
            "content": "Paragraph with evidence.",
            "evidence_ids": ["main_agent:P1E1R1"],
        }
    ]
    assert payload["used_evidence_ids"] == ["main_agent:P1E1R1"]
    assert "main_agent:P1E1R1" in payload["hydrated_evidence_by_id"]


def test_fetch_llm_usage_for_roundtrip_reads_llm_call_events_first() -> None:
    roundtrip_id = uuid4()

    class FakeRepo:
        def list_conversation_events_for_roundtrip(self, requested_roundtrip_id):
            assert requested_roundtrip_id == roundtrip_id
            return [
                ConversationEvent(
                    id=1,
                    conversation_id=uuid4(),
                    roundtrip_id=roundtrip_id,
                    event_type='llm_call',
                    source='planner',
                    agent_name='main_agent',
                    node_name='planner',
                    step_id='',
                    iteration=1,
                    payload={
                        'agent': 'shared',
                        'model_scope': 'shared',
                        'owner_agent_name': 'main_agent',
                        'stage': 'planner',
                        'callsite': 'planner',
                        'model': 'gpt-5.6-terra',
                        'input_tokens': 100,
                        'output_tokens': 20,
                        'total_tokens': 120,
                        'cached_input_tokens': 0,
                        'input_price_per_million_tokens': '1.25',
                        'output_price_per_million_tokens': '10',
                        'computed_input_cost': '0.000125',
                        'computed_output_cost': '0.0002',
                        'computed_total_cost': '0.000325',
                        'latency_ms': 250,
                        'metadata': {},
                    },
                    created_at='2026-08-14T12:00:00Z',
                )
            ]

        def list_llm_calls_for_roundtrip(self, requested_roundtrip_id):
            raise AssertionError('should not need llm_call table fallback when event payloads are complete')

    with patch('common.logging.conversation_event_view.get_conversation_repo', return_value=FakeRepo()):
        payload = fetch_llm_usage_for_roundtrip(str(roundtrip_id))

    assert payload is not None
    assert payload['retrieved_call_count'] == 1
    assert payload['summary']['input_tokens'] == 100
    assert payload['summary']['output_tokens'] == 20
    assert payload['summary']['total_tokens'] == 120
    assert payload['summary']['computed_total_cost'] == '0.000325'


def test_fetch_llm_usage_for_roundtrip_falls_back_to_llm_call_rows_when_event_trace_is_missing() -> None:
    roundtrip_id = uuid4()

    class FakeRepo:
        def list_conversation_events_for_roundtrip(self, requested_roundtrip_id):
            assert requested_roundtrip_id == roundtrip_id
            return [
                ConversationEvent(
                    id=1,
                    conversation_id=uuid4(),
                    roundtrip_id=roundtrip_id,
                    event_type='llm_call',
                    source='synthesis',
                    agent_name='request_orchestrator',
                    node_name='synthesis',
                    step_id='',
                    iteration=None,
                    payload={
                        'agent': 'main_agent',
                        'model_scope': 'main_agent',
                        'owner_agent_name': None,
                        'stage': 'synthesis',
                        'callsite': 'shared_synthesis.run_synthesis',
                        'model': 'gpt-5.6-luna-2026-03-17',
                        'input_tokens': 2059,
                        'output_tokens': 396,
                        'total_tokens': 2455,
                        'cached_input_tokens': 0,
                        'input_price_per_million_tokens': '0.75',
                        'output_price_per_million_tokens': '4.5',
                        'computed_input_cost': '0.00154425',
                        'computed_output_cost': '0.001782',
                        'computed_total_cost': '0.00332625',
                        'latency_ms': None,
                        'input_object': None,
                        'output_object': None,
                        'metadata': {},
                    },
                    created_at='2026-08-14T12:00:00Z',
                )
            ]

        def list_llm_calls_for_roundtrip(self, requested_roundtrip_id):
            assert requested_roundtrip_id == roundtrip_id
            return [
                {
                    'agent': 'main_agent',
                    'stage': 'synthesis',
                    'callsite': 'shared_synthesis.run_synthesis',
                    'model': 'gpt-5.6-luna-2026-03-17',
                    'input_tokens': 2059,
                    'output_tokens': 396,
                    'total_tokens': 2455,
                    'cached_input_tokens': 0,
                    'input_price_per_million_tokens': '0.75',
                    'output_price_per_million_tokens': '4.5',
                    'computed_input_cost': '0.00154425',
                    'computed_output_cost': '0.001782',
                    'computed_total_cost': '0.00332625',
                    'metadata': {
                        'latency_ms': 450,
                        'input_object': {'prompt': 'hello'},
                        'output_object': {'raw_content': '{}'},
                        'owner_agent_name': 'request_orchestrator',
                    },
                }
            ]

    with patch('common.logging.conversation_event_view.get_conversation_repo', return_value=FakeRepo()):
        payload = fetch_llm_usage_for_roundtrip(str(roundtrip_id))

    assert payload is not None
    assert payload['retrieved_call_count'] == 1
    assert payload['calls'][0]['input_object'] == {'prompt': 'hello'}
    assert payload['calls'][0]['output_object'] == {'raw_content': '{}'}
    assert payload['calls'][0]['latency_ms'] == 450
    assert payload['calls'][0]['owner_agent_name'] == 'request_orchestrator'


def test_build_log_payload_labels_llm_call_entries() -> None:
    title, payload = _build_log_payload(
        {
            'kind': 'llm_call',
            'model_scope': 'shared',
            'model': 'gpt-5.6-terra',
            'stage': 'planner',
            'callsite': 'planner',
            'input_tokens': 100,
            'output_tokens': 20,
            'total_tokens': 120,
            'cached_input_tokens': 0,
            'computed_input_cost': '0.000125',
            'computed_output_cost': '0.0002',
            'computed_total_cost': '0.000325',
            'latency_ms': 250,
            'metadata': {},
        }
    )

    assert title == 'LLM Call'
    assert payload['model'] == 'gpt-5.6-terra'
    assert payload['total_tokens'] == 120
    assert payload['computed_total_cost'] == '0.000325'


def test_build_request_analysis_log_payload_displays_all_agent_goals() -> None:
    title, payload = _build_log_payload(
        {
            'kind': 'request_analysis',
            'data': {
                'goals': [
                    {
                        'agent': 'main_agent',
                        'goal': 'Find hiking boots.',
                        'tool_categories': ['products'],
                    },
                    {
                        'agent': 'profile_management',
                        'goal': 'Record durable preferences when appropriate.',
                        'tool_categories': ['user_attributes'],
                    },
                ],
                'requested_user_attribute_types': ['products.likes'],
            },
        }
    )

    assert title == 'Request Analysis'
    assert payload['goals'] == [
        {
            'agent': 'main_agent',
            'goal': 'Find hiking boots.',
            'tool_categories': ['products'],
        },
        {
            'agent': 'profile_management',
            'goal': 'Record durable preferences when appropriate.',
            'tool_categories': ['user_attributes'],
        },
    ]
    assert payload['requested_user_attribute_types'] == ['products.likes']


def test_build_request_analysis_log_payload_supports_legacy_event_shape() -> None:
    _, payload = _build_log_payload(
        {
            'kind': 'request_analysis',
            'data': {
                'goal': 'Find hiking boots.',
                'applicable_tool_categories': ['products'],
            },
        }
    )

    assert payload['goals'] == [
        {
            'goal': 'Find hiking boots.',
            'tool_categories': ['products'],
        }
    ]


def test_build_llm_call_payload_reads_input_and_output_objects_from_metadata() -> None:
    payload = _build_llm_call_payload(
        {
            'kind': 'llm_call',
            'model_scope': 'shared',
            'model': 'gpt-5.6-terra',
            'stage': 'synthesis',
            'callsite': 'shared_synthesis.run_synthesis',
            'input_tokens': 100,
            'output_tokens': 20,
            'total_tokens': 120,
            'cached_input_tokens': 0,
            'computed_input_cost': '0.000125',
            'computed_output_cost': '0.0002',
            'computed_total_cost': '0.000325',
            'metadata': {
                'input_object': {'prompt': 'hello'},
                'output_object': {'raw_content': '{}'},
            },
        }
    )

    assert payload['input_object'] == {'prompt': 'hello'}
    assert payload['output_object'] == {'raw_content': '{}'}


def test_fetch_agent_logs_for_roundtrip_excludes_prompt_and_llm_call_events() -> None:
    roundtrip_id = uuid4()

    class FakeRepo:
        def list_conversation_events_for_roundtrip(self, requested_roundtrip_id):
            assert requested_roundtrip_id == roundtrip_id
            return [
                ConversationEvent(
                    id=1,
                    conversation_id=uuid4(),
                    roundtrip_id=roundtrip_id,
                    event_type='llm_call',
                    source='planner',
                    agent_name='main_agent',
                    node_name='planner',
                    step_id='',
                    iteration=1,
                    payload={'kind': 'llm_call'},
                    created_at='2026-08-14T12:00:00Z',
                ),
                ConversationEvent(
                    id=2,
                    conversation_id=uuid4(),
                    roundtrip_id=roundtrip_id,
                    event_type='prompt',
                    source='main_agent',
                    agent_name='main_agent',
                    node_name='planner',
                    step_id='',
                    iteration=1,
                    payload={'kind': 'prompt'},
                    created_at='2026-08-14T12:00:01Z',
                ),
                ConversationEvent(
                    id=3,
                    conversation_id=uuid4(),
                    roundtrip_id=roundtrip_id,
                    event_type='plan',
                    source='main_agent',
                    agent_name='main_agent',
                    node_name='planner',
                    step_id='',
                    iteration=1,
                    payload={'kind': 'plan', 'data': {'step_plans': ['Do the thing']}},
                    created_at='2026-08-14T12:00:02Z',
                ),
            ]

    with patch('common.logging.conversation_event_view.get_conversation_repo', return_value=FakeRepo()):
        logs = fetch_agent_logs_for_roundtrip(str(roundtrip_id))

    assert list(logs) == ['main_agent']
    assert len(logs['main_agent']) == 1
    assert logs['main_agent'][0]['kind'] == 'plan'


def test_build_plan_payload_omits_embedded_llm_usage() -> None:
    title, payload = _build_log_payload(
        {
            'kind': 'plan',
            'status': 'ready',
            'data': {
                'planner_reason': '',
                'step_plans': ['Do the thing'],
                'llm_usage': [{'total_tokens': 120}],
            },
        }
    )

    assert title == 'Plan Generated'
    assert payload['step_plans'] == ['Do the thing']
    assert 'llm_usage' not in payload


def test_ordered_agent_log_sections_follow_orchestrator_sequence() -> None:
    sections = _ordered_agent_log_sections(
        {
            'main_agent': [{'kind': 'plan'}],
            'request_orchestrator': [{'kind': 'request_analysis'}],
            'profile_management': [{'kind': 'plan'}],
            'some_other_agent': [{'kind': 'plan'}],
        }
    )

    assert [agent_name for agent_name, _ in sections] == [
        'request_orchestrator',
        'profile_management',
        'main_agent',
        'some_other_agent',
    ]


def test_split_orchestrator_entries_places_agents_before_synthesis() -> None:
    before, after = _split_orchestrator_entries_for_agents(
        [
            {'kind': 'request_analysis'},
            {'kind': 'profile_load'},
            {'kind': 'synthesis'},
        ]
    )

    assert [entry['kind'] for entry in before] == ['request_analysis', 'profile_load']
    assert [entry['kind'] for entry in after] == ['synthesis']
