from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import mock_open, patch
from uuid import UUID, uuid4


from integrations.brave.models import NewsResult
from llm.clients.llm_client import LlmClient
from personalization.profile.models import UserProfile
from common.logging import record_node_log
from agents.main.request import TOP_LEVEL_PROFILE
from agents.profile_management.profile import build_profile_management_profile
from agents.profile_management.profile import PROFILE_MANAGEMENT_PROFILE
from retrieval.evidence import ToolResult
from agent_runtime.models.evaluation_result import EVALUATION_STATUS_RETRYABLE, EVALUATION_STATUS_TERMINAL
from agent_runtime.evaluation.evaluator import run_evaluator
from agent_runtime.planning.planner import run_planner
from agent_runtime.context import bind_agent_context
from agent_runtime.synthesis.synthesis import run_synthesis
from retrieval.reranker_models import Candidate
from retrieval.reranker import rerank_candidates


def _latest_event_payload(
    repo,
    *,
    event_type: str,
    agent_name: str,
) -> dict:
    for event in reversed(repo.conversation_events):
        if event.get('event_type') != event_type:
            continue
        if event.get('node_name') != agent_name:
            continue
        return event['payload']
    raise AssertionError(f'No event found for type={event_type} agent={agent_name}')


def _agent_profiles_for(user_profile: UserProfile) -> list:
    return [
        build_profile_management_profile(user_profile),
        TOP_LEVEL_PROFILE,
    ]


def test_run_planner_records_main_and_profile_scopes(
    llm_factory,
    recording_conversation_repository_factory,
    agent_state_factory,
    execution_context_factory,
) -> None:
    repo = recording_conversation_repository_factory()
    main_state = agent_state_factory(
        task='Find me boots.',
        execution_context=execution_context_factory(conversation_id=str(uuid4()), roundtrip_id=uuid4()),
        llm=llm_factory(
            'native response',
            tool_calls=[{
                'id': 'call-1',
                'name': 'generic_web_search',
                'args': {
                    'parameters': {'query_text': 'boots'},
                    'hypothesis': 'Find matching boots.',
                    'metadata': {'source': 'test'},
                },
            }],
        ),
        profile=TOP_LEVEL_PROFILE,
    )
    profile_state = agent_state_factory(
        task='Remember I like pizza.',
        execution_context=execution_context_factory(conversation_id=str(uuid4()), roundtrip_id=uuid4()),
        llm=llm_factory('{"steps": [], "needs_replan": false}'),
        profile=PROFILE_MANAGEMENT_PROFILE,
    )
    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'common.logging.message_trace_logger.get_conversation_repo',
        return_value=repo,
    ), patch(
        'llm.chat_models.build_chat_model',
        return_value=main_state.llm,
    ):
        with bind_agent_context(main_state):
            run_planner(main_state)
            run_planner(profile_state)

    assert [call['agent'] for call in repo.llm_calls] == ['main_agent', 'profile_agent']
    assert all(call['stage'] == 'planner' for call in repo.llm_calls)
    assert 'input_object' not in repo.llm_calls[0]['metadata']
    assert repo.llm_calls[0]['metadata']['input']['message_count'] > 0
    assert 'output_object' not in repo.llm_calls[0]['metadata']
    assert repo.llm_calls[0]['metadata']['output']['bytes'] > 0
    assert main_state.node_states.planner.plan.steps[0].args == {'query_text': 'boots'}
    assert main_state.node_states.planner.plan.steps[0].hypothesis == 'Find matching boots.'
    assert main_state.node_states.planner.plan.steps[0].metadata == {}
    plan_log = next(
        event['payload']
        for event in repo.conversation_events
        if event['event_type'] == 'plan' and event['payload']['data']['steps']
    )
    assert plan_log['data']['steps'][0]['tool'] == 'generic_web_search'
    assert plan_log['data']['steps'][0]['hypothesis'] == 'Find matching boots.'
    assert plan_log['data']['steps'][0]['args'] == {'query_text': 'boots'}


def test_agent_log_persists_conversation_event_immediately(
    llm_factory,
    recording_conversation_repository_factory,
    agent_state_factory,
    execution_context_factory,
) -> None:
    repo = recording_conversation_repository_factory()
    conversation_id = uuid4()
    roundtrip_id = uuid4()
    state = agent_state_factory(
        task='Find me boots.',
        execution_context=execution_context_factory(conversation_id=str(conversation_id), roundtrip_id=roundtrip_id),
        llm=llm_factory('{"steps": [], "needs_replan": false}'),
        profile=TOP_LEVEL_PROFILE,
    )

    with patch('common.logging.message_trace_logger.get_conversation_repo', return_value=repo):
        with bind_agent_context(state):
            record_node_log(
                conversation_id=state.execution_context.conversation_id,
                roundtrip_id=state.execution_context.roundtrip_id,
                event_type='planner',
                source='main_agent',
                agent_name='main_agent',
                node_name='plan',
                iteration=1,
                payload={
                    'agent_name': 'main_agent',
                    'kind': 'planner',
                    'node_name': 'plan',
                    'iteration': 1,
                    'data': {'steps': []},
                },
            )

    assert len(repo.messages) == 0
    event = repo.conversation_events[-1]
    assert event['roundtrip_id'] == roundtrip_id
    assert event['event_type'] == 'planner'
    assert event['node_name'] == 'plan'
    assert event['payload']['data']['steps'] == []


def test_agent_log_preserves_metadata_next_to_structured_data(recording_conversation_repository_factory) -> None:
    repo = recording_conversation_repository_factory()
    roundtrip_id = uuid4()

    with patch('common.logging.message_trace_logger.get_conversation_repo', return_value=repo):
        record_node_log(
            roundtrip_id=roundtrip_id,
            event_type='plan',
            node_name='planner',
            payload={
                'kind': 'plan',
                'status': 'ready',
                'data': {'steps': [{'tool': 'search'}]},
            },
        )

    payload = repo.conversation_events[-1]['payload']['data']
    assert payload['kind'] == 'plan'
    assert payload['status'] == 'ready'
    assert payload['steps'] == [{'tool': 'search'}]


def test_run_planner_completes_when_no_native_tool_calls_are_returned(
    llm_factory,
    recording_conversation_repository_factory,
    agent_state_factory,
    execution_context_factory,
) -> None:
    repo = recording_conversation_repository_factory()
    state = agent_state_factory(
        task='Find me boots.',
        execution_context=execution_context_factory(conversation_id=str(uuid4()), roundtrip_id=uuid4()),
        llm=llm_factory('no tool calls'),
        profile=TOP_LEVEL_PROFILE,
    )
    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'common.logging.message_trace_logger.get_conversation_repo',
        return_value=repo,
    ), patch(
        'llm.chat_models.build_chat_model',
        return_value=state.llm,
    ):
        with bind_agent_context(state):
            run_planner(state)

    assert len(repo.llm_calls) == 1
    assert state.node_states.evaluator.evaluation_status == EVALUATION_STATUS_TERMINAL
    assert state.node_states.planner.plan is not None
    assert state.node_states.planner.plan.steps == []


def test_run_synthesis_records_llm_usage_after_tool_results(
    main_state_factory,
    llm_factory,
    recording_conversation_repository_factory,
    agent_state_factory,
    execution_context_factory,
) -> None:
    repo = recording_conversation_repository_factory()
    state = main_state_factory(
        task='Summarize this.',
        execution_context=execution_context_factory(conversation_id=str(uuid4()), roundtrip_id=uuid4()),
        llm=llm_factory(
            '{"result": [{"content": "done", "evidence_ids": []}], "next_question": "Do you want a deeper breakdown?", "roundtrip_summary": "summary"}',
            'gpt-5.6-terra',
        ),
        agent_profiles=_agent_profiles_for(UserProfile()),
    )

    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'common.logging.message_trace_logger.get_conversation_repo',
        return_value=repo,
    ), patch(
        'llm.chat_models.build_chat_model',
        return_value=state.llm,
    ):
        with bind_agent_context(agent_state_factory(
            profile=TOP_LEVEL_PROFILE,
            execution_context=state.execution_context,
        )):
            run_synthesis(state)

    assert len(repo.llm_calls) == 1
    assert repo.llm_calls[0]['stage'] == 'synthesis'
    assert repo.llm_calls[0]['model'] == 'gpt-5.6-terra'
    assert repo.llm_calls[0]['metadata']['input']['message_count'] > 0
    assert 'input_object' not in repo.llm_calls[0]['metadata']
    payload = _latest_event_payload(repo, event_type='synthesis', agent_name='synthesize')
    assert 'llm_usage' not in payload['data']


def test_reranker_does_not_record_llm_usage_when_it_runs(
    recording_conversation_repository_factory,
    agent_state_factory,
    execution_context_factory,
    static_reranker_factory,
) -> None:
    repo = recording_conversation_repository_factory()
    candidates = [
            Candidate(id=f'c{i}', fields={'name': f'Item {i}'})
        for i in range(12)
    ]
    backend = static_reranker_factory(results=[
        {"id": "c3", "score": 3.0},
        {"id": "c1", "score": 2.0},
        {"id": "c2", "score": 1.0},
    ])

    with patch('llm.usage.get_conversation_repo', return_value=repo):
        with bind_agent_context(agent_state_factory(
            profile=TOP_LEVEL_PROFILE,
            execution_context=execution_context_factory(conversation_id=str(uuid4()), roundtrip_id=uuid4()),
        )):
            rerank_candidates(candidates, goal='Find the best one', backend=backend)

    assert repo.llm_calls == []


def test_llm_client_records_tool_calling_and_image_caption_usage(
    openai_client_factory,
    recording_conversation_repository_factory,
    agent_state_factory,
    execution_context_factory,
) -> None:
    repo = recording_conversation_repository_factory()
    client = openai_client_factory()
    client.responses = [
        SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=25, total_tokens=125),
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[]))],
        ),
        SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=80, completion_tokens=15, total_tokens=95),
            choices=[SimpleNamespace(message=SimpleNamespace(content='caption'))],
        )]

    llm_client = LlmClient(client=client, default_model='gpt-5.6-luna')

    with patch('llm.usage.get_conversation_repo', return_value=repo), patch('builtins.open', mock_open(read_data=b'image-bytes')):
        with bind_agent_context(agent_state_factory(
            execution_context=execution_context_factory(conversation_id=str(uuid4()), roundtrip_id=uuid4()),
        )):
            llm_client.call_with_tools(system_prompt='sys', messages=[], tools=[])
            llm_client.generate_caption_from_image_file('fake.jpg')

    assert [call['stage'] for call in repo.llm_calls] == ['tool_calling', 'image_caption']
    assert all(isinstance(call['metadata']['latency_ms'], int) for call in repo.llm_calls)


def test_run_evaluator_records_llm_usage_and_preserves_missing_information_for_replanning(
    evidence_factory,
    llm_factory,
    recording_conversation_repository_factory,
    set_plan_results,
    agent_state_factory,
    plan_factory,
    plan_step_factory,
    execution_context_factory,
) -> None:
    repo = recording_conversation_repository_factory()
    state = agent_state_factory(
        task='Find current pricing for shortlisted products.',
        execution_context=execution_context_factory(conversation_id=str(uuid4()), roundtrip_id=uuid4()),
        llm=llm_factory(
            '{"status": "RETRYABLE", "relevant_evidence": ["25a4bcc1-2b18-5a36-940c-29c535bae654"], "missing_information": ["Need current pricing for the top two products", "Need shipping availability in Canada"]}',
            'gpt-5.6-terra',
        ),
        profile=TOP_LEVEL_PROFILE,
    )
    set_plan_results(
        state,
        plan=plan_factory(steps=[
            plan_step_factory(id='E1', plan='Search for the shortlisted products.', tool='generic_web_search', args={'query_text': 'shortlisted products'}),
        ]),
        results={
            'E1': ToolResult(
                result={'items': ['result']},
                evidence=[
                    evidence_factory(
                        id=UUID('25a4bcc1-2b18-5a36-940c-29c535bae654'),
                        summary='Current product search result.',
        data={
            "title": 'Shortlisted product',
        })
                ],
            )
        },
    )

    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'common.logging.message_trace_logger.get_conversation_repo',
        return_value=repo,
    ), patch(
        'llm.chat_models.build_chat_model',
        return_value=state.llm,
    ):
        with bind_agent_context(state):
            run_evaluator(state)

    assert len(repo.llm_calls) == 1
    assert repo.llm_calls[0]['agent'] == 'shared'
    assert repo.llm_calls[0]['stage'] == 'evaluator'
    assert repo.llm_calls[0]['model'] == 'gpt-5.6-terra'
    assert 'input_object' not in repo.llm_calls[0]['metadata']
    assert repo.llm_calls[0]['metadata']['input']['message_count'] > 0
    assert state.node_states.evaluator.evaluation_status == EVALUATION_STATUS_RETRYABLE
    assert state.inputs.task == 'Find current pricing for shortlisted products.'
    assert state.node_states.evaluator.missing_information == [
        'Need current pricing for the top two products',
        'Need shipping availability in Canada',
    ]
    payload = _latest_event_payload(repo, event_type='evaluator', agent_name='evaluator')
    assert 'llm_usage' not in payload['data']
    assert payload['data']['relevant_evidence'] == ['25a4bcc1-2b18-5a36-940c-29c535bae654']
    assert payload['data']['missing_information'] == [
        'Need current pricing for the top two products',
        'Need shipping availability in Canada']


def test_news_result_model_dump_excludes_thumbnail_url() -> None:
    result = NewsResult(
        title='Result',
        url='https://example.com/result',
        description='Description',
        thumbnail_url='https://example.com/thumb.jpg',
    )

    assert result.thumbnail_url == 'https://example.com/thumb.jpg'
    assert result.model_dump() == {
        'title': 'Result',
        'url': 'https://example.com/result',
        'description': 'Description',
        'age': None,
    }


def test_run_synthesis_filters_to_relevant_evidence_ids_when_available(
    evidence_factory,
    main_state_factory,
    llm_factory,
    recording_conversation_repository_factory,
    set_plan_results,
    agent_state_factory,
    plan_factory,
    plan_step_factory,
    execution_context_factory,
) -> None:
    repo = recording_conversation_repository_factory()
    state = main_state_factory(
        task='Summarize this.',
        execution_context=execution_context_factory(conversation_id=str(uuid4()), roundtrip_id=uuid4()),
        llm=llm_factory(
            '{"result": [{"content": "done", "evidence_ids": ["e5cf297f-8f55-55a7-b1b2-7fb389482919"]}], "next_question": "Do you want a deeper breakdown?", "roundtrip_summary": "summary"}',
            'gpt-5.6-terra',
        ),
        agent_profiles=_agent_profiles_for(UserProfile()),
    )
    main_agent_state = state.agent_states['request_orchestrator']
    set_plan_results(
        main_agent_state,
        plan=plan_factory(steps=[
            plan_step_factory(id='E1', plan='First step', tool='generic_web_search', args={}),
            plan_step_factory(id='E2', plan='Second step', tool='generic_web_search', args={}),
        ]),
        results={
            'E1': ToolResult(
                result={'value': 'a'},
                evidence=[
                    evidence_factory(
                        id=UUID('e5cf297f-8f55-55a7-b1b2-7fb389482919'),
                        summary='First item',
        data={
            "item_id": 'item-a',
            "title": 'Item A',
        })
                ],
            ),
            'E2': ToolResult(
                result={'value': 'b'},
                evidence=[
                    evidence_factory(
                        id=UUID('c8271821-2b18-5a36-940c-29c535bae654'),
                        summary='Second item',
        data={
            "item_id": 'item-b',
            "title": 'Item B',
        })
                ],
            ),
        },
    )
    main_agent_state.result = main_agent_state.result.copy(
        relevant_evidence_ids=[UUID('e5cf297f-8f55-55a7-b1b2-7fb389482919')]
    )

    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'common.logging.message_trace_logger.get_conversation_repo',
        return_value=repo,
    ), patch(
        'llm.chat_models.build_chat_model',
        return_value=state.llm,
    ):
        with bind_agent_context(agent_state_factory(
            profile=TOP_LEVEL_PROFILE,
            execution_context=state.execution_context,
        )):
            run_synthesis(state)

    prompt_text = str(state.llm.prompts[-1])
    payload = _latest_event_payload(repo, event_type='synthesis', agent_name='synthesize')
    assert payload['data']['relevant_evidence_ids'] == ['e5cf297f-8f55-55a7-b1b2-7fb389482919']


def test_run_synthesis_uses_all_fed_evidence_when_citations_are_missing(
    evidence_factory,
    main_state_factory,
    llm_factory,
    recording_conversation_repository_factory,
    set_plan_results,
    agent_state_factory,
    plan_factory,
    plan_step_factory,
    execution_context_factory,
) -> None:
    repo = recording_conversation_repository_factory()
    first_id = UUID('e5cf297f-8f55-55a7-b1b2-7fb389482919')
    second_id = UUID('c8271821-2d4c-51a1-bc00-1f4932d052d7')
    state = main_state_factory(
        task='Summarize this.',
        execution_context=execution_context_factory(conversation_id=str(uuid4()), roundtrip_id=uuid4()),
        llm=llm_factory(
            '{"result": [{"content": "done", "evidence_ids": []}], '
            '"next_question": "", "roundtrip_summary": "summary"}'
        ),
        agent_profiles=_agent_profiles_for(UserProfile()),
    )
    set_plan_results(
        state.agent_states['request_orchestrator'],
        plan=plan_factory(steps=[
            plan_step_factory(id='E1', plan='Search', tool='generic_web_search', args={}),
        ]),
        results={
            'E1': ToolResult(
                result={'value': 'results'},
                evidence=[
                    evidence_factory(id=first_id, summary='First item',
        data={
            "item_id": 'item-a',
            "title": 'Item A',
        }),
                    evidence_factory(id=second_id, summary='Second item',
        data={
            "item_id": 'item-b',
            "title": 'Item B',
        }),
                ],
            )
        },
    )

    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'common.logging.message_trace_logger.get_conversation_repo',
        return_value=repo,
    ), patch(
        'llm.chat_models.build_chat_model',
        return_value=state.llm,
    ):
        with bind_agent_context(agent_state_factory(
            profile=TOP_LEVEL_PROFILE,
            execution_context=state.execution_context,
        )):
            run_synthesis(state)

    assert state.result.result_blocks[0].evidence_ids == [str(first_id), str(second_id)]
    assert state.result.used_evidence_ids == [str(first_id), str(second_id)]


