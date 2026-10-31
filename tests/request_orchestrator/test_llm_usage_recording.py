from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import mock_open, patch
from uuid import UUID, uuid4

if 'yfinance' not in sys.modules:
    sys.modules['yfinance'] = ModuleType('yfinance')

if 'pycountry' not in sys.modules:
    pycountry_module = ModuleType('pycountry')
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules['pycountry'] = pycountry_module

from conversation.models.conversation_models import ConversationContext
from integrations.brave.models import NewsResult
from llm.clients.llm_client import LlmClient
from personalization.profile.models import UserProfile
from common.logging import record_node_log
from request_orchestrator.strategies.main_request_strategy import TOP_LEVEL_PROFILE
from request_orchestrator.agents.profile_management.profile import build_profile_management_profile
from request_orchestrator.agents.profile_management.profile import PROFILE_MANAGEMENT_PROFILE
from request_orchestrator.models.agent_execution_context import AgentExecutionContext
from request_orchestrator.models.agent_prompt import PromptSectionKeys
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.orchestrator_result import OrchestratorResult
from request_orchestrator.models.main_state import MainState
from request_orchestrator.models.evidence import EvidenceView, ToolResult
from request_orchestrator.models.evaluation_result import EVALUATION_STATUS_RETRYABLE, EVALUATION_STATUS_TERMINAL
from request_orchestrator.models.plan import Plan
from request_orchestrator.shared.evaluator.evaluator import run_evaluator
from request_orchestrator.shared.planner.planner import run_planner
from request_orchestrator.shared.runtime_context import bind_agent_context, bind_runtime_context
from request_orchestrator.shared.synthesis.synthesis import run_synthesis
from reranker.models import Candidate
from reranker.service import rerank_candidates


class RecordingRepo:
    def __init__(self) -> None:
        self.llm_calls: list[dict] = []
        self.conversation_events: list[dict] = []
        self.messages: list[dict] = []

    def record_llm_usage(self, **kwargs):
        self.llm_calls.append(kwargs)
        return kwargs

    def append_node_log(self, scope_type, scope_id, **kwargs):
        payload = kwargs.get("payload") or {}
        node_name = kwargs.get("node_name") or payload.get("agent_name") or payload.get("source", "")
        self.conversation_events.append({
            "scope_type": scope_type,
            "scope_id": scope_id,
            "roundtrip_id": scope_id if scope_type == "roundtrip" else None,
            "event_type": kwargs.get("event_type"),
            "source": node_name,
            "agent_name": node_name,
            "node_name": node_name,
            "payload": payload,
        })
        return type("RecordedLog", (), {"id": uuid4()})()

    def append_message(self, roundtrip_id, **kwargs):
        message = {"roundtrip_id": roundtrip_id, **kwargs}
        self.messages.append(message)
        payload = kwargs.get("payload") or {}
        if payload.get("kind"):
            self.conversation_events.append({
                "roundtrip_id": roundtrip_id,
                "event_type": payload["kind"],
                "source": payload.get("source", ""),
                "agent_name": payload.get("agent_name", ""),
                "node_name": payload.get("node_name", ""),
                "iteration": payload.get("iteration"),
                "payload": payload,
            })
        return type("RecordedMessage", (), {"id": uuid4()})()

    def create_roundtrip_prompt(self, *args, **kwargs):
        return None


def _latest_event_payload(
    repo: RecordingRepo,
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


class FakeLangChainResponse:
    def __init__(self, content: str, model_name: str = 'gpt-5.6-luna', tool_calls: list[dict] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage_metadata = {
            'input_tokens': 100,
            'output_tokens': 20,
            'total_tokens': 120}
        self.response_metadata = {'model_name': model_name}


class FakeInvokeLLM:
    def __init__(self, content: str, model_name: str = 'gpt-5.6-luna', tool_calls: list[dict] | None = None) -> None:
        self.content = content
        self.model_name = model_name
        self.tool_calls = tool_calls or []

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, prompt: str):
        return FakeLangChainResponse(self.content, self.model_name, self.tool_calls)


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.responses: list[object] = []

    def create(self, **kwargs):
        return self.responses.pop(0)


def _agent_profiles_for(user_profile: UserProfile) -> list:
    return [
        build_profile_management_profile(user_profile),
        TOP_LEVEL_PROFILE,
    ]


def _bind_args_for_agent_state(state: AgentState) -> dict[str, object]:
    execution_context = state.execution_context
    return {
        "conversation_id": execution_context.conversation_id or str(uuid4()),
        "conversation_model_config": execution_context.model_config,
        "roundtrip_id": str(execution_context.roundtrip_id) if execution_context.roundtrip_id else str(uuid4()),
        "user_id": execution_context.user_profile.user_id,
    }


def _bind_args_for_main_state(state: MainState) -> dict[str, object]:
    execution_context = state.execution_context
    return {
        "conversation_id": execution_context.conversation_id or str(uuid4()),
        "conversation_model_config": execution_context.model_config,
        "roundtrip_id": str(execution_context.roundtrip_id) if execution_context.roundtrip_id else str(uuid4()),
        "user_id": execution_context.user_profile.user_id,
    }


def _set_agent_tool_results(
    state: AgentState,
    *,
    plan: Plan | None = None,
    results: dict[str, object] | None = None,
    plan_count: int = 1,
) -> None:
    if plan is not None:
        state.node_states.planner.plan = plan.model_copy(deep=True)
        state.node_states.planner.plan_count = plan_count
    step_by_local_id = {step.id: step for step in (plan.steps if plan is not None else [])}
    normalized_tool_results: list[ToolResult] = []
    for local_step_id, value in (results or {}).items():
        step = step_by_local_id[local_step_id]
        if isinstance(value, ToolResult):
            normalized_tool_results.append(
                value.model_copy(
                    update={
                        "tool_call_id": value.tool_call_id or uuid4(),
                        "plan_step_id": value.plan_step_id or step.db_id,
                        "tool_name": value.tool_name or step.tool,
                    }
                )
            )
            continue
        normalized_tool_results.append(
            ToolResult(
                tool_call_id=uuid4(),
                plan_step_id=step.db_id,
                tool_name=step.tool,
                result=value,
            )
        )
    state.gather_tool_results = lambda: normalized_tool_results


def test_run_planner_records_main_and_profile_scopes() -> None:
    repo = RecordingRepo()
    main_state = AgentState.new(
        task='Find me boots.',
        execution_context=AgentExecutionContext.new(
            conversation_context=ConversationContext(),
            user_profile=UserProfile(),
            conversation_id=str(uuid4()),
        ),
        llm=FakeInvokeLLM(
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
        agent_profile=TOP_LEVEL_PROFILE,
    )
    profile_state = AgentState.new(
        task='Remember I like pizza.',
        execution_context=AgentExecutionContext.new(
            conversation_context=ConversationContext(),
            user_profile=UserProfile(),
            conversation_id=str(uuid4()),
        ),
        llm=FakeInvokeLLM('{"steps": [], "needs_replan": false}'),
        agent_profile=PROFILE_MANAGEMENT_PROFILE,
    )
    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'common.logging.message_trace_logger.get_conversation_repo',
        return_value=repo,
    ), patch(
        'llm.chat_models.build_chat_model',
        return_value=main_state.llm,
    ):
        with bind_runtime_context(**_bind_args_for_agent_state(main_state)):
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


def test_agent_log_persists_conversation_event_immediately() -> None:
    repo = RecordingRepo()
    conversation_id = uuid4()
    roundtrip_id = uuid4()
    state = AgentState.new(
        task='Find me boots.',
        execution_context=AgentExecutionContext.new(
            conversation_context=ConversationContext(),
            user_profile=UserProfile(),
            conversation_id=str(conversation_id),
        ),
        llm=FakeInvokeLLM('{"steps": [], "needs_replan": false}'),
        agent_profile=TOP_LEVEL_PROFILE,
    )

    with patch('common.logging.message_trace_logger.get_conversation_repo', return_value=repo):
        with bind_runtime_context(
            conversation_id=str(conversation_id),
            conversation_model_config=None,
            roundtrip_id=str(roundtrip_id),
        ):
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


def test_agent_log_preserves_metadata_next_to_structured_data() -> None:
    repo = RecordingRepo()
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


def test_run_planner_completes_when_no_native_tool_calls_are_returned() -> None:
    repo = RecordingRepo()
    state = AgentState.new(
        task='Find me boots.',
        execution_context=AgentExecutionContext.new(
            conversation_context=ConversationContext(),
            user_profile=UserProfile(),
            conversation_id=str(uuid4()),
        ),
        llm=FakeInvokeLLM('no tool calls'),
        agent_profile=TOP_LEVEL_PROFILE,
    )
    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'common.logging.message_trace_logger.get_conversation_repo',
        return_value=repo,
    ), patch(
        'llm.chat_models.build_chat_model',
        return_value=state.llm,
    ):
        with bind_runtime_context(**_bind_args_for_agent_state(state)):
            run_planner(state)

    assert len(repo.llm_calls) == 1
    assert state.node_states.evaluator.evaluation_status == EVALUATION_STATUS_TERMINAL
    assert state.node_states.planner.plan is not None
    assert state.node_states.planner.plan.steps == []


def test_run_synthesis_records_llm_usage_after_tool_results() -> None:
    repo = RecordingRepo()
    state = MainState.new(
        task='Summarize this.',
        execution_context=AgentExecutionContext.new(
            conversation_context=ConversationContext(),
            user_profile=UserProfile(),
            conversation_id=str(uuid4()),
        ),
        llm=FakeInvokeLLM('{"result": [{"content": "done", "evidence_ids": []}], "next_question": "Do you want a deeper breakdown?", "roundtrip_summary": "summary"}', 'gpt-5.6-terra'),
        agent_profiles=_agent_profiles_for(UserProfile()),
    )

    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'common.logging.message_trace_logger.get_conversation_repo',
        return_value=repo,
    ), patch(
        'llm.chat_models.build_chat_model',
        return_value=state.llm,
    ):
        with bind_runtime_context(**_bind_args_for_main_state(state)):
            run_synthesis(state)

    assert len(repo.llm_calls) == 1
    assert repo.llm_calls[0]['stage'] == 'synthesis'
    assert repo.llm_calls[0]['model'] == 'gpt-5.6-terra'
    assert repo.llm_calls[0]['metadata']['input']['message_count'] > 0
    assert 'input_object' not in repo.llm_calls[0]['metadata']
    payload = _latest_event_payload(repo, event_type='synthesis', agent_name='synthesize')
    assert 'llm_usage' not in payload['data']


def test_reranker_does_not_record_llm_usage_when_it_runs() -> None:
    repo = RecordingRepo()
    candidates = [
            Candidate(id=f'c{i}', fields={'name': f'Item {i}'})
        for i in range(12)
    ]
    class FakeReranker:
        def rerank(self, query, candidates):
            return [
                {'id': 'c3', 'score': 3.0},
                {'id': 'c1', 'score': 2.0},
                {'id': 'c2', 'score': 1.0},
            ]

    with patch('llm.usage.get_conversation_repo', return_value=repo):
        with bind_runtime_context(
            conversation_id=str(uuid4()),
            conversation_model_config=None,
            roundtrip_id=str(uuid4()),
        ):
            with bind_agent_context(agent_name='main_agent'):
                rerank_candidates(candidates, goal='Find the best one', backend=FakeReranker())

    assert repo.llm_calls == []


def test_llm_client_records_tool_calling_and_image_caption_usage() -> None:
    repo = RecordingRepo()
    client = FakeOpenAIClient()
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
        with bind_runtime_context(conversation_id=str(uuid4()), conversation_model_config=None, roundtrip_id=str(uuid4())):
            llm_client.call_with_tools(system_prompt='sys', messages=[], tools=[])
            llm_client.generate_caption_from_image_file('fake.jpg')

    assert [call['stage'] for call in repo.llm_calls] == ['tool_calling', 'image_caption']
    assert all(isinstance(call['metadata']['latency_ms'], int) for call in repo.llm_calls)


def test_run_evaluator_records_llm_usage_and_preserves_missing_information_for_replanning() -> None:
    repo = RecordingRepo()
    state = AgentState.new(
        task='Find current pricing for shortlisted products.',
        execution_context=AgentExecutionContext.new(
            conversation_context=ConversationContext(),
            user_profile=UserProfile(),
            conversation_id=str(uuid4()),
        ),
        llm=FakeInvokeLLM('{"status": "RETRYABLE", "relevant_evidence": ["25a4bcc1-2b18-5a36-940c-29c535bae654"], "missing_information": ["Need current pricing for the top two products", "Need shipping availability in Canada"]}', 'gpt-5.6-terra'),
        agent_profile=TOP_LEVEL_PROFILE,
    )
    _set_agent_tool_results(
        state,
        plan=Plan.model_validate({
            'steps': [
                {
                    'id': 'E1',
                    'plan': 'Search for the shortlisted products.',
                    'tool': 'generic_web_search',
                    'args': {'query_text': 'shortlisted products'}}
            ]
        }),
        results={
            'E1': ToolResult(
                result={'items': ['result']},
                evidence=[
                    EvidenceView(
                        id=UUID('25a4bcc1-2b18-5a36-940c-29c535bae654'),
                        title='Shortlisted product',
                        summary='Current product search result.',
                    )
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
        with bind_runtime_context(**_bind_args_for_agent_state(state)):
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


def test_run_synthesis_filters_to_relevant_evidence_ids_when_available() -> None:
    repo = RecordingRepo()
    captured_prompt: dict[str, object] = {}

    class CapturingLLM(FakeInvokeLLM):
        def invoke(self, prompt: str):
            captured_prompt['text'] = prompt
            return super().invoke(prompt)

    state = MainState.new(
        task='Summarize this.',
        execution_context=AgentExecutionContext.new(
            conversation_context=ConversationContext(),
            user_profile=UserProfile(),
            conversation_id=str(uuid4()),
        ),
        llm=CapturingLLM('{"result": [{"content": "done", "evidence_ids": ["e5cf297f-8f55-55a7-b1b2-7fb389482919"]}], "next_question": "Do you want a deeper breakdown?", "roundtrip_summary": "summary"}', 'gpt-5.6-terra'),
        agent_profiles=_agent_profiles_for(UserProfile()),
    )
    main_agent_state = state.agent_states['request_orchestrator']
    _set_agent_tool_results(
        main_agent_state,
        plan=Plan.model_validate({
            'steps': [
                {'id': 'E1', 'plan': 'First step', 'tool': 'generic_web_search', 'args': {}},
                {'id': 'E2', 'plan': 'Second step', 'tool': 'generic_web_search', 'args': {}}]
        }),
        results={
            'E1': ToolResult(
                result={'value': 'a'},
                evidence=[
                    EvidenceView(
                        id=UUID('e5cf297f-8f55-55a7-b1b2-7fb389482919'),
                        item_id='item-a',
                        title='Item A',
                        summary='First item',
                    )
                ],
            ),
            'E2': ToolResult(
                result={'value': 'b'},
                evidence=[
                    EvidenceView(
                        id=UUID('c8271821-2b18-5a36-940c-29c535bae654'),
                        item_id='item-b',
                        title='Item B',
                        summary='Second item',
                    )
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
        with bind_runtime_context(**_bind_args_for_main_state(state)):
            run_synthesis(state)

    prompt_text = str(captured_prompt['text'])
    assert '"evidence_id": "e5cf297f-8f55-55a7-b1b2-7fb389482919"' in prompt_text
    assert '"evidence_id": "c8271821-2d4c-51a1-bc00-1f4932d052d7"' not in prompt_text
    payload = _latest_event_payload(repo, event_type='synthesis', agent_name='synthesize')
    assert payload['data']['relevant_evidence_ids'] == ['e5cf297f-8f55-55a7-b1b2-7fb389482919']


def test_run_synthesis_uses_all_fed_evidence_when_citations_are_missing() -> None:
    repo = RecordingRepo()
    first_id = UUID('e5cf297f-8f55-55a7-b1b2-7fb389482919')
    second_id = UUID('c8271821-2d4c-51a1-bc00-1f4932d052d7')
    state = MainState.new(
        task='Summarize this.',
        execution_context=AgentExecutionContext.new(
            conversation_context=ConversationContext(),
            user_profile=UserProfile(),
            conversation_id=str(uuid4()),
        ),
        llm=FakeInvokeLLM(
            '{"result": [{"content": "done", "evidence_ids": []}], '
            '"next_question": "", "roundtrip_summary": "summary"}'
        ),
        agent_profiles=_agent_profiles_for(UserProfile()),
    )
    _set_agent_tool_results(
        state.agent_states['request_orchestrator'],
        plan=Plan.model_validate({
            'steps': [{'id': 'E1', 'plan': 'Search', 'tool': 'generic_web_search', 'args': {}}]
        }),
        results={
            'E1': ToolResult(
                result={'value': 'results'},
                evidence=[
                    EvidenceView(id=first_id, item_id='item-a', title='Item A', summary='First item'),
                    EvidenceView(id=second_id, item_id='item-b', title='Item B', summary='Second item'),
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
        with bind_runtime_context(**_bind_args_for_main_state(state)):
            run_synthesis(state)

    assert state.result.result_blocks[0].evidence_ids == [str(first_id), str(second_id)]
    assert state.result.used_evidence_ids == [str(first_id), str(second_id)]
