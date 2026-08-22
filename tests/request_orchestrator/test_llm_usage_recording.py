from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import mock_open, patch
from uuid import uuid4

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
from common.logging import create_conversation_event
from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE
from request_orchestrator.agents.profile_management.profile import build_profile_management_profile
from request_orchestrator.shared.request_analysis.analyze_request import analyze_request
from request_orchestrator.agents.profile_management.profile import PROFILE_MANAGEMENT_PROFILE
from request_orchestrator.models.agent_execution_context import AgentExecutionContext
from request_orchestrator.models.agent_prompt import PromptSectionKeys
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.orchestrator_result import OrchestratorResult
from request_orchestrator.models.request_analysis import RequestAnalysis, RequestAnalysisGoal
from request_orchestrator.models.main_state import MainState
from request_orchestrator.models.evidence import EvidenceView, HydratedEvidence, ToolResult
from request_orchestrator.models.evaluation_result import EVALUATION_STATUS_RETRYABLE
from request_orchestrator.models.plan import Plan
from request_orchestrator.models.plan_step_ids import format_plan_step_id, namespace_step_id
from request_orchestrator.shared.evaluator.evaluator import run_evaluator
from request_orchestrator.shared.planner.planner import REQUIRED_CAPABILITY_UNAVAILABLE_REASON, run_planner
from request_orchestrator.shared.runtime_context import bind_agent_context, bind_runtime_context
from request_orchestrator.shared.synthesis.synthesis import run_synthesis
from reranker.models import Candidate, CandidateContent
from reranker.service import rerank_candidates


class RecordingRepo:
    def __init__(self) -> None:
        self.llm_calls: list[dict] = []
        self.conversation_events: list[dict] = []

    def create_llm_call(self, **kwargs):
        self.llm_calls.append(kwargs)
        return kwargs

    def create_conversation_event(self, **kwargs):
        self.conversation_events.append(kwargs)
        return kwargs

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
        if event.get('agent_name') != agent_name:
            continue
        return event['payload']
    raise AssertionError(f'No event found for type={event_type} agent={agent_name}')


class FakeLangChainResponse:
    def __init__(self, content: str, model_name: str = 'gpt-5.6-luna') -> None:
        self.content = content
        self.usage_metadata = {
            'input_tokens': 100,
            'output_tokens': 20,
            'total_tokens': 120}
        self.response_metadata = {'model_name': model_name}


class FakeInvokeLLM:
    def __init__(self, content: str, model_name: str = 'gpt-5.6-luna') -> None:
        self.content = content
        self.model_name = model_name

    def invoke(self, prompt: str):
        return FakeLangChainResponse(self.content, self.model_name)


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.responses: list[object] = []

    def create(self, **kwargs):
        return self.responses.pop(0)


def _agent_profiles_for(user_profile: UserProfile) -> list:
    return [
        build_profile_management_profile(user_profile),
        MAIN_AGENT_PROFILE,
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
    tool_name_by_namespaced_step_id = {
        namespace_step_id(
            state.agent_profile.name,
            format_plan_step_id(plan_count, step.id),
        ): step.tool
        for step in (plan.steps if plan is not None else [])
    }
    normalized_tool_results: list[ToolResult] = []
    for step_id, value in (results or {}).items():
        namespaced_step_id = (
            step_id
            if ":" in step_id
            else namespace_step_id(state.agent_profile.name, step_id)
        )
        if isinstance(value, ToolResult):
            normalized_tool_results.append(
                value.model_copy(
                    update={
                        "step_id": value.step_id or namespaced_step_id,
                        "tool_name": value.tool_name or tool_name_by_namespaced_step_id.get(namespaced_step_id, ""),
                        "iteration": plan_count if value.iteration is None else value.iteration,
                    }
                )
            )
            continue
        normalized_tool_results.append(
            ToolResult(
                step_id=namespaced_step_id,
                tool_name=tool_name_by_namespaced_step_id.get(namespaced_step_id, ""),
                iteration=plan_count,
                result=value,
            )
        )
    state.result = state.result.copy(tool_results=normalized_tool_results)


def test_request_analysis_records_llm_usage() -> None:
    user_profile = UserProfile()
    repo = RecordingRepo()
    state = MainState.new(
        task='Find me boots.',
        execution_context=AgentExecutionContext.new(
            conversation_context=ConversationContext(),
            user_profile=user_profile,
            conversation_id=str(uuid4()),
        ),
        llm=FakeInvokeLLM('{"goals":[{"agent":"main_agent","goal":"Find boots","tool_categories":[]}],"requested_user_attribute_types":[]}'),
        agent_profiles=_agent_profiles_for(user_profile),
    )

    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'common.logging.conversation_event_logger.get_conversation_repo',
        return_value=repo,
    ), patch(
        'llm.chat_models.build_chat_model',
        return_value=state.llm,
    ):
        with bind_runtime_context(**_bind_args_for_main_state(state)):
            analyze_request(state)

    assert len(repo.llm_calls) == 1
    assert repo.llm_calls[0]['agent'] == 'main_agent'
    assert repo.llm_calls[0]['stage'] == 'request_analysis'
    input_object = repo.llm_calls[0]['metadata']['input_object']
    assert input_object['prompt_token_count'] > 0
    assert PromptSectionKeys.TASK in input_object['sections_raw']
    assert PromptSectionKeys.USER_PROFILE not in input_object['sections_raw']
    payload = _latest_event_payload(repo, event_type='request_analysis', agent_name='request_orchestrator')
    assert payload['data']['llm_usage']['total_tokens'] == 120
    assert isinstance(payload['data']['llm_usage']['latency_ms'], int)


def test_run_planner_records_main_and_profile_scopes() -> None:
    repo = RecordingRepo()
    main_state = AgentState.new(
        task='Find me boots.',
        execution_context=AgentExecutionContext.new(
            conversation_context=ConversationContext(),
            user_profile=UserProfile(),
            conversation_id=str(uuid4()),
        ),
        llm=FakeInvokeLLM('{"steps": [], "needs_replan": false}'),
        agent_profile=MAIN_AGENT_PROFILE,
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
        'common.logging.conversation_event_logger.get_conversation_repo',
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
    assert PromptSectionKeys.AVAILABLE_TOOLS in repo.llm_calls[0]['metadata']['input_object']['sections_raw']
    payload = _latest_event_payload(repo, event_type='plan', agent_name='main_agent')
    assert len(payload['data']['llm_usage']) == 1
    assert payload['data']['llm_usage'][0]['total_tokens'] == 120
    assert isinstance(payload['data']['llm_usage'][0]['latency_ms'], int)


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
        agent_profile=MAIN_AGENT_PROFILE,
    )

    with patch('common.logging.conversation_event_logger.get_conversation_repo', return_value=repo):
        with bind_runtime_context(
            conversation_id=str(conversation_id),
            conversation_model_config=None,
            roundtrip_id=str(roundtrip_id),
        ):
            create_conversation_event(
                conversation_id=state.execution_context.conversation_id,
                roundtrip_id=state.execution_context.roundtrip_id,
                event_type='planner',
                source='main_agent',
                agent_name='main_agent',
                node_name='plan',
                step_id='P1E1',
                iteration=1,
                payload={
                    'agent_name': 'main_agent',
                    'kind': 'planner',
                    'node_name': 'plan',
                    'step_id': 'P1E1',
                    'iteration': 1,
                    'data': {'step_plans': ['Find boots']},
                },
            )

    assert len(repo.conversation_events) == 1
    event = repo.conversation_events[0]
    assert event['conversation_id'] == conversation_id
    assert event['roundtrip_id'] == roundtrip_id
    assert event['event_type'] == 'planner'
    assert event['source'] == 'main_agent'
    assert event['agent_name'] == 'main_agent'
    assert event['node_name'] == 'plan'
    assert event['step_id'] == 'P1E1'
    assert event['iteration'] == 1
    assert event['payload']['kind'] == 'planner'
    assert event['payload']['node_name'] == 'plan'
    assert event['payload']['step_id'] == 'P1E1'


def test_run_planner_marks_blocked_when_tools_are_required_but_no_steps_are_returned() -> None:
    repo = RecordingRepo()
    state = AgentState.new(
        task='Find me boots.',
        execution_context=AgentExecutionContext.new(
            conversation_context=ConversationContext(),
            user_profile=UserProfile(),
            conversation_id=str(uuid4()),
        ),
        llm=FakeInvokeLLM('{"steps": [], "status": "blocked", "reason": "required capability unavailable.", "needs_replan": false}'),
        agent_profile=MAIN_AGENT_PROFILE,
    )
    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'common.logging.conversation_event_logger.get_conversation_repo',
        return_value=repo,
    ), patch(
        'llm.chat_models.build_chat_model',
        return_value=state.llm,
    ):
        with bind_runtime_context(**_bind_args_for_agent_state(state)):
            run_planner(state)

    assert len(repo.llm_calls) == 1
    assert state.node_states.evaluator.goal_reached is True
    assert state.node_states.planner.plan is not None
    assert state.node_states.planner.plan.steps == []
    payload = _latest_event_payload(repo, event_type='plan', agent_name='main_agent')
    assert payload['status'] == 'blocked'
    assert payload['data']['planner_status'] == 'blocked'
    assert payload['data']['planner_reason'] == REQUIRED_CAPABILITY_UNAVAILABLE_REASON
    assert payload['data']['needs_replan'] is False


def test_run_synthesis_records_llm_usage_after_tool_results() -> None:
    repo = RecordingRepo()
    state = MainState.new(
        task='Summarize this.',
        execution_context=AgentExecutionContext.new(
            conversation_context=ConversationContext(),
            user_profile=UserProfile(),
            conversation_id=str(uuid4()),
        ),
        llm=FakeInvokeLLM('{"result": [{"content": "done", "evidence_ids": []}], "next_question": "Do you want a deeper breakdown?", "roundtrip_summary": "summary", "tool_summary": {"produced": [], "entities": []}}', 'gpt-5.6-terra'),
        agent_profiles=_agent_profiles_for(UserProfile()),
    )

    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'common.logging.conversation_event_logger.get_conversation_repo',
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
    assert repo.llm_calls[0]['metadata']['input_object']['prompt_token_count'] > 0
    assert PromptSectionKeys.LATEST_USER_PROMPT in repo.llm_calls[0]['metadata']['input_object']['sections_raw']
    payload = _latest_event_payload(repo, event_type='synthesis', agent_name='request_orchestrator')
    assert payload['data']['llm_usage']['model'] == 'gpt-5.6-terra'
    assert isinstance(payload['data']['llm_usage']['latency_ms'], int)


def test_reranker_records_llm_usage_when_it_runs() -> None:
    repo = RecordingRepo()
    candidates = [
        Candidate(id=f'c{i}', source='web', content=CandidateContent(text=f'Item {i}'))
        for i in range(7)
    ]
    llm = FakeInvokeLLM('{"ranked_ids": ["c3", "c1", "c2"]}')

    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'common.logging.conversation_event_logger.get_conversation_repo',
        return_value=repo,
    ):
        with bind_runtime_context(
            conversation_id=str(uuid4()),
            conversation_model_config=None,
            roundtrip_id=str(uuid4()),
        ):
            with bind_agent_context(agent_name='main_agent'):
                rerank_candidates(candidates, goal='Find the best one', llm=llm)

    assert len(repo.llm_calls) == 1
    assert repo.llm_calls[0]['stage'] == 'reranker'
    assert repo.llm_calls[0]['metadata']['caller_agent_name'] == 'main_agent'
    assert repo.llm_calls[0]['metadata']['candidate_count'] == 7
    assert isinstance(repo.llm_calls[0]['metadata']['latency_ms'], int)
    assert repo.llm_calls[0]['metadata']['input_object']['goal'] == 'Find the best one'
    assert repo.llm_calls[0]['metadata']['input_object']['candidate_ids'] == [f'c{i}' for i in range(7)]
    assert repo.llm_calls[0]['metadata']['output_object']['raw_content'] == '{"ranked_ids": ["c3", "c1", "c2"]}'
    assert any(event['agent_name'] == 'main_agent' and event['event_type'] == 'llm_call' for event in repo.conversation_events)


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


def test_run_evaluator_records_llm_usage_and_refines_goal() -> None:
    repo = RecordingRepo()
    state = AgentState.new(
        task='Find current pricing for shortlisted products.',
        execution_context=AgentExecutionContext.new(
            conversation_context=ConversationContext(),
            user_profile=UserProfile(),
            conversation_id=str(uuid4()),
        ),
        llm=FakeInvokeLLM('{"status": "RETRYABLE", "relevant_evidence": ["P1E1R1"], "missing_information": ["Need current pricing for the top two products", "Need shipping availability in Canada"], "refined_goal": "Find current Canadian pricing and availability for the two shortlisted products."}', 'gpt-5.6-terra'),
        agent_profile=MAIN_AGENT_PROFILE,
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
        results={'P1E1': {'items': ['result']}},
    )

    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'common.logging.conversation_event_logger.get_conversation_repo',
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
    assert PromptSectionKeys.EVIDENCE in repo.llm_calls[0]['metadata']['input_object']['sections_raw']
    assert state.node_states.evaluator.evaluation_status == EVALUATION_STATUS_RETRYABLE
    assert state.inputs.task == 'Find current Canadian pricing and availability for the two shortlisted products.'
    payload = _latest_event_payload(repo, event_type='evaluator', agent_name='main_agent')
    assert payload['data']['llm_usage']['model'] == 'gpt-5.6-terra'
    assert isinstance(payload['data']['llm_usage']['latency_ms'], int)
    assert payload['data']['relevant_evidence'] == ['main_agent:P1E1R1']
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
        llm=CapturingLLM('{"result": [{"content": "done", "evidence_ids": ["P1E2R1"]}], "next_question": "Do you want a deeper breakdown?", "roundtrip_summary": "summary", "tool_summary": {"produced": [], "entities": []}}', 'gpt-5.6-terra'),
        agent_profiles=_agent_profiles_for(UserProfile()),
    )
    main_agent_state = state.agent_states['main_agent']
    _set_agent_tool_results(
        main_agent_state,
        plan=Plan.model_validate({
            'steps': [
                {'id': 'E1', 'plan': 'First step', 'tool': 'generic_web_search', 'args': {}},
                {'id': 'E2', 'plan': 'Second step', 'tool': 'generic_web_search', 'args': {}}]
        }),
        results={
            'P1E1': ToolResult(
                result={'value': 'a'},
                evidence_views=[
                    EvidenceView(
                        evidence_id='',
                        item_id='item-a',
                        title='Item A',
                        summary='First item',
                    )
                ],
                hydrated_evidence=[
                    HydratedEvidence(
                        item_id='item-a',
                        title='Item A',
                        summary='First item',
                    )
                ],
            ),
            'P1E2': ToolResult(
                result={'value': 'b'},
                evidence_views=[
                    EvidenceView(
                        evidence_id='',
                        item_id='item-b',
                        title='Item B',
                        summary='Second item',
                    )
                ],
                hydrated_evidence=[
                    HydratedEvidence(
                        item_id='item-b',
                        title='Item B',
                        summary='Second item',
                    )
                ],
            ),
        },
    )
    main_agent_state.result = main_agent_state.result.copy(relevant_evidence_ids=['main_agent:P1E2R1'])

    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'common.logging.conversation_event_logger.get_conversation_repo',
        return_value=repo,
    ), patch(
        'llm.chat_models.build_chat_model',
        return_value=state.llm,
    ):
        with bind_runtime_context(**_bind_args_for_main_state(state)):
            run_synthesis(state)

    prompt_text = str(captured_prompt['text'])
    assert '"evidence_id": "main_agent:P1E2R1"' in prompt_text
    assert '"evidence_id": "main_agent:P1E1R1"' not in prompt_text
    payload = _latest_event_payload(repo, event_type='synthesis', agent_name='request_orchestrator')
    assert payload['data']['relevant_evidence_ids'] == ['P1E2R1']
