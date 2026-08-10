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
from llm.clients.llm_client import LlmClient
from personalization.profile.models import UserProfile
from request_orchestrator.agents.main_agent.request_analysis.analyze_request import analyze_request
from request_orchestrator.agents.profile_management.profile import PROFILE_MANAGEMENT_PROFILE
from request_orchestrator.models.agent_state import AgentState, IterationState, RequestAnalysis
from request_orchestrator.models.evaluation_result import EVALUATION_STATUS_RETRYABLE
from request_orchestrator.models.evaluation_result import EVALUATION_STATUS_RETRYABLE
from request_orchestrator.models.plan import Plan
from request_orchestrator.shared.evaluator.evaluator import run_evaluator
from request_orchestrator.shared.planner.planner import run_planner
from request_orchestrator.shared.runtime_context import bind_runtime_context
from request_orchestrator.shared.synthesis.synthesis import run_synthesis
from reranker.models import Candidate, CandidateContent
from reranker.service import rerank_candidates


class RecordingRepo:
    def __init__(self) -> None:
        self.llm_calls: list[dict] = []

    def create_llm_call(self, **kwargs):
        self.llm_calls.append(kwargs)
        return kwargs

    def create_roundtrip_prompt(self, *args, **kwargs):
        return None


class FakeLangChainResponse:
    def __init__(self, content: str, model_name: str = 'gpt-5.4-mini') -> None:
        self.content = content
        self.usage_metadata = {
            'input_tokens': 100,
            'output_tokens': 20,
            'total_tokens': 120}
        self.response_metadata = {'model_name': model_name}


class FakeInvokeLLM:
    def __init__(self, content: str, model_name: str = 'gpt-5.4-mini') -> None:
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


def test_request_analysis_records_llm_usage() -> None:
    repo = RecordingRepo()
    state = AgentState.new(
        task='Find me boots.',
        max_turns=5,
        conversation_context=ConversationContext(),
        user_profile=UserProfile(),
        conversation_id=str(uuid4()),
        llm=FakeInvokeLLM('{"goal":"Find boots","applicable_tool_categories":[],"requested_user_attribute_types":[],"requires_tools":false,"context_answer_confidence":0.5}'),
    )

    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'request_orchestrator.agents.main_agent.request_analysis.analyze_request.get_conversation_repo',
        return_value=repo,
    ):
        analyze_request(state)

    assert len(repo.llm_calls) == 1
    assert repo.llm_calls[0]['agent'] == 'main_agent'
    assert repo.llm_calls[0]['stage'] == 'request_analysis'
    assert state.agent_log.entries[-1].data['llm_usage']['total_tokens'] == 120


def test_run_planner_records_main_and_profile_scopes() -> None:
    repo = RecordingRepo()
    main_state = AgentState.new(
        task='Find me boots.',
        max_turns=5,
        conversation_context=ConversationContext(),
        user_profile=UserProfile(),
        conversation_id=str(uuid4()),
        llm=FakeInvokeLLM('{"steps": []}'),
    )
    main_state.request_analysis = RequestAnalysis(goal='Find boots', requires_tools=False)

    profile_state = AgentState.new(
        task='Remember I like pizza.',
        max_turns=5,
        conversation_context=ConversationContext(),
        user_profile=UserProfile(),
        conversation_id=str(uuid4()),
        llm=FakeInvokeLLM('{"steps": []}'),
        agent_profile=PROFILE_MANAGEMENT_PROFILE,
    )
    profile_state.request_analysis = RequestAnalysis(goal='Update profile', requires_tools=False)

    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'request_orchestrator.shared.planner.planner.get_conversation_repo',
        return_value=repo,
    ):
        run_planner(main_state)
        run_planner(profile_state)

    assert [call['agent'] for call in repo.llm_calls] == ['main_agent', 'profile_agent']
    assert all(call['stage'] == 'planner' for call in repo.llm_calls)
    assert len(main_state.agent_log.entries[-1].data['llm_usage']) == 1
    assert main_state.agent_log.entries[-1].data['llm_usage'][0]['total_tokens'] == 120


def test_run_synthesis_records_llm_usage_after_tool_results() -> None:
    repo = RecordingRepo()
    state = AgentState.new(
        task='Summarize this.',
        max_turns=5,
        conversation_context=ConversationContext(),
        user_profile=UserProfile(),
        conversation_id=str(uuid4()),
        llm=FakeInvokeLLM('{"result": ["done"], "follow_up": "", "clarifying_question": "", "roundtrip_summary": "summary", "tool_summary": {"used_tools": [], "produced": [], "entities": [], "freshness": ""}}', 'gpt-5.4'),
    )
    state.iteration_trace = [IterationState(plan=Plan.model_validate({'steps': []}), results={})]

    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'request_orchestrator.shared.synthesis.synthesis.get_conversation_repo',
        return_value=repo,
    ):
        run_synthesis(state)

    assert len(repo.llm_calls) == 1
    assert repo.llm_calls[0]['stage'] == 'synthesis'
    assert repo.llm_calls[0]['model'] == 'gpt-5.4'
    assert state.agent_log.entries[-1].data['llm_usage']['model'] == 'gpt-5.4'


def test_reranker_records_llm_usage_when_it_runs() -> None:
    repo = RecordingRepo()
    candidates = [
        Candidate(id=f'c{i}', source='web', content=CandidateContent(text=f'Item {i}'))
        for i in range(7)
    ]
    llm = FakeInvokeLLM('{"ranked_ids": ["c3", "c1", "c2"]}')

    with patch('llm.usage.get_conversation_repo', return_value=repo):
        with bind_runtime_context(conversation_id=str(uuid4()), conversation_model_config=None, roundtrip_id=str(uuid4())):
            rerank_candidates(candidates, goal='Find the best one', llm=llm)

    assert len(repo.llm_calls) == 1
    assert repo.llm_calls[0]['stage'] == 'reranker'
    assert repo.llm_calls[0]['metadata']['candidate_count'] == 7


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

    llm_client = LlmClient(client=client, default_model='gpt-5.4-mini')

    with patch('llm.usage.get_conversation_repo', return_value=repo), patch('builtins.open', mock_open(read_data=b'image-bytes')):
        with bind_runtime_context(conversation_id=str(uuid4()), conversation_model_config=None, roundtrip_id=str(uuid4())):
            llm_client.call_with_tools(system_prompt='sys', messages=[], tools=[])
            llm_client.generate_caption_from_image_file('fake.jpg')

    assert [call['stage'] for call in repo.llm_calls] == ['tool_calling', 'image_caption']


def test_run_evaluator_records_llm_usage_and_refines_goal() -> None:
    repo = RecordingRepo()
    state = AgentState.new(
        task='Find current pricing for shortlisted products.',
        max_turns=5,
        conversation_context=ConversationContext(),
        user_profile=UserProfile(),
        conversation_id=str(uuid4()),
        llm=FakeInvokeLLM('{"status": "RETRYABLE", "relevant_evidence": ["E1"], "missing_information": ["Need current pricing for the top two products", "Need shipping availability in Canada"], "refined_goal": "Find current Canadian pricing and availability for the two shortlisted products."}', 'gpt-5.4'),
    )
    state.request_analysis = RequestAnalysis(goal='Check whether the shortlisted products satisfy the request.', requires_tools=False)
    state.iteration_trace = [
        IterationState(
            plan=Plan.model_validate({
                'steps': [
                    {
                        'id': 'E1',
                        'plan': 'Search for the shortlisted products.',
                        'tool': 'generic_web_search',
                        'args': {'query_text': 'shortlisted products'}}
                ]
            }),
            results={'E1': {'items': ['result']}},
        )
    ]

    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'request_orchestrator.shared.evaluator.evaluator.get_conversation_repo',
        return_value=repo,
    ):
        run_evaluator(state)

    assert len(repo.llm_calls) == 1
    assert repo.llm_calls[0]['agent'] == 'shared'
    assert repo.llm_calls[0]['stage'] == 'evaluator'
    assert repo.llm_calls[0]['model'] == 'gpt-5.4'
    assert state.evaluation_status == EVALUATION_STATUS_RETRYABLE
    assert state.request_analysis.goal == 'Find current Canadian pricing and availability for the two shortlisted products.'
    assert state.agent_log.entries[-1].data['llm_usage']['model'] == 'gpt-5.4'
    assert state.agent_log.entries[-1].data['relevant_evidence'] == ['E1']
    assert state.agent_log.entries[-1].data['missing_information'] == [
        'Need current pricing for the top two products',
        'Need shipping availability in Canada']


def test_run_synthesis_filters_to_relevant_evidence_ids_when_available() -> None:
    repo = RecordingRepo()
    captured_prompt: dict[str, object] = {}

    class CapturingLLM(FakeInvokeLLM):
        def invoke(self, prompt: str):
            captured_prompt['text'] = prompt
            return super().invoke(prompt)

    state = AgentState.new(
        task='Summarize this.',
        max_turns=5,
        conversation_context=ConversationContext(),
        user_profile=UserProfile(),
        conversation_id=str(uuid4()),
        llm=CapturingLLM('{"result": ["done"], "follow_up": "", "clarifying_question": "", "roundtrip_summary": "summary", "tool_summary": {"used_tools": [], "produced": [], "entities": [], "freshness": ""}}', 'gpt-5.4'),
    )
    state.relevant_evidence_ids = ['E2']
    state.iteration_trace = [
        IterationState(
            plan=Plan.model_validate({
                'steps': [
                    {'id': 'E1', 'plan': 'First step', 'tool': 'tool_a', 'args': {}},
                    {'id': 'E2', 'plan': 'Second step', 'tool': 'tool_b', 'args': {}}]
            }),
            results={'E1': {'value': 'a'}, 'E2': {'value': 'b'}},
        )
    ]

    with patch('llm.usage.get_conversation_repo', return_value=repo), patch(
        'request_orchestrator.shared.synthesis.synthesis.get_conversation_repo',
        return_value=repo,
    ):
        run_synthesis(state)

    prompt_text = str(captured_prompt['text'])
    assert '"step_id": "E2"' in prompt_text
    assert '"step_id": "E1"' not in prompt_text
    assert state.agent_log.entries[-1].data['relevant_evidence_ids'] == ['E2']
