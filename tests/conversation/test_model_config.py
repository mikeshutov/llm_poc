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

from conversation.model_config_resolver import resolve_conversation_model_config
from conversation.models.conversation_model_config import (
    ConversationModelConfig,
    ConversationModelConfigEntry,
    MAIN_AGENT_MODEL_SCOPE,
    PLANNER_STAGE,
    PROFILE_AGENT_MODEL_SCOPE,
    REQUEST_ANALYSIS_STAGE,
    RERANKER_STAGE,
    SHARED_MODEL_SCOPE,
)
from conversation.models.conversation_models import ConversationContext, ConversationRoundtrip
from personalization.profile.models import UserProfile
from rendering.sidebar import build_model_config_rows
from request_orchestrator.service import run_request_orchestrator_for_query
from request_orchestrator.shared.runtime_context import bind_runtime_context
from reranker.service import CandidateReranker


class TrackingChatOpenAI:
    def __init__(self, model: str):
        self.model = model

    def invoke(self, prompt: str):
        raise AssertionError(f'Unexpected invoke for prompt: {prompt[:80]}')


class DummyAgentResult:
    def __init__(self, answer: list[str], roundtrip_summary: str = ''):
        self.answer = answer
        self.roundtrip_summary = roundtrip_summary

    @property
    def raw_response(self) -> str:
        return '\n\n'.join(self.answer)

    def to_payload_for_update_roundtrip(self) -> dict:
        return {
            'response': self.raw_response,
            'roundtrip_summary': self.roundtrip_summary,
            'cards': [],
            'follow_up': '',
            'clarifying_question': '',
            'tool_summary': {},
            'agent_logs': {},
        }


class FakeConversationRepository:
    def __init__(self, config: ConversationModelConfig):
        self.config = config
        self.pending_calls: list[dict] = []
        self.update_calls: list[dict] = []
        self.conversation_id = uuid4()
        self.roundtrip_id = uuid4()

    def resolve_conversation_model_config(self, conversation_id):
        assert conversation_id == self.conversation_id
        return self.config

    def create_pending_roundtrip(self, conversation_id, user_prompt, model=None, roundtrip_summary=None, roundtrip_summary_embedding=None, metadata=None):
        self.pending_calls.append(
            {
                'conversation_id': conversation_id,
                'user_prompt': user_prompt,
                'model': model,
                'metadata': metadata,
            }
        )
        return ConversationRoundtrip(
            id=self.roundtrip_id,
            conversation_id=conversation_id,
            message_index=0,
            user_prompt=user_prompt,
            generated_response='',
            roundtrip_summary=None,
            roundtrip_summary_embedding=None,
            response_payload={},
            parsed_query={},
            created_at='2026-08-08T00:00:00Z',
            metadata=metadata or {},
            model=model,
        )

    def update_roundtrip(self, roundtrip_id, response, payload, roundtrip_summary=None, roundtrip_summary_embedding=None):
        self.update_calls.append(
            {
                'roundtrip_id': roundtrip_id,
                'response': response,
                'payload': payload,
                'roundtrip_summary': roundtrip_summary,
            }
        )
        return ConversationRoundtrip(
            id=roundtrip_id,
            conversation_id=self.conversation_id,
            message_index=0,
            user_prompt='original prompt',
            generated_response=response,
            roundtrip_summary=roundtrip_summary,
            roundtrip_summary_embedding=roundtrip_summary_embedding,
            response_payload=payload,
            parsed_query={},
            created_at='2026-08-08T00:00:00Z',
            metadata={},
            model=self.pending_calls[0]['model'],
        )


def test_conversation_model_config_resolves_partial_overrides_with_defaults() -> None:
    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=MAIN_AGENT_MODEL_SCOPE,
                stage=REQUEST_ANALYSIS_STAGE,
                model='gpt-5.4',
            ),
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=SHARED_MODEL_SCOPE,
                stage=RERANKER_STAGE,
                model='gpt-5.4',
            ),
        ]
    )

    assert config.main_agent.request_analysis == 'gpt-5.4'
    assert config.main_agent.planner == 'gpt-5.4'
    assert config.main_agent.synthesis == 'gpt-5.4'
    assert config.profile_agent.planner == 'gpt-5.4-mini'
    assert config.shared.reranker == 'gpt-5.4'


def test_conversation_model_config_build_default_returns_defaults() -> None:
    config = ConversationModelConfig.build_default()

    assert config.main_agent.request_analysis == 'gpt-5.4-mini'
    assert config.main_agent.planner == 'gpt-5.4'
    assert config.main_agent.synthesis == 'gpt-5.4'
    assert config.profile_agent.planner == 'gpt-5.4-mini'
    assert config.shared.reranker == 'gpt-5.4-mini'


def test_agent_state_build_llm_for_stage_uses_conversation_model_config() -> None:
    from request_orchestrator.models.agent_state import AgentState

    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=MAIN_AGENT_MODEL_SCOPE,
                stage=REQUEST_ANALYSIS_STAGE,
                model='gpt-5.4',
            ),
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=PROFILE_AGENT_MODEL_SCOPE,
                stage=PLANNER_STAGE,
                model='gpt-5.4',
            ),
        ]
    )

    with patch('request_orchestrator.models.agent_state.ChatOpenAI', TrackingChatOpenAI):
        state = AgentState.new(
            task='Help me remember this.',
            max_turns=5,
            conversation_context=ConversationContext(),
            user_profile=UserProfile(),
            llm=TrackingChatOpenAI(model='gpt-5.4-mini'),
            conversation_model_config=config,
        )

        request_analysis_llm = state.build_llm_for_stage(
            agent=MAIN_AGENT_MODEL_SCOPE,
            stage=REQUEST_ANALYSIS_STAGE,
        )
        assert request_analysis_llm.model == 'gpt-5.4'

        profile_planner_llm = state.build_llm_for_stage(
            agent=PROFILE_AGENT_MODEL_SCOPE,
            stage=PLANNER_STAGE,
        )
        assert profile_planner_llm.model == 'gpt-5.4'


def test_candidate_reranker_uses_shared_conversation_model_from_runtime_context() -> None:
    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=SHARED_MODEL_SCOPE,
                stage=RERANKER_STAGE,
                model='gpt-5.4',
            )
        ]
    )

    with patch('reranker.service.ChatOpenAI', TrackingChatOpenAI):
        with bind_runtime_context(conversation_id=str(conversation_id), conversation_model_config=config):
            reranker = CandidateReranker()

    assert reranker.llm.model == 'gpt-5.4'


def test_run_request_orchestrator_records_resolved_model_config_snapshot() -> None:
    conversation_id = uuid4()
    config = ConversationModelConfig.build_default()
    fake_repo = FakeConversationRepository(config)
    fake_repo.conversation_id = conversation_id

    with patch('request_orchestrator.service.get_conversation_repo', return_value=fake_repo), patch(
        'request_orchestrator.service.build_roundtrip_context',
        return_value=ConversationContext(),
    ), patch(
        'request_orchestrator.service.build_user_profile',
        return_value=UserProfile(),
    ), patch(
        'request_orchestrator.service.run_agent',
        return_value=DummyAgentResult(answer=['done'], roundtrip_summary='summary'),
    ), patch(
        'request_orchestrator.service.embed_text',
        return_value=[0.1, 0.2, 0.3],
    ):
        run_request_orchestrator_for_query(str(conversation_id), 'hello world')

    assert fake_repo.pending_calls[0]['model'] == config.main_agent.planner
    assert fake_repo.pending_calls[0]['metadata'] == {
        'resolved_model_config': config.to_metadata_payload(),
    }


def test_build_model_config_rows_exposes_effective_and_override_values() -> None:
    conversation_id = uuid4()
    override = ConversationModelConfigEntry(
        conversation_id=conversation_id,
        agent=SHARED_MODEL_SCOPE,
        stage=RERANKER_STAGE,
        model='gpt-5.4',
    )
    resolved = resolve_conversation_model_config([override])

    rows = build_model_config_rows(resolved, [override])
    reranker_row = next(row for row in rows if row['agent'] == SHARED_MODEL_SCOPE and row['stage'] == RERANKER_STAGE)

    assert reranker_row['effective_model'] == 'gpt-5.4'
    assert reranker_row['override_model'] == 'gpt-5.4'
