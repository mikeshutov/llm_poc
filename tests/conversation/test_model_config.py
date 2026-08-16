from __future__ import annotations

import sys
from decimal import Decimal
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
    ModelPricing,
    EVALUATOR_STAGE,
    PLANNER_STAGE,
    PROFILE_AGENT_MODEL_SCOPE,
    REQUEST_ANALYSIS_STAGE,
    RERANKER_STAGE,
    SHARED_MODEL_SCOPE,
)
from conversation.models.conversation_models import ConversationContext, ConversationRoundtrip
from conversation.models.conversation_models import Conversation
from integrations.ip_api.models import IpLocation
from personalization.profile.models import UserProfile
from rendering.sidebar import build_model_config_rows
from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE
from request_orchestrator.models.agent_execution_context import AgentExecutionContext
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.orchestrator_result import OrchestratorResult
from request_orchestrator.service import run_request_orchestrator_for_query
from request_orchestrator.shared.llm_factory import build_llm_for_stage
from request_orchestrator.shared.runtime_context import bind_runtime_context
from reranker.service import CandidateReranker


class TrackingChatOpenAI:
    def __init__(self, model: str):
        self.model = model

    def invoke(self, prompt: str):
        raise AssertionError(f'Unexpected invoke for prompt: {prompt[:80]}')


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

    def get_conversation(self, conversation_id):
        assert conversation_id == self.conversation_id
        return Conversation(
            id=self.conversation_id,
            user_id='anonymous',
            title='Test Conversation',
            created_at='2026-08-08T00:00:00Z',
            metadata={},
            tone_state={},
            summary='',
            summary_embedding=None,
        )

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
                stage=EVALUATOR_STAGE,
                model='gpt-4o-mini',
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
    assert config.shared.evaluator == 'gpt-4o-mini'
    assert config.shared.reranker == 'gpt-5.4'


def test_conversation_model_config_build_default_returns_defaults() -> None:
    config = ConversationModelConfig.build_default()

    assert config.main_agent.request_analysis == 'gpt-5.4-mini'
    assert config.main_agent.planner == 'gpt-5.4'
    assert config.main_agent.synthesis == 'gpt-5.4'
    assert config.profile_agent.planner == 'gpt-5.4-mini'
    assert config.shared.evaluator == 'gpt-5.4-mini'
    assert config.shared.reranker == 'gpt-5.4-mini'


def test_conversation_model_config_build_default_resolves_pricing_for_every_stage() -> None:
    config = ConversationModelConfig.build_default()

    assert config.resolve_pricing(MAIN_AGENT_MODEL_SCOPE, REQUEST_ANALYSIS_STAGE) == ModelPricing(
        input_price_per_million_tokens=Decimal('0.75'),
        output_price_per_million_tokens=Decimal('4.50'),
    )
    assert config.resolve_pricing(MAIN_AGENT_MODEL_SCOPE, PLANNER_STAGE) == ModelPricing(
        input_price_per_million_tokens=Decimal('2.50'),
        output_price_per_million_tokens=Decimal('15.00'),
    )
    assert config.resolve_pricing(PROFILE_AGENT_MODEL_SCOPE, PLANNER_STAGE) == ModelPricing(
        input_price_per_million_tokens=Decimal('0.75'),
        output_price_per_million_tokens=Decimal('4.50'),
    )
    assert config.resolve_pricing(SHARED_MODEL_SCOPE, EVALUATOR_STAGE) == ModelPricing(
        input_price_per_million_tokens=Decimal('0.75'),
        output_price_per_million_tokens=Decimal('4.50'),
    )
    assert config.resolve_pricing(SHARED_MODEL_SCOPE, RERANKER_STAGE) == ModelPricing(
        input_price_per_million_tokens=Decimal('0.75'),
        output_price_per_million_tokens=Decimal('4.50'),
    )


def test_conversation_model_config_override_changes_resolved_pricing_for_stage() -> None:
    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=PROFILE_AGENT_MODEL_SCOPE,
                stage=PLANNER_STAGE,
                model='gpt-4o-mini',
            ),
        ]
    )

    assert config.resolve_pricing(PROFILE_AGENT_MODEL_SCOPE, PLANNER_STAGE) == ModelPricing(
        input_price_per_million_tokens=Decimal('0.15'),
        output_price_per_million_tokens=Decimal('0.60'),
    )
    assert config.resolve_pricing(MAIN_AGENT_MODEL_SCOPE, PLANNER_STAGE) == ModelPricing(
        input_price_per_million_tokens=Decimal('2.50'),
        output_price_per_million_tokens=Decimal('15.00'),
    )


def test_conversation_model_config_resolve_model_pricing_returns_expected_values() -> None:
    pricing = ConversationModelConfig.resolve_model_pricing('o3')

    assert pricing == ModelPricing(
        input_price_per_million_tokens=Decimal('2.00'),
        output_price_per_million_tokens=Decimal('8.00'),
    )


def test_llm_factory_build_llm_for_stage_uses_conversation_model_config() -> None:
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

    with patch('request_orchestrator.shared.llm_factory.ChatOpenAI', TrackingChatOpenAI):
        state = AgentState.new(
            task='Help me remember this.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
                model_config=config,
            ),
            agent_profile=MAIN_AGENT_PROFILE,
            llm=TrackingChatOpenAI(model='gpt-5.4-mini'),
        )

        request_analysis_llm = build_llm_for_stage(
            execution_context=state.execution_context,
            llm=state.llm,
            agent=MAIN_AGENT_MODEL_SCOPE,
            stage=REQUEST_ANALYSIS_STAGE,
        )
        assert request_analysis_llm.model == 'gpt-5.4'

        profile_planner_llm = build_llm_for_stage(
            execution_context=state.execution_context,
            llm=state.llm,
            agent=PROFILE_AGENT_MODEL_SCOPE,
            stage=PLANNER_STAGE,
            reuse_llm_for_agent_scope=state.resolve_agent_scope(),
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
    captured_context_kwargs: dict[str, object] = {}

    def fake_build_roundtrip_context(conversation_id_arg, limit=5):
        captured_context_kwargs['conversation_id'] = conversation_id_arg
        captured_context_kwargs['limit'] = limit
        return ConversationContext()

    with patch('request_orchestrator.service.get_conversation_repo', return_value=fake_repo), patch(
        'request_orchestrator.service.build_roundtrip_context',
        side_effect=fake_build_roundtrip_context,
    ), patch(
        'request_orchestrator.service.build_user_profile',
        return_value=UserProfile(),
    ), patch(
        'request_orchestrator.service.MainState.initialize_agent_states',
        return_value=None,
    ), patch(
        'request_orchestrator.service.run_agent',
        return_value=OrchestratorResult(agent_result=AgentResult(), answer=['done']),
    ), patch(
        'request_orchestrator.service.embed_text',
        return_value=[0.1, 0.2, 0.3],
    ):
        run_request_orchestrator_for_query(str(conversation_id), 'hello world', user_id='anonymous')

    assert fake_repo.pending_calls[0]['model'] == config.main_agent.planner
    assert fake_repo.pending_calls[0]['metadata'] == {
        'resolved_model_config': config.to_metadata_payload(),
    }
    assert captured_context_kwargs == {
        'conversation_id': str(conversation_id),
        'limit': 5,
    }
    assert 'llm_usage' not in fake_repo.update_calls[0]['payload']
    assert 'agent_logs' not in fake_repo.update_calls[0]['payload']
    assert isinstance(fake_repo.update_calls[0]['payload']['roundtrip_latency_ms'], int)
    assert fake_repo.update_calls[0]['payload']['roundtrip_latency_ms'] >= 0


def test_run_request_orchestrator_passes_geometadata_to_user_profile_builder() -> None:
    conversation_id = uuid4()
    config = ConversationModelConfig.build_default()
    fake_repo = FakeConversationRepository(config)
    fake_repo.conversation_id = conversation_id
    captured_profile_kwargs: dict[str, object] = {}

    def fake_build_user_profile(**kwargs):
        captured_profile_kwargs.update(kwargs)
        return UserProfile(
            user_id='anonymous',
            geometadata=kwargs.get('geometadata'),
        )

    with patch('request_orchestrator.service.get_conversation_repo', return_value=fake_repo), patch(
        'request_orchestrator.service.build_roundtrip_context',
        return_value=ConversationContext(),
    ), patch(
        'request_orchestrator.service.build_user_profile',
        side_effect=fake_build_user_profile,
    ), patch(
        'request_orchestrator.service.MainState.initialize_agent_states',
        return_value=None,
    ), patch(
        'request_orchestrator.service.run_agent',
        return_value=OrchestratorResult(agent_result=AgentResult(), answer=['done']),
    ), patch(
        'request_orchestrator.service.embed_text',
        return_value=[0.1, 0.2, 0.3],
    ):
        run_request_orchestrator_for_query(str(conversation_id), 'hello world', user_id='anonymous')

    assert captured_profile_kwargs['geometadata'] is None


def test_build_model_config_rows_exposes_effective_models_overrides_and_pricing() -> None:
    conversation_id = uuid4()
    evaluator_override = ConversationModelConfigEntry(
        conversation_id=conversation_id,
        agent=SHARED_MODEL_SCOPE,
        stage=EVALUATOR_STAGE,
        model='gpt-4o-mini',
    )
    reranker_override = ConversationModelConfigEntry(
        conversation_id=conversation_id,
        agent=SHARED_MODEL_SCOPE,
        stage=RERANKER_STAGE,
        model='gpt-5.4',
    )
    resolved = resolve_conversation_model_config([evaluator_override, reranker_override])

    rows = build_model_config_rows(resolved, [evaluator_override, reranker_override])
    evaluator_row = next(row for row in rows if row['agent'] == SHARED_MODEL_SCOPE and row['stage'] == EVALUATOR_STAGE)
    reranker_row = next(row for row in rows if row['agent'] == SHARED_MODEL_SCOPE and row['stage'] == RERANKER_STAGE)

    assert evaluator_row['effective_model'] == 'gpt-4o-mini'
    assert evaluator_row['override_model'] == 'gpt-4o-mini'
    assert evaluator_row['input_price'] == '$0.15 per 1M'
    assert evaluator_row['output_price'] == '$0.6 per 1M'
    assert reranker_row['effective_model'] == 'gpt-5.4'
    assert reranker_row['override_model'] == 'gpt-5.4'
    assert reranker_row['input_price'] == '$2.5 per 1M'
    assert reranker_row['output_price'] == '$15 per 1M'


def test_build_model_config_rows_reset_to_default_restores_model_and_pricing() -> None:
    resolved = ConversationModelConfig.build_default()

    rows = build_model_config_rows(resolved, [])
    request_analysis_row = next(row for row in rows if row['agent'] == MAIN_AGENT_MODEL_SCOPE and row['stage'] == REQUEST_ANALYSIS_STAGE)

    assert request_analysis_row['effective_model'] == 'gpt-5.4-mini'
    assert request_analysis_row['override_model'] is None
    assert request_analysis_row['input_price'] == '$0.75 per 1M'
    assert request_analysis_row['output_price'] == '$4.5 per 1M'
