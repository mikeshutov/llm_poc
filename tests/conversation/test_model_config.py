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

from conversation.models.conversation_models import ConversationContext, ConversationRoundtrip
from conversation.models.conversation_models import Conversation
from integrations.ip_api.models import IpLocation
from llm.chat_models import build_llm_for_stage
from llm.conversation_model_config import (
    ANTHROPIC_PROVIDER,
    COHERE_PROVIDER,
    ConversationModelConfig,
    ConversationModelConfigEntry,
    DEEPSEEK_PROVIDER,
    EVALUATOR_STAGE,
    GOOGLE_PROVIDER,
    MAIN_AGENT_MODEL_SCOPE,
    MISTRAL_PROVIDER,
    ModelPricing,
    OPENAI_PROVIDER,
    PLANNER_STAGE,
    PROFILE_AGENT_MODEL_SCOPE,
    REQUEST_ANALYSIS_STAGE,
    RERANKER_STAGE,
    SHARED_MODEL_SCOPE,
    SYNTHESIS_STAGE,
    XAI_PROVIDER,
)
from llm.model_config_resolver import resolve_conversation_model_config
from personalization.profile.models import UserProfile
from rendering.sidebar import build_model_config_rows
from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE
from request_orchestrator.models.agent_execution_context import AgentExecutionContext
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.orchestrator_result import OrchestratorResult
from request_orchestrator.service import run_request_orchestrator_for_query
from request_orchestrator.shared.runtime_context import bind_runtime_context
from reranker.service import CandidateReranker


class TrackingChatOpenAI:
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def invoke(self, prompt: str):
        raise AssertionError(f'Unexpected invoke for prompt: {prompt[:80]}')


class TrackingChatAnthropic:
    def __init__(self, model_name: str, **kwargs):
        self.model = model_name
        self.kwargs = kwargs

    def invoke(self, prompt: str):
        raise AssertionError(f'Unexpected invoke for prompt: {prompt[:80]}')


class FakeConversationRepository:
    def __init__(self):
        self.pending_calls: list[dict] = []
        self.update_calls: list[dict] = []
        self.conversation_id = uuid4()
        self.roundtrip_id = uuid4()

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

    def update_roundtrip(self, roundtrip_id, response, payload, roundtrip_summary=None, roundtrip_summary_embedding=None, assistant_follow_up=None):
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
            assistant_follow_up=assistant_follow_up or '',
            response_payload=payload,
            parsed_query={},
            created_at='2026-08-08T00:00:00Z',
            metadata={},
            model=self.pending_calls[0]['model'],
        )


class FakeConversationModelConfigRepository:
    def __init__(self, config: ConversationModelConfig, conversation_id):
        self.config = config
        self.conversation_id = conversation_id

    def resolve(self, conversation_id):
        assert conversation_id == self.conversation_id
        return self.config


def test_conversation_model_config_resolves_partial_overrides_with_defaults() -> None:
    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=MAIN_AGENT_MODEL_SCOPE,
                stage=REQUEST_ANALYSIS_STAGE,
                provider=OPENAI_PROVIDER,
                model='gpt-5.6-terra',
            ),
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=SHARED_MODEL_SCOPE,
                stage=EVALUATOR_STAGE,
                provider=OPENAI_PROVIDER,
                model='gpt-5.6-luna',
            ),
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=SHARED_MODEL_SCOPE,
                stage=RERANKER_STAGE,
                provider=OPENAI_PROVIDER,
                model='gpt-5.6-terra',
            ),
        ]
    )

    assert config.main_agent.request_analysis.provider == OPENAI_PROVIDER
    assert config.main_agent.request_analysis.model == 'gpt-5.6-terra'
    assert config.main_agent.planner.model == 'gpt-5.6-terra'
    assert config.main_agent.synthesis.model == 'gpt-5.6-terra'
    assert config.profile_agent.planner.model == 'gpt-5.6-luna'
    assert config.shared.evaluator.model == 'gpt-5.6-luna'
    assert config.shared.reranker.model == 'gpt-5.6-terra'


def test_conversation_model_config_build_default_returns_defaults() -> None:
    config = ConversationModelConfig.build_default()

    assert config.main_agent.request_analysis.provider == OPENAI_PROVIDER
    assert config.main_agent.request_analysis.model == 'gpt-5.6-luna'
    assert config.main_agent.planner.model == 'gpt-5.6-terra'
    assert config.main_agent.synthesis.model == 'gpt-5.6-terra'
    assert config.profile_agent.planner.model == 'gpt-5.6-luna'
    assert config.shared.evaluator.model == 'gpt-5.6-luna'
    assert config.shared.reranker.model == 'gpt-5.6-luna'
    assert ConversationModelConfig.default_shared_reranker_model() == 'gpt-5.6-luna'


def test_conversation_model_config_build_default_resolves_pricing_for_every_stage() -> None:
    config = ConversationModelConfig.build_default()

    assert config.resolve_pricing(MAIN_AGENT_MODEL_SCOPE, REQUEST_ANALYSIS_STAGE) == ModelPricing(
        input_price_per_million_tokens=Decimal('1.00'),
        cached_input_price_per_million_tokens=Decimal('0.10'),
        output_price_per_million_tokens=Decimal('6.00'),
    )
    assert config.resolve_pricing(MAIN_AGENT_MODEL_SCOPE, PLANNER_STAGE) == ModelPricing(
        input_price_per_million_tokens=Decimal('2.50'),
        cached_input_price_per_million_tokens=Decimal('0.25'),
        output_price_per_million_tokens=Decimal('15.00'),
    )
    assert config.resolve_pricing(PROFILE_AGENT_MODEL_SCOPE, PLANNER_STAGE) == ModelPricing(
        input_price_per_million_tokens=Decimal('1.00'),
        cached_input_price_per_million_tokens=Decimal('0.10'),
        output_price_per_million_tokens=Decimal('6.00'),
    )
    assert config.resolve_pricing(SHARED_MODEL_SCOPE, EVALUATOR_STAGE) == ModelPricing(
        input_price_per_million_tokens=Decimal('1.00'),
        cached_input_price_per_million_tokens=Decimal('0.10'),
        output_price_per_million_tokens=Decimal('6.00'),
    )
    assert config.resolve_pricing(SHARED_MODEL_SCOPE, RERANKER_STAGE) == ModelPricing(
        input_price_per_million_tokens=Decimal('1.00'),
        cached_input_price_per_million_tokens=Decimal('0.10'),
        output_price_per_million_tokens=Decimal('6.00'),
    )


def test_conversation_model_config_override_changes_resolved_pricing_for_stage() -> None:
    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=PROFILE_AGENT_MODEL_SCOPE,
                stage=PLANNER_STAGE,
                provider=OPENAI_PROVIDER,
                model='gpt-5.6-luna',
            ),
        ]
    )

    assert config.resolve_pricing(PROFILE_AGENT_MODEL_SCOPE, PLANNER_STAGE) == ModelPricing(
        input_price_per_million_tokens=Decimal('1.00'),
        cached_input_price_per_million_tokens=Decimal('0.10'),
        output_price_per_million_tokens=Decimal('6.00'),
    )
    assert config.resolve_pricing(MAIN_AGENT_MODEL_SCOPE, PLANNER_STAGE) == ModelPricing(
        input_price_per_million_tokens=Decimal('2.50'),
        cached_input_price_per_million_tokens=Decimal('0.25'),
        output_price_per_million_tokens=Decimal('15.00'),
    )


def test_conversation_model_config_resolve_model_pricing_returns_expected_values() -> None:
    pricing = ConversationModelConfig.resolve_model_pricing('o3')

    assert pricing == ModelPricing(
        input_price_per_million_tokens=Decimal('2.00'),
        output_price_per_million_tokens=Decimal('8.00'),
    )


def test_conversation_model_config_groups_models_by_provider() -> None:
    grouped = ConversationModelConfig.model_names_by_provider()

    assert OPENAI_PROVIDER in grouped
    assert ANTHROPIC_PROVIDER in grouped
    assert GOOGLE_PROVIDER in grouped
    assert XAI_PROVIDER in grouped
    assert MISTRAL_PROVIDER in grouped
    assert COHERE_PROVIDER in grouped
    assert DEEPSEEK_PROVIDER in grouped
    assert "gpt-5.6-terra" in grouped[OPENAI_PROVIDER]
    assert "o3" in grouped[OPENAI_PROVIDER]
    assert "claude-sonnet-5" in grouped[ANTHROPIC_PROVIDER]
    assert "gemini-3.5-flash" in grouped[GOOGLE_PROVIDER]
    assert "grok-4.5" in grouped[XAI_PROVIDER]
    assert "mistral-small-latest" in grouped[MISTRAL_PROVIDER]
    assert "command-a-plus-05-2026" in grouped[COHERE_PROVIDER]
    assert "deepseek-v4-flash" in grouped[DEEPSEEK_PROVIDER]


def test_llm_factory_build_llm_for_stage_uses_conversation_model_config() -> None:
    from request_orchestrator.models.agent_state import AgentState

    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=MAIN_AGENT_MODEL_SCOPE,
                stage=REQUEST_ANALYSIS_STAGE,
                provider=OPENAI_PROVIDER,
                model='gpt-5.6-terra',
            ),
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=PROFILE_AGENT_MODEL_SCOPE,
                stage=PLANNER_STAGE,
                provider=OPENAI_PROVIDER,
                model='gpt-5.6-terra',
            ),
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=MAIN_AGENT_MODEL_SCOPE,
                stage=PLANNER_STAGE,
                provider=OPENAI_PROVIDER,
                model='gpt-5.6-luna',
            ),
        ]
    )

    with patch('llm.chat_models.ChatOpenAI', TrackingChatOpenAI):
        state = AgentState.new(
            task='Help me remember this.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
                model_config=config,
            ),
            agent_profile=MAIN_AGENT_PROFILE,
            llm=TrackingChatOpenAI(model='gpt-5.6-luna'),
        )

        request_analysis_llm = build_llm_for_stage(
            execution_context=state.execution_context,
            llm=state.llm,
            agent=MAIN_AGENT_MODEL_SCOPE,
            stage=REQUEST_ANALYSIS_STAGE,
        )
        assert request_analysis_llm.model == 'gpt-5.6-terra'

        profile_planner_llm = build_llm_for_stage(
            execution_context=state.execution_context,
            llm=state.llm,
            agent=PROFILE_AGENT_MODEL_SCOPE,
            stage=PLANNER_STAGE,
            reuse_llm_for_agent_scope=state.resolve_agent_scope(),
        )
        assert profile_planner_llm.model == 'gpt-5.6-terra'

        main_planner_llm = build_llm_for_stage(
            execution_context=state.execution_context,
            llm=state.llm,
            agent=MAIN_AGENT_MODEL_SCOPE,
            stage=PLANNER_STAGE,
            agent_profile=MAIN_AGENT_PROFILE,
            reuse_llm_for_agent_scope=state.resolve_agent_scope(),
        )
        assert main_planner_llm.model == 'gpt-5.6-luna'


def test_llm_factory_build_llm_for_stage_uses_anthropic_provider() -> None:
    from request_orchestrator.models.agent_state import AgentState

    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=MAIN_AGENT_MODEL_SCOPE,
                stage=REQUEST_ANALYSIS_STAGE,
                provider=ANTHROPIC_PROVIDER,
                model='claude-sonnet-5',
            ),
        ]
    )

    with patch('llm.chat_models.ChatAnthropic', TrackingChatAnthropic), patch(
        'request_orchestrator.models.agent_state.build_chat_model',
        side_effect=lambda provider, model_name: TrackingChatOpenAI(model=model_name),
    ):
        state = AgentState.new(
            task='Help me remember this.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
                model_config=config,
            ),
            agent_profile=MAIN_AGENT_PROFILE,
        )

        request_analysis_llm = build_llm_for_stage(
            execution_context=state.execution_context,
            llm=state.llm,
            agent=MAIN_AGENT_MODEL_SCOPE,
            stage=REQUEST_ANALYSIS_STAGE,
        )

    assert request_analysis_llm.model == 'claude-sonnet-5'


def test_llm_factory_build_llm_for_stage_uses_deepseek_openai_compatible_endpoint() -> None:
    from request_orchestrator.models.agent_state import AgentState

    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=MAIN_AGENT_MODEL_SCOPE,
                stage=REQUEST_ANALYSIS_STAGE,
                provider=DEEPSEEK_PROVIDER,
                model='deepseek-v4-flash',
            ),
        ]
    )

    with patch('llm.chat_models.ChatOpenAI', TrackingChatOpenAI), patch(
        'request_orchestrator.models.agent_state.build_chat_model',
        side_effect=lambda provider, model_name: TrackingChatOpenAI(model=model_name),
    ):
        state = AgentState.new(
            task='Help me remember this.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
                model_config=config,
            ),
            agent_profile=MAIN_AGENT_PROFILE,
        )

        request_analysis_llm = build_llm_for_stage(
            execution_context=state.execution_context,
            llm=state.llm,
            agent=MAIN_AGENT_MODEL_SCOPE,
            stage=REQUEST_ANALYSIS_STAGE,
        )

    assert request_analysis_llm.model == 'deepseek-v4-flash'
    assert request_analysis_llm.kwargs['base_url'] == 'https://api.deepseek.com'


def test_llm_factory_build_llm_for_stage_uses_google_openai_compatible_endpoint() -> None:
    from request_orchestrator.models.agent_state import AgentState

    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=MAIN_AGENT_MODEL_SCOPE,
                stage=REQUEST_ANALYSIS_STAGE,
                provider=GOOGLE_PROVIDER,
                model='gemini-3.5-flash',
            ),
        ]
    )

    with patch('llm.chat_models.ChatOpenAI', TrackingChatOpenAI), patch(
        'request_orchestrator.models.agent_state.build_chat_model',
        side_effect=lambda provider, model_name: TrackingChatOpenAI(model=model_name),
    ):
        state = AgentState.new(
            task='Help me remember this.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
                model_config=config,
            ),
            agent_profile=MAIN_AGENT_PROFILE,
        )

        request_analysis_llm = build_llm_for_stage(
            execution_context=state.execution_context,
            llm=state.llm,
            agent=MAIN_AGENT_MODEL_SCOPE,
            stage=REQUEST_ANALYSIS_STAGE,
        )

    assert request_analysis_llm.model == 'gemini-3.5-flash'
    assert (
        request_analysis_llm.kwargs['base_url']
        == 'https://generativelanguage.googleapis.com/v1beta/openai/'
    )


def test_llm_factory_build_llm_for_stage_uses_xai_openai_compatible_endpoint() -> None:
    from request_orchestrator.models.agent_state import AgentState

    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=MAIN_AGENT_MODEL_SCOPE,
                stage=REQUEST_ANALYSIS_STAGE,
                provider=XAI_PROVIDER,
                model='grok-4.5',
            ),
        ]
    )

    with patch('llm.chat_models.ChatOpenAI', TrackingChatOpenAI), patch(
        'request_orchestrator.models.agent_state.build_chat_model',
        side_effect=lambda provider, model_name: TrackingChatOpenAI(model=model_name),
    ):
        state = AgentState.new(
            task='Help me remember this.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
                model_config=config,
            ),
            agent_profile=MAIN_AGENT_PROFILE,
        )

        request_analysis_llm = build_llm_for_stage(
            execution_context=state.execution_context,
            llm=state.llm,
            agent=MAIN_AGENT_MODEL_SCOPE,
            stage=REQUEST_ANALYSIS_STAGE,
        )

    assert request_analysis_llm.model == 'grok-4.5'
    assert request_analysis_llm.kwargs['base_url'] == 'https://api.x.ai/v1'


def test_llm_factory_build_llm_for_stage_uses_cohere_openai_compatible_endpoint() -> None:
    from request_orchestrator.models.agent_state import AgentState

    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=MAIN_AGENT_MODEL_SCOPE,
                stage=REQUEST_ANALYSIS_STAGE,
                provider=COHERE_PROVIDER,
                model='command-a-plus-05-2026',
            ),
        ]
    )

    with patch('llm.chat_models.ChatOpenAI', TrackingChatOpenAI), patch(
        'request_orchestrator.models.agent_state.build_chat_model',
        side_effect=lambda provider, model_name: TrackingChatOpenAI(model=model_name),
    ):
        state = AgentState.new(
            task='Help me remember this.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
                model_config=config,
            ),
            agent_profile=MAIN_AGENT_PROFILE,
        )

        request_analysis_llm = build_llm_for_stage(
            execution_context=state.execution_context,
            llm=state.llm,
            agent=MAIN_AGENT_MODEL_SCOPE,
            stage=REQUEST_ANALYSIS_STAGE,
        )

    assert request_analysis_llm.model == 'command-a-plus-05-2026'
    assert request_analysis_llm.kwargs['base_url'] == 'https://api.cohere.ai/compatibility/v1'


def test_llm_factory_build_llm_for_stage_uses_mistral_openai_compatible_endpoint() -> None:
    from request_orchestrator.models.agent_state import AgentState

    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=MAIN_AGENT_MODEL_SCOPE,
                stage=REQUEST_ANALYSIS_STAGE,
                provider=MISTRAL_PROVIDER,
                model='mistral-small-latest',
            ),
        ]
    )

    with patch('llm.chat_models.ChatOpenAI', TrackingChatOpenAI), patch(
        'request_orchestrator.models.agent_state.build_chat_model',
        side_effect=lambda provider, model_name: TrackingChatOpenAI(model=model_name),
    ):
        state = AgentState.new(
            task='Help me remember this.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
                model_config=config,
            ),
            agent_profile=MAIN_AGENT_PROFILE,
        )

        request_analysis_llm = build_llm_for_stage(
            execution_context=state.execution_context,
            llm=state.llm,
            agent=MAIN_AGENT_MODEL_SCOPE,
            stage=REQUEST_ANALYSIS_STAGE,
        )

    assert request_analysis_llm.model == 'mistral-small-latest'
    assert request_analysis_llm.kwargs['base_url'] == 'https://api.mistral.ai/v1'


def test_candidate_reranker_uses_shared_conversation_model_from_runtime_context() -> None:
    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=SHARED_MODEL_SCOPE,
                stage=RERANKER_STAGE,
                provider=OPENAI_PROVIDER,
                model='gpt-5.6-terra',
            )
        ]
    )

    with patch('reranker.service.build_chat_model', side_effect=lambda provider, model_name: TrackingChatOpenAI(model=model_name)):
        with bind_runtime_context(conversation_id=str(conversation_id), conversation_model_config=config):
            reranker = CandidateReranker()

    assert reranker.llm.model == 'gpt-5.6-terra'


def test_run_request_orchestrator_records_resolved_model_config_snapshot() -> None:
    conversation_id = uuid4()
    config = ConversationModelConfig.build_default()
    fake_repo = FakeConversationRepository()
    fake_repo.conversation_id = conversation_id
    fake_model_config_repo = FakeConversationModelConfigRepository(config, conversation_id)
    captured_context_kwargs: dict[str, object] = {}

    def fake_build_roundtrip_context(conversation_id_arg, limit=5):
        captured_context_kwargs['conversation_id'] = conversation_id_arg
        captured_context_kwargs['limit'] = limit
        return ConversationContext()

    with patch('request_orchestrator.service.get_conversation_repo', return_value=fake_repo), patch(
        'request_orchestrator.service.get_conversation_model_config_repo',
        return_value=fake_model_config_repo,
    ), patch(
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

    assert fake_repo.pending_calls[0]['model'] == config.main_agent.planner.model
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
    fake_repo = FakeConversationRepository()
    fake_repo.conversation_id = conversation_id
    fake_model_config_repo = FakeConversationModelConfigRepository(config, conversation_id)
    captured_profile_kwargs: dict[str, object] = {}

    def fake_build_user_profile(**kwargs):
        captured_profile_kwargs.update(kwargs)
        return UserProfile(
            user_id='anonymous',
            geometadata=kwargs.get('geometadata'),
        )

    with patch('request_orchestrator.service.get_conversation_repo', return_value=fake_repo), patch(
        'request_orchestrator.service.get_conversation_model_config_repo',
        return_value=fake_model_config_repo,
    ), patch(
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
        provider=OPENAI_PROVIDER,
        model='gpt-5.6-luna',
    )
    reranker_override = ConversationModelConfigEntry(
        conversation_id=conversation_id,
        agent=SHARED_MODEL_SCOPE,
        stage=RERANKER_STAGE,
        provider=OPENAI_PROVIDER,
        model='gpt-5.6-terra',
    )
    resolved = resolve_conversation_model_config([evaluator_override, reranker_override])

    rows = build_model_config_rows(resolved, [evaluator_override, reranker_override])
    evaluator_row = next(row for row in rows if row['agent'] == SHARED_MODEL_SCOPE and row['stage'] == EVALUATOR_STAGE)
    reranker_row = next(row for row in rows if row['agent'] == SHARED_MODEL_SCOPE and row['stage'] == RERANKER_STAGE)

    assert evaluator_row['effective_model'] == 'gpt-5.6-luna'
    assert evaluator_row['override_model'] == 'gpt-5.6-luna'
    assert evaluator_row['override_provider'] == OPENAI_PROVIDER
    assert evaluator_row['input_price'] == '$1 per 1M'
    assert evaluator_row['output_price'] == '$6 per 1M'
    assert evaluator_row['effective_provider'] == 'OpenAI'
    assert OPENAI_PROVIDER in evaluator_row['provider_options']
    assert reranker_row['effective_model'] == 'gpt-5.6-terra'
    assert reranker_row['override_model'] == 'gpt-5.6-terra'
    assert reranker_row['input_price'] == '$2.5 per 1M'
    assert reranker_row['output_price'] == '$15 per 1M'


def test_build_model_config_rows_reset_to_default_restores_model_and_pricing() -> None:
    resolved = ConversationModelConfig.build_default()

    rows = build_model_config_rows(resolved, [])
    request_analysis_row = next(row for row in rows if row['agent'] == MAIN_AGENT_MODEL_SCOPE and row['stage'] == REQUEST_ANALYSIS_STAGE)

    assert request_analysis_row['effective_model'] == 'gpt-5.6-luna'
    assert request_analysis_row['override_model'] is None
    assert request_analysis_row['input_price'] == '$1 per 1M'
    assert request_analysis_row['output_price'] == '$6 per 1M'


def test_build_model_config_rows_exposes_provider_filtered_options() -> None:
    rows = build_model_config_rows(ConversationModelConfig.build_default(), [])
    synthesis_row = next(
        row
        for row in rows
        if row['agent'] == MAIN_AGENT_MODEL_SCOPE and row['stage'] == SYNTHESIS_STAGE
    )

    assert synthesis_row['effective_provider'] == 'OpenAI'
    assert synthesis_row['effective_model_option'] in synthesis_row['provider_options'][OPENAI_PROVIDER]
