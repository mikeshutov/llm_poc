from __future__ import annotations

from decimal import Decimal
from uuid import uuid4
from unittest.mock import patch


from conversation.models.conversation_models import ConversationContext
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
    SHARED_MODEL_SCOPE,
    SYNTHESIS_STAGE,
    XAI_PROVIDER,
)
from llm.model_config_resolver import resolve_conversation_model_config
from personalization.profile.models import UserProfile
from rendering.sidebar import build_model_config_rows
from agents.main.request import TOP_LEVEL_PROFILE
from agent_runtime.models.agent_execution_context import AgentExecutionContext
from agent_runtime.models.agent_result import AgentResult
from agent_runtime.models.orchestrator_result import OrchestratorResult
from agent_runtime.service import run_request_orchestrator_for_query


def test_conversation_model_config_resolves_partial_overrides_with_defaults() -> None:
    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=SHARED_MODEL_SCOPE,
                stage=EVALUATOR_STAGE,
                provider=OPENAI_PROVIDER,
                model='gpt-5.6-luna',
            ),
        ]
    )

    assert config.main_agent.planner.model == 'gpt-5.6-luna'
    assert config.main_agent.synthesis.model == 'gpt-5.6-luna'
    assert config.profile_agent.planner.model == 'gpt-5.6-luna'
    assert config.shared.evaluator.model == 'gpt-5.6-luna'


def test_conversation_model_config_build_default_returns_defaults() -> None:
    config = ConversationModelConfig.build_default()

    assert config.main_agent.planner.model == 'gpt-5.6-luna'
    assert config.main_agent.synthesis.model == 'gpt-5.6-luna'
    assert config.profile_agent.planner.model == 'gpt-5.6-luna'
    assert config.shared.evaluator.model == 'gpt-5.6-luna'


def test_conversation_model_config_build_default_resolves_pricing_for_every_stage() -> None:
    config = ConversationModelConfig.build_default()

    assert config.resolve_pricing(MAIN_AGENT_MODEL_SCOPE, PLANNER_STAGE) == ModelPricing(
        input_price_per_million_tokens=Decimal('0.20'),
        cached_input_price_per_million_tokens=Decimal('0.02'),
        output_price_per_million_tokens=Decimal('1.20'),
    )
    assert config.resolve_pricing(PROFILE_AGENT_MODEL_SCOPE, PLANNER_STAGE) == ModelPricing(
        input_price_per_million_tokens=Decimal('0.20'),
        cached_input_price_per_million_tokens=Decimal('0.02'),
        output_price_per_million_tokens=Decimal('1.20'),
    )
    assert config.resolve_pricing(SHARED_MODEL_SCOPE, EVALUATOR_STAGE) == ModelPricing(
        input_price_per_million_tokens=Decimal('0.20'),
        cached_input_price_per_million_tokens=Decimal('0.02'),
        output_price_per_million_tokens=Decimal('1.20'),
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
        input_price_per_million_tokens=Decimal('0.20'),
        cached_input_price_per_million_tokens=Decimal('0.02'),
        output_price_per_million_tokens=Decimal('1.20'),
    )
    assert config.resolve_pricing(MAIN_AGENT_MODEL_SCOPE, PLANNER_STAGE) == ModelPricing(
        input_price_per_million_tokens=Decimal('0.20'),
        cached_input_price_per_million_tokens=Decimal('0.02'),
        output_price_per_million_tokens=Decimal('1.20'),
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


def test_llm_factory_build_llm_for_stage_uses_conversation_model_config(tracking_openai_model_factory) -> None:
    from agent_runtime.models.agent_state import AgentState

    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=MAIN_AGENT_MODEL_SCOPE,
                stage=PLANNER_STAGE,
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

    with patch('llm.chat_models.ChatOpenAI', tracking_openai_model_factory):
        state = AgentState.new(
            task='Help me remember this.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
                model_config=config,
            ),
            agent_profile=TOP_LEVEL_PROFILE,
            llm=tracking_openai_model_factory(model='gpt-5.6-luna'),
        )

        planner_llm = build_llm_for_stage(
            execution_context=state.execution_context,
            llm=state.llm,
            agent=MAIN_AGENT_MODEL_SCOPE,
            stage=PLANNER_STAGE,
        )

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
            agent_profile=TOP_LEVEL_PROFILE,
            reuse_llm_for_agent_scope=state.resolve_agent_scope(),
        )
        assert main_planner_llm.model == 'gpt-5.6-luna'


def test_llm_factory_build_llm_for_stage_uses_anthropic_provider(tracking_anthropic_model_factory, tracking_openai_model_factory) -> None:
    from agent_runtime.models.agent_state import AgentState

    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=MAIN_AGENT_MODEL_SCOPE,
                stage=PLANNER_STAGE,
                provider=ANTHROPIC_PROVIDER,
                model='claude-sonnet-5',
            ),
        ]
    )

    with patch('llm.chat_models.ChatAnthropic', tracking_anthropic_model_factory), patch(
        'agent_runtime.models.agent_state.build_chat_model',
        side_effect=lambda provider, model_name: tracking_openai_model_factory(model=model_name),
    ):
        state = AgentState.new(
            task='Help me remember this.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
                model_config=config,
            ),
            agent_profile=TOP_LEVEL_PROFILE,
        )

        planner_llm = build_llm_for_stage(
            execution_context=state.execution_context,
            llm=state.llm,
            agent=MAIN_AGENT_MODEL_SCOPE,
            stage=PLANNER_STAGE,
        )

    assert planner_llm.model == 'claude-sonnet-5'


def test_llm_factory_build_llm_for_stage_uses_deepseek_openai_compatible_endpoint(tracking_openai_model_factory) -> None:
    from agent_runtime.models.agent_state import AgentState

    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=MAIN_AGENT_MODEL_SCOPE,
                stage=PLANNER_STAGE,
                provider=DEEPSEEK_PROVIDER,
                model='deepseek-v4-flash',
            ),
        ]
    )

    with patch('llm.chat_models.ChatOpenAI', tracking_openai_model_factory), patch(
        'agent_runtime.models.agent_state.build_chat_model',
        side_effect=lambda provider, model_name: tracking_openai_model_factory(model=model_name),
    ):
        state = AgentState.new(
            task='Help me remember this.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
                model_config=config,
            ),
            agent_profile=TOP_LEVEL_PROFILE,
        )

        planner_llm = build_llm_for_stage(
            execution_context=state.execution_context,
            llm=state.llm,
            agent=MAIN_AGENT_MODEL_SCOPE,
            stage=PLANNER_STAGE,
        )

    assert planner_llm.model == 'deepseek-v4-flash'
    assert planner_llm.kwargs['base_url'] == 'https://api.deepseek.com'


def test_llm_factory_build_llm_for_stage_uses_google_openai_compatible_endpoint(tracking_openai_model_factory) -> None:
    from agent_runtime.models.agent_state import AgentState

    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=MAIN_AGENT_MODEL_SCOPE,
                stage=PLANNER_STAGE,
                provider=GOOGLE_PROVIDER,
                model='gemini-3.5-flash',
            ),
        ]
    )

    with patch('llm.chat_models.ChatOpenAI', tracking_openai_model_factory), patch(
        'agent_runtime.models.agent_state.build_chat_model',
        side_effect=lambda provider, model_name: tracking_openai_model_factory(model=model_name),
    ):
        state = AgentState.new(
            task='Help me remember this.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
                model_config=config,
            ),
            agent_profile=TOP_LEVEL_PROFILE,
        )

        planner_llm = build_llm_for_stage(
            execution_context=state.execution_context,
            llm=state.llm,
            agent=MAIN_AGENT_MODEL_SCOPE,
            stage=PLANNER_STAGE,
        )

    assert planner_llm.model == 'gemini-3.5-flash'
    assert (
        planner_llm.kwargs['base_url']
        == 'https://generativelanguage.googleapis.com/v1beta/openai/'
    )


def test_llm_factory_build_llm_for_stage_uses_xai_openai_compatible_endpoint(tracking_openai_model_factory) -> None:
    from agent_runtime.models.agent_state import AgentState

    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=MAIN_AGENT_MODEL_SCOPE,
                stage=PLANNER_STAGE,
                provider=XAI_PROVIDER,
                model='grok-4.5',
            ),
        ]
    )

    with patch('llm.chat_models.ChatOpenAI', tracking_openai_model_factory), patch(
        'agent_runtime.models.agent_state.build_chat_model',
        side_effect=lambda provider, model_name: tracking_openai_model_factory(model=model_name),
    ):
        state = AgentState.new(
            task='Help me remember this.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
                model_config=config,
            ),
            agent_profile=TOP_LEVEL_PROFILE,
        )

        planner_llm = build_llm_for_stage(
            execution_context=state.execution_context,
            llm=state.llm,
            agent=MAIN_AGENT_MODEL_SCOPE,
            stage=PLANNER_STAGE,
        )

    assert planner_llm.model == 'grok-4.5'
    assert planner_llm.kwargs['base_url'] == 'https://api.x.ai/v1'


def test_llm_factory_build_llm_for_stage_uses_cohere_openai_compatible_endpoint(tracking_openai_model_factory) -> None:
    from agent_runtime.models.agent_state import AgentState

    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=MAIN_AGENT_MODEL_SCOPE,
                stage=PLANNER_STAGE,
                provider=COHERE_PROVIDER,
                model='command-a-plus-05-2026',
            ),
        ]
    )

    with patch('llm.chat_models.ChatOpenAI', tracking_openai_model_factory), patch(
        'agent_runtime.models.agent_state.build_chat_model',
        side_effect=lambda provider, model_name: tracking_openai_model_factory(model=model_name),
    ):
        state = AgentState.new(
            task='Help me remember this.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
                model_config=config,
            ),
            agent_profile=TOP_LEVEL_PROFILE,
        )

        planner_llm = build_llm_for_stage(
            execution_context=state.execution_context,
            llm=state.llm,
            agent=MAIN_AGENT_MODEL_SCOPE,
            stage=PLANNER_STAGE,
        )

    assert planner_llm.model == 'command-a-plus-05-2026'
    assert planner_llm.kwargs['base_url'] == 'https://api.cohere.ai/compatibility/v1'


def test_llm_factory_build_llm_for_stage_uses_mistral_openai_compatible_endpoint(tracking_openai_model_factory) -> None:
    from agent_runtime.models.agent_state import AgentState

    conversation_id = uuid4()
    config = resolve_conversation_model_config(
        [
            ConversationModelConfigEntry(
                conversation_id=conversation_id,
                agent=MAIN_AGENT_MODEL_SCOPE,
                stage=PLANNER_STAGE,
                provider=MISTRAL_PROVIDER,
                model='mistral-small-latest',
            ),
        ]
    )

    with patch('llm.chat_models.ChatOpenAI', tracking_openai_model_factory), patch(
        'agent_runtime.models.agent_state.build_chat_model',
        side_effect=lambda provider, model_name: tracking_openai_model_factory(model=model_name),
    ):
        state = AgentState.new(
            task='Help me remember this.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
                model_config=config,
            ),
            agent_profile=TOP_LEVEL_PROFILE,
        )

        planner_llm = build_llm_for_stage(
            execution_context=state.execution_context,
            llm=state.llm,
            agent=MAIN_AGENT_MODEL_SCOPE,
            stage=PLANNER_STAGE,
        )

    assert planner_llm.model == 'mistral-small-latest'
    assert planner_llm.kwargs['base_url'] == 'https://api.mistral.ai/v1'


def test_run_request_orchestrator_records_resolved_model_config_snapshot(conversation_model_config_repository_factory, request_conversation_repository_factory) -> None:
    conversation_id = uuid4()
    config = ConversationModelConfig.build_default()
    fake_repo = request_conversation_repository_factory()
    fake_repo.conversation_id = conversation_id
    fake_model_config_repo = conversation_model_config_repository_factory(config, conversation_id)
    captured_context_kwargs: dict[str, object] = {}

    def fake_build_roundtrip_context(conversation_id_arg, limit=5):
        captured_context_kwargs['conversation_id'] = conversation_id_arg
        captured_context_kwargs['limit'] = limit
        return ConversationContext()

    with patch('agent_runtime.service.get_conversation_repo', return_value=fake_repo), patch(
        'agent_runtime.service.get_conversation_model_config_repo',
        return_value=fake_model_config_repo,
    ), patch(
        'agent_runtime.service.build_roundtrip_context',
        side_effect=fake_build_roundtrip_context,
    ), patch(
        'agent_runtime.service.build_user_profile',
        return_value=UserProfile(),
    ), patch(
        'agent_runtime.service.MainState.initialize_agent_states',
        return_value=None,
    ), patch(
        'agent_runtime.service.run_agent',
        return_value=OrchestratorResult(agent_result=AgentResult(), answer=['done']),
    ), patch(
        'agent_runtime.service.embed_text',
        return_value=[0.1, 0.2, 0.3],
    ):
        run_request_orchestrator_for_query(str(conversation_id), 'hello world', user_id='anonymous')

    assert fake_repo.pending_calls[0]['model'] == config.main_agent.planner.model
    assert fake_repo.pending_calls[0]['metadata']['resolved_model_config'] == config.model_dump(mode='json')
    assert captured_context_kwargs == {
        'conversation_id': str(conversation_id),
        'limit': 5,
    }
    assert 'llm_usage' not in fake_repo.update_calls[0]['payload']
    assert 'agent_logs' not in fake_repo.update_calls[0]['payload']
    assert isinstance(fake_repo.update_calls[0]['payload'].roundtrip_latency_ms, int)
    assert fake_repo.update_calls[0]['payload'].roundtrip_latency_ms >= 0


def test_run_request_orchestrator_passes_geometadata_to_user_profile_builder(conversation_model_config_repository_factory, request_conversation_repository_factory) -> None:
    conversation_id = uuid4()
    config = ConversationModelConfig.build_default()
    fake_repo = request_conversation_repository_factory()
    fake_repo.conversation_id = conversation_id
    fake_model_config_repo = conversation_model_config_repository_factory(config, conversation_id)
    captured_profile_kwargs: dict[str, object] = {}

    def fake_build_user_profile(**kwargs):
        captured_profile_kwargs.update(kwargs)
        return UserProfile(
            user_id='anonymous',
            geometadata=kwargs.get('geometadata'),
        )

    with patch('agent_runtime.service.get_conversation_repo', return_value=fake_repo), patch(
        'agent_runtime.service.get_conversation_model_config_repo',
        return_value=fake_model_config_repo,
    ), patch(
        'agent_runtime.service.build_roundtrip_context',
        return_value=ConversationContext(),
    ), patch(
        'agent_runtime.service.build_user_profile',
        side_effect=fake_build_user_profile,
    ), patch(
        'agent_runtime.service.MainState.initialize_agent_states',
        return_value=None,
    ), patch(
        'agent_runtime.service.run_agent',
        return_value=OrchestratorResult(agent_result=AgentResult(), answer=['done']),
    ), patch(
        'agent_runtime.service.embed_text',
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
    resolved = resolve_conversation_model_config([evaluator_override])

    rows = build_model_config_rows(resolved, [evaluator_override])
    evaluator_row = next(row for row in rows if row['agent'] == SHARED_MODEL_SCOPE and row['stage'] == EVALUATOR_STAGE)

    assert evaluator_row['effective_model'] == 'gpt-5.6-luna'
    assert evaluator_row['override_model'] == 'gpt-5.6-luna'
    assert evaluator_row['override_provider'] == OPENAI_PROVIDER
    assert evaluator_row['input_price'] == '$0.2 per 1M'
    assert evaluator_row['output_price'] == '$1.2 per 1M'
    assert evaluator_row['effective_provider'] == 'OpenAI'
    assert OPENAI_PROVIDER in evaluator_row['provider_options']


def test_build_model_config_rows_reset_to_default_restores_model_and_pricing() -> None:
    resolved = ConversationModelConfig.build_default()

    rows = build_model_config_rows(resolved, [])
    planner_row = next(row for row in rows if row['agent'] == MAIN_AGENT_MODEL_SCOPE and row['stage'] == PLANNER_STAGE)

    assert planner_row['effective_model'] == 'gpt-5.6-luna'
    assert planner_row['override_model'] is None
    assert planner_row['input_price'] == '$0.2 per 1M'
    assert planner_row['output_price'] == '$1.2 per 1M'


def test_build_model_config_rows_exposes_provider_filtered_options() -> None:
    rows = build_model_config_rows(ConversationModelConfig.build_default(), [])
    synthesis_row = next(
        row
        for row in rows
        if row['agent'] == MAIN_AGENT_MODEL_SCOPE and row['stage'] == SYNTHESIS_STAGE
    )

    assert synthesis_row['effective_provider'] == 'OpenAI'
    assert synthesis_row['effective_model_option'] in synthesis_row['provider_options'][OPENAI_PROVIDER]
