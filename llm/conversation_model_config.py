from __future__ import annotations

import os
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel

MAIN_AGENT_MODEL_SCOPE = "main_agent"
PROFILE_AGENT_MODEL_SCOPE = "profile_agent"
SHARED_MODEL_SCOPE = "shared"

REQUEST_ANALYSIS_STAGE = "request_analysis"
PLANNER_STAGE = "planner"
SYNTHESIS_STAGE = "synthesis"
EVALUATOR_STAGE = "evaluator"
RERANKER_STAGE = "reranker"

OPENAI_PROVIDER = "openai"
ANTHROPIC_PROVIDER = "anthropic"
GOOGLE_PROVIDER = "google"
XAI_PROVIDER = "xai"
MISTRAL_PROVIDER = "mistral"
COHERE_PROVIDER = "cohere"
DEEPSEEK_PROVIDER = "deepseek"


def _configured_model(environment_variable: str, default: str) -> str:
    return os.getenv(environment_variable, "").strip() or default


DEFAULT_MINI_MODEL = "gpt-5.6-luna"
DEFAULT_MAIN_AGENT_MODEL = _configured_model("LLM_MODEL", DEFAULT_MINI_MODEL)
DEFAULT_MAIN_AGENT_REQUEST_ANALYSIS_MODEL = DEFAULT_MINI_MODEL
DEFAULT_MAIN_AGENT_PLANNER_MODEL = _configured_model("MAIN_AGENT_PLANNER_MODEL", DEFAULT_MAIN_AGENT_MODEL)
DEFAULT_MAIN_AGENT_SYNTHESIS_MODEL = _configured_model("MAIN_AGENT_SYNTHESIS_MODEL", DEFAULT_MAIN_AGENT_MODEL)
DEFAULT_PROFILE_AGENT_PLANNER_MODEL = _configured_model("PROFILE_AGENT_PLANNER_MODEL", DEFAULT_MINI_MODEL)
DEFAULT_SHARED_EVALUATOR_MODEL = DEFAULT_MINI_MODEL
DEFAULT_SHARED_RERANKER_MODEL = DEFAULT_MINI_MODEL


class ModelPricing(BaseModel):
    input_price_per_million_tokens: Decimal
    output_price_per_million_tokens: Decimal
    cached_input_price_per_million_tokens: Decimal | None = None


class ModelSelection(BaseModel):
    provider: str
    model: str


@dataclass(frozen=True)
class ConversationModelConfigSpec:
    agent: str
    stage: str
    label: str
    default_provider: str
    default_model: str

    @property
    def path(self) -> str:
        return f"{self.agent}.{self.stage}"


CONVERSATION_MODEL_CONFIG_SPECS: tuple[ConversationModelConfigSpec, ...] = (
    ConversationModelConfigSpec(
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=REQUEST_ANALYSIS_STAGE,
        label="MainAgent / request_analysis",
        default_provider=OPENAI_PROVIDER,
        default_model=DEFAULT_MAIN_AGENT_REQUEST_ANALYSIS_MODEL,
    ),
    ConversationModelConfigSpec(
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=PLANNER_STAGE,
        label="MainAgent / planner",
        default_provider=OPENAI_PROVIDER,
        default_model=DEFAULT_MAIN_AGENT_PLANNER_MODEL,
    ),
    ConversationModelConfigSpec(
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=SYNTHESIS_STAGE,
        label="MainAgent / synthesis",
        default_provider=OPENAI_PROVIDER,
        default_model=DEFAULT_MAIN_AGENT_SYNTHESIS_MODEL,
    ),
    ConversationModelConfigSpec(
        agent=PROFILE_AGENT_MODEL_SCOPE,
        stage=PLANNER_STAGE,
        label="ProfileAgent / planner",
        default_provider=OPENAI_PROVIDER,
        default_model=DEFAULT_PROFILE_AGENT_PLANNER_MODEL,
    ),
    ConversationModelConfigSpec(
        agent=SHARED_MODEL_SCOPE,
        stage=EVALUATOR_STAGE,
        label="Shared / evaluator",
        default_provider=OPENAI_PROVIDER,
        default_model=DEFAULT_SHARED_EVALUATOR_MODEL,
    ),
    ConversationModelConfigSpec(
        agent=SHARED_MODEL_SCOPE,
        stage=RERANKER_STAGE,
        label="Shared / reranker",
        default_provider=OPENAI_PROVIDER,
        default_model=DEFAULT_SHARED_RERANKER_MODEL,
    ),
)


def get_model_config_spec(agent: str, stage: str) -> ConversationModelConfigSpec:
    for spec in CONVERSATION_MODEL_CONFIG_SPECS:
        if spec.agent == agent and spec.stage == stage:
            return spec
    raise KeyError(f"Unsupported conversation model config key: {agent}.{stage}")


@dataclass(frozen=False)
class ConversationModelConfigEntry:
    conversation_id: UUID
    agent: str
    stage: str
    provider: str
    model: str
    created_at: str = ""
    updated_at: str = ""


class MainAgentConversationModelConfig(BaseModel):
    request_analysis: ModelSelection
    planner: ModelSelection
    synthesis: ModelSelection


class ProfileAgentConversationModelConfig(BaseModel):
    planner: ModelSelection


class SharedConversationModelConfig(BaseModel):
    evaluator: ModelSelection
    reranker: ModelSelection


class ConversationModelConfig(BaseModel):
    SNAPSHOT_MODEL_SUFFIX_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^(?P<base>.+)-\d{4}-\d{2}-\d{2}$")
    PROVIDER_DISPLAY_NAMES: ClassVar[dict[str, str]] = {
        OPENAI_PROVIDER: "OpenAI",
        ANTHROPIC_PROVIDER: "Anthropic",
        GOOGLE_PROVIDER: "Google",
        XAI_PROVIDER: "xAI",
        MISTRAL_PROVIDER: "Mistral",
        COHERE_PROVIDER: "Cohere",
        DEEPSEEK_PROVIDER: "DeepSeek",
    }
    MODEL_PRICING_REGISTRY: ClassVar[dict[str, dict[str, ModelPricing]]] = {
        OPENAI_PROVIDER: {
            "gpt-5.6": ModelPricing(input_price_per_million_tokens=Decimal("5.00"), cached_input_price_per_million_tokens=Decimal("0.50"), output_price_per_million_tokens=Decimal("30.00")),
            "gpt-5.6-sol": ModelPricing(input_price_per_million_tokens=Decimal("5.00"), cached_input_price_per_million_tokens=Decimal("0.50"), output_price_per_million_tokens=Decimal("30.00")),
            "gpt-5.6-terra": ModelPricing(input_price_per_million_tokens=Decimal("2.50"), cached_input_price_per_million_tokens=Decimal("0.25"), output_price_per_million_tokens=Decimal("15.00")),
            "gpt-5.6-luna": ModelPricing(input_price_per_million_tokens=Decimal("1.00"), cached_input_price_per_million_tokens=Decimal("0.10"), output_price_per_million_tokens=Decimal("6.00")),
            "o3": ModelPricing(input_price_per_million_tokens=Decimal("2.00"), output_price_per_million_tokens=Decimal("8.00")),
            "o3-pro": ModelPricing(input_price_per_million_tokens=Decimal("20.00"), output_price_per_million_tokens=Decimal("80.00")),
            "o3-mini": ModelPricing(input_price_per_million_tokens=Decimal("1.10"), output_price_per_million_tokens=Decimal("4.40")),
            "o4-mini": ModelPricing(input_price_per_million_tokens=Decimal("1.10"), output_price_per_million_tokens=Decimal("4.40")),
            "o1": ModelPricing(input_price_per_million_tokens=Decimal("15.00"), output_price_per_million_tokens=Decimal("60.00")),
            "o1-pro": ModelPricing(input_price_per_million_tokens=Decimal("150.00"), output_price_per_million_tokens=Decimal("600.00")),
        },
        ANTHROPIC_PROVIDER: {
            "claude-haiku-4-5": ModelPricing(input_price_per_million_tokens=Decimal("1.00"), output_price_per_million_tokens=Decimal("5.00")),
            "claude-sonnet-5": ModelPricing(input_price_per_million_tokens=Decimal("2.00"), output_price_per_million_tokens=Decimal("10.00")),
            "claude-opus-4-8": ModelPricing(input_price_per_million_tokens=Decimal("5.00"), output_price_per_million_tokens=Decimal("25.00")),
            "claude-fable-5": ModelPricing(input_price_per_million_tokens=Decimal("10.00"), output_price_per_million_tokens=Decimal("50.00")),
        },
        GOOGLE_PROVIDER: {
            "gemini-3.5-flash": ModelPricing(input_price_per_million_tokens=Decimal("1.50"), output_price_per_million_tokens=Decimal("9.00")),
            "gemini-3.5-flash-lite": ModelPricing(input_price_per_million_tokens=Decimal("0.30"), output_price_per_million_tokens=Decimal("2.50")),
        },
        XAI_PROVIDER: {
            "grok-4.20": ModelPricing(input_price_per_million_tokens=Decimal("1.25"), output_price_per_million_tokens=Decimal("2.50")),
            "grok-4.5": ModelPricing(input_price_per_million_tokens=Decimal("2.00"), output_price_per_million_tokens=Decimal("6.00")),
        },
        MISTRAL_PROVIDER: {
            "mistral-small-latest": ModelPricing(input_price_per_million_tokens=Decimal("0.15"), output_price_per_million_tokens=Decimal("0.60")),
            "mistral-large-latest": ModelPricing(input_price_per_million_tokens=Decimal("0.50"), output_price_per_million_tokens=Decimal("1.50")),
            "mistral-medium-latest": ModelPricing(input_price_per_million_tokens=Decimal("1.50"), output_price_per_million_tokens=Decimal("7.50")),
        },
        COHERE_PROVIDER: {
            "command-a-plus-05-2026": ModelPricing(input_price_per_million_tokens=Decimal("2.50"), output_price_per_million_tokens=Decimal("10.00")),
        },
        DEEPSEEK_PROVIDER: {
            "deepseek-v4-flash": ModelPricing(input_price_per_million_tokens=Decimal("0.14"), output_price_per_million_tokens=Decimal("0.28")),
            "deepseek-v4-pro": ModelPricing(input_price_per_million_tokens=Decimal("0.435"), output_price_per_million_tokens=Decimal("0.87")),
        },
    }

    main_agent: MainAgentConversationModelConfig
    profile_agent: ProfileAgentConversationModelConfig
    shared: SharedConversationModelConfig

    @classmethod
    def build_default(cls) -> ConversationModelConfig:
        return cls(
            main_agent=MainAgentConversationModelConfig(
                request_analysis=ModelSelection(provider=OPENAI_PROVIDER, model=DEFAULT_MAIN_AGENT_REQUEST_ANALYSIS_MODEL),
                planner=ModelSelection(provider=OPENAI_PROVIDER, model=DEFAULT_MAIN_AGENT_PLANNER_MODEL),
                synthesis=ModelSelection(provider=OPENAI_PROVIDER, model=DEFAULT_MAIN_AGENT_SYNTHESIS_MODEL),
            ),
            profile_agent=ProfileAgentConversationModelConfig(
                planner=ModelSelection(provider=OPENAI_PROVIDER, model=DEFAULT_PROFILE_AGENT_PLANNER_MODEL),
            ),
            shared=SharedConversationModelConfig(
                evaluator=ModelSelection(provider=OPENAI_PROVIDER, model=DEFAULT_SHARED_EVALUATOR_MODEL),
                reranker=ModelSelection(provider=OPENAI_PROVIDER, model=DEFAULT_SHARED_RERANKER_MODEL),
            ),
        )

    @classmethod
    def default_main_agent_planner_model(cls) -> str:
        return DEFAULT_MAIN_AGENT_PLANNER_MODEL

    @classmethod
    def default_shared_reranker_model(cls) -> str:
        return DEFAULT_SHARED_RERANKER_MODEL

    @classmethod
    def provider_display_name(cls, provider: str) -> str:
        return cls.PROVIDER_DISPLAY_NAMES.get(provider, provider)

    @classmethod
    def normalize_model_pricing_key(cls, model_name: str) -> str:
        normalized_name = model_name.strip()
        match = cls.SNAPSHOT_MODEL_SUFFIX_PATTERN.match(normalized_name)
        if match:
            return match.group("base")
        return normalized_name

    @classmethod
    def resolve_model_pricing(cls, provider: str, model_name: str | None = None) -> ModelPricing:
        if model_name is None:
            model_name = provider
            provider = OPENAI_PROVIDER
        provider_registry = cls.MODEL_PRICING_REGISTRY.get(provider, {})
        pricing = provider_registry.get(model_name)
        if pricing is None:
            pricing = provider_registry.get(cls.normalize_model_pricing_key(model_name))
        if pricing is None:
            raise KeyError(f"Unsupported model pricing key: {provider}.{model_name}")
        return pricing

    @classmethod
    def model_names_by_provider(cls) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for provider, provider_registry in cls.MODEL_PRICING_REGISTRY.items():
            grouped[provider] = sorted(provider_registry.keys())
        return dict(sorted(grouped.items(), key=lambda item: cls.provider_display_name(item[0])))

    def _config_for_agent(self, agent: str) -> MainAgentConversationModelConfig | ProfileAgentConversationModelConfig | SharedConversationModelConfig:
        if agent == MAIN_AGENT_MODEL_SCOPE:
            return self.main_agent
        if agent == PROFILE_AGENT_MODEL_SCOPE:
            return self.profile_agent
        if agent == SHARED_MODEL_SCOPE:
            return self.shared
        raise KeyError(f"Unsupported conversation model config key: {agent}")

    def set_value(self, agent: str, stage: str, provider: str, model: str) -> None:
        get_model_config_spec(agent, stage)
        setattr(
            self._config_for_agent(agent),
            stage,
            ModelSelection(provider=provider, model=model),
        )

    def resolve_selection(self, agent: str, stage: str) -> ModelSelection:
        get_model_config_spec(agent, stage)
        selection = getattr(self._config_for_agent(agent), stage)
        if isinstance(selection, ModelSelection):
            return selection
        if isinstance(selection, str):
            return ModelSelection(provider=OPENAI_PROVIDER, model=selection)
        raise TypeError(f"Unsupported model selection value for {agent}.{stage}: {type(selection)!r}")

    def resolve(self, agent: str, stage: str) -> str:
        return self.resolve_selection(agent, stage).model

    def resolve_provider(self, agent: str, stage: str) -> str:
        return self.resolve_selection(agent, stage).provider

    def resolve_pricing(self, agent: str, stage: str) -> ModelPricing:
        selection = self.resolve_selection(agent, stage)
        return self.resolve_model_pricing(selection.provider, selection.model)

    def to_flat_dict(self) -> dict[str, str]:
        return {
            spec.path: self.resolve(spec.agent, spec.stage)
            for spec in CONVERSATION_MODEL_CONFIG_SPECS
        }

    def to_metadata_payload(self) -> dict[str, Any]:
        return self.model_dump()
