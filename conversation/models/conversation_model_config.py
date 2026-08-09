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

DEFAULT_MINI_MODEL = "gpt-5.4-mini"
DEFAULT_MAIN_AGENT_MODEL = os.getenv("LLM_MODEL", "gpt-5.4")
DEFAULT_MAIN_AGENT_REQUEST_ANALYSIS_MODEL = DEFAULT_MINI_MODEL
DEFAULT_MAIN_AGENT_PLANNER_MODEL = os.getenv("MAIN_AGENT_PLANNER_MODEL", DEFAULT_MAIN_AGENT_MODEL)
DEFAULT_MAIN_AGENT_SYNTHESIS_MODEL = os.getenv("MAIN_AGENT_SYNTHESIS_MODEL", DEFAULT_MAIN_AGENT_MODEL)
DEFAULT_PROFILE_AGENT_PLANNER_MODEL = DEFAULT_MINI_MODEL
DEFAULT_SHARED_EVALUATOR_MODEL = DEFAULT_MINI_MODEL
DEFAULT_SHARED_RERANKER_MODEL = DEFAULT_MINI_MODEL


class ModelPricing(BaseModel):
    input_price_per_million_tokens: Decimal
    output_price_per_million_tokens: Decimal


@dataclass(frozen=True)
class ConversationModelConfigSpec:
    agent: str
    stage: str
    label: str
    default_model: str

    @property
    def path(self) -> str:
        return f"{self.agent}.{self.stage}"


CONVERSATION_MODEL_CONFIG_SPECS: tuple[ConversationModelConfigSpec, ...] = (
    ConversationModelConfigSpec(
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=REQUEST_ANALYSIS_STAGE,
        label="MainAgent / request_analysis",
        default_model=DEFAULT_MAIN_AGENT_REQUEST_ANALYSIS_MODEL,
    ),
    ConversationModelConfigSpec(
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=PLANNER_STAGE,
        label="MainAgent / planner",
        default_model=DEFAULT_MAIN_AGENT_PLANNER_MODEL,
    ),
    ConversationModelConfigSpec(
        agent=MAIN_AGENT_MODEL_SCOPE,
        stage=SYNTHESIS_STAGE,
        label="MainAgent / synthesis",
        default_model=DEFAULT_MAIN_AGENT_SYNTHESIS_MODEL,
    ),
    ConversationModelConfigSpec(
        agent=PROFILE_AGENT_MODEL_SCOPE,
        stage=PLANNER_STAGE,
        label="ProfileAgent / planner",
        default_model=DEFAULT_PROFILE_AGENT_PLANNER_MODEL,
    ),
    ConversationModelConfigSpec(
        agent=SHARED_MODEL_SCOPE,
        stage=EVALUATOR_STAGE,
        label="Shared / evaluator",
        default_model=DEFAULT_SHARED_EVALUATOR_MODEL,
    ),
    ConversationModelConfigSpec(
        agent=SHARED_MODEL_SCOPE,
        stage=RERANKER_STAGE,
        label="Shared / reranker",
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
    model: str
    created_at: str = ""
    updated_at: str = ""


class MainAgentConversationModelConfig(BaseModel):
    request_analysis: str
    planner: str
    synthesis: str


class ProfileAgentConversationModelConfig(BaseModel):
    planner: str


class SharedConversationModelConfig(BaseModel):
    evaluator: str
    reranker: str


class ConversationModelConfig(BaseModel):
    SNAPSHOT_MODEL_SUFFIX_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^(?P<base>.+)-\d{4}-\d{2}-\d{2}$")
    MODEL_PRICING_REGISTRY: ClassVar[dict[str, ModelPricing]] = {
        "gpt-5.6": ModelPricing(input_price_per_million_tokens=Decimal("5.00"), output_price_per_million_tokens=Decimal("30.00")),
        "gpt-5.6-sol": ModelPricing(input_price_per_million_tokens=Decimal("5.00"), output_price_per_million_tokens=Decimal("30.00")),
        "gpt-5.6-terra": ModelPricing(input_price_per_million_tokens=Decimal("2.50"), output_price_per_million_tokens=Decimal("15.00")),
        "gpt-5.6-luna": ModelPricing(input_price_per_million_tokens=Decimal("1.00"), output_price_per_million_tokens=Decimal("6.00")),
        "gpt-5.4": ModelPricing(input_price_per_million_tokens=Decimal("2.50"), output_price_per_million_tokens=Decimal("15.00")),
        "gpt-5.4-mini": ModelPricing(input_price_per_million_tokens=Decimal("0.75"), output_price_per_million_tokens=Decimal("4.50")),
        "gpt-5.1": ModelPricing(input_price_per_million_tokens=Decimal("1.25"), output_price_per_million_tokens=Decimal("10.00")),
        "gpt-5": ModelPricing(input_price_per_million_tokens=Decimal("1.25"), output_price_per_million_tokens=Decimal("10.00")),
        "gpt-5-mini": ModelPricing(input_price_per_million_tokens=Decimal("0.25"), output_price_per_million_tokens=Decimal("2.00")),
        "gpt-5-nano": ModelPricing(input_price_per_million_tokens=Decimal("0.05"), output_price_per_million_tokens=Decimal("0.40")),
        "gpt-5-pro": ModelPricing(input_price_per_million_tokens=Decimal("15.00"), output_price_per_million_tokens=Decimal("120.00")),
        "gpt-4.1": ModelPricing(input_price_per_million_tokens=Decimal("2.00"), output_price_per_million_tokens=Decimal("8.00")),
        "gpt-4.1-mini": ModelPricing(input_price_per_million_tokens=Decimal("0.40"), output_price_per_million_tokens=Decimal("1.60")),
        "gpt-4.1-nano": ModelPricing(input_price_per_million_tokens=Decimal("0.10"), output_price_per_million_tokens=Decimal("0.40")),
        "gpt-4o": ModelPricing(input_price_per_million_tokens=Decimal("2.50"), output_price_per_million_tokens=Decimal("10.00")),
        "gpt-4o-mini": ModelPricing(input_price_per_million_tokens=Decimal("0.15"), output_price_per_million_tokens=Decimal("0.60")),
        "o3": ModelPricing(input_price_per_million_tokens=Decimal("2.00"), output_price_per_million_tokens=Decimal("8.00")),
        "o3-pro": ModelPricing(input_price_per_million_tokens=Decimal("20.00"), output_price_per_million_tokens=Decimal("80.00")),
        "o3-mini": ModelPricing(input_price_per_million_tokens=Decimal("1.10"), output_price_per_million_tokens=Decimal("4.40")),
        "o4-mini": ModelPricing(input_price_per_million_tokens=Decimal("1.10"), output_price_per_million_tokens=Decimal("4.40")),
        "o1": ModelPricing(input_price_per_million_tokens=Decimal("15.00"), output_price_per_million_tokens=Decimal("60.00")),
        "o1-pro": ModelPricing(input_price_per_million_tokens=Decimal("150.00"), output_price_per_million_tokens=Decimal("600.00")),
    }

    main_agent: MainAgentConversationModelConfig
    profile_agent: ProfileAgentConversationModelConfig
    shared: SharedConversationModelConfig

    @classmethod
    def build_default(cls) -> ConversationModelConfig:
        return cls(
            main_agent=MainAgentConversationModelConfig(
                request_analysis=DEFAULT_MAIN_AGENT_REQUEST_ANALYSIS_MODEL,
                planner=DEFAULT_MAIN_AGENT_PLANNER_MODEL,
                synthesis=DEFAULT_MAIN_AGENT_SYNTHESIS_MODEL,
            ),
            profile_agent=ProfileAgentConversationModelConfig(
                planner=DEFAULT_PROFILE_AGENT_PLANNER_MODEL,
            ),
            shared=SharedConversationModelConfig(
                evaluator=DEFAULT_SHARED_EVALUATOR_MODEL,
                reranker=DEFAULT_SHARED_RERANKER_MODEL,
            ),
        )

    @classmethod
    def default_main_agent_planner_model(cls) -> str:
        return cls.build_default().main_agent.planner

    @classmethod
    def normalize_model_pricing_key(cls, model_name: str) -> str:
        normalized_name = model_name.strip()
        match = cls.SNAPSHOT_MODEL_SUFFIX_PATTERN.match(normalized_name)
        if match:
            return match.group("base")
        return normalized_name

    @classmethod
    def resolve_model_pricing(cls, model_name: str) -> ModelPricing:
        pricing = cls.MODEL_PRICING_REGISTRY.get(model_name)
        if pricing is None:
            pricing = cls.MODEL_PRICING_REGISTRY.get(cls.normalize_model_pricing_key(model_name))
        if pricing is None:
            raise KeyError(f"Unsupported model pricing key: {model_name}")
        return pricing

    def set_value(self, agent: str, stage: str, model: str) -> None:
        get_model_config_spec(agent, stage)
        if agent == MAIN_AGENT_MODEL_SCOPE:
            setattr(self.main_agent, stage, model)
            return
        if agent == PROFILE_AGENT_MODEL_SCOPE:
            setattr(self.profile_agent, stage, model)
            return
        if agent == SHARED_MODEL_SCOPE:
            setattr(self.shared, stage, model)
            return
        raise KeyError(f"Unsupported conversation model config key: {agent}.{stage}")

    def resolve(self, agent: str, stage: str) -> str:
        get_model_config_spec(agent, stage)
        if agent == MAIN_AGENT_MODEL_SCOPE:
            return getattr(self.main_agent, stage)
        if agent == PROFILE_AGENT_MODEL_SCOPE:
            return getattr(self.profile_agent, stage)
        if agent == SHARED_MODEL_SCOPE:
            return getattr(self.shared, stage)
        raise KeyError(f"Unsupported conversation model config key: {agent}.{stage}")

    def resolve_pricing(self, agent: str, stage: str) -> ModelPricing:
        return self.resolve_model_pricing(self.resolve(agent, stage))

    def to_flat_dict(self) -> dict[str, str]:
        return {
            spec.path: self.resolve(spec.agent, spec.stage)
            for spec in CONVERSATION_MODEL_CONFIG_SPECS
        }

    def to_metadata_payload(self) -> dict[str, Any]:
        return self.model_dump()
