from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pydantic import BaseModel

MAIN_AGENT_MODEL_SCOPE = "main_agent"
PROFILE_AGENT_MODEL_SCOPE = "profile_agent"
SHARED_MODEL_SCOPE = "shared"

REQUEST_ANALYSIS_STAGE = "request_analysis"
PLANNER_STAGE = "planner"
SYNTHESIS_STAGE = "synthesis"
RERANKER_STAGE = "reranker"

DEFAULT_MAIN_AGENT_MODEL = os.getenv("LLM_MODEL", "gpt-5.4")
DEFAULT_MAIN_AGENT_REQUEST_ANALYSIS_MODEL = os.getenv("MAIN_AGENT_REQUEST_ANALYSIS_MODEL", "gpt-5.4-mini")
DEFAULT_MAIN_AGENT_PLANNER_MODEL = os.getenv("MAIN_AGENT_PLANNER_MODEL", DEFAULT_MAIN_AGENT_MODEL)
DEFAULT_MAIN_AGENT_SYNTHESIS_MODEL = os.getenv("MAIN_AGENT_SYNTHESIS_MODEL", DEFAULT_MAIN_AGENT_MODEL)
DEFAULT_PROFILE_AGENT_PLANNER_MODEL = os.getenv("PROFILE_MANAGEMENT_MODEL", "gpt-5.4-mini")
DEFAULT_SHARED_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "gpt-5.4-mini")


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
    reranker: str


class ConversationModelConfig(BaseModel):
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
                reranker=DEFAULT_SHARED_RERANKER_MODEL,
            ),
        )

    @classmethod
    def default_main_agent_planner_model(cls) -> str:
        return cls.build_default().main_agent.planner

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

    def to_flat_dict(self) -> dict[str, str]:
        return {
            spec.path: self.resolve(spec.agent, spec.stage)
            for spec in CONVERSATION_MODEL_CONFIG_SPECS
        }

    def to_metadata_payload(self) -> dict[str, Any]:
        return self.model_dump()
