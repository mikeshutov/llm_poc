from __future__ import annotations

import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from llm.conversation_model_config import (
    ANTHROPIC_PROVIDER,
    COHERE_PROVIDER,
    ConversationModelConfig,
    DEEPSEEK_PROVIDER,
    GOOGLE_PROVIDER,
    MISTRAL_PROVIDER,
    OPENAI_PROVIDER,
    PLANNER_STAGE,
    XAI_PROVIDER,
)
from request_orchestrator.models.agent_execution_context import AgentExecutionContext


def default_planner_model_for_agent_scope(agent_scope: str) -> str:
    return ConversationModelConfig.build_default().resolve(
        agent=agent_scope,
        stage=PLANNER_STAGE,
    )


def default_planner_provider_for_agent_scope(agent_scope: str) -> str:
    return ConversationModelConfig.build_default().resolve_provider(
        agent=agent_scope,
        stage=PLANNER_STAGE,
    )


def resolve_stage_model_name(
    *,
    execution_context: AgentExecutionContext,
    agent: str,
    stage: str,
) -> str:
    return execution_context.model_config.resolve(agent, stage)


def resolve_stage_provider_name(
    *,
    execution_context: AgentExecutionContext,
    agent: str,
    stage: str,
) -> str:
    return execution_context.model_config.resolve_provider(agent, stage)


def build_chat_model(*, provider: str, model_name: str) -> Any:
    if provider == GOOGLE_PROVIDER:
        return ChatOpenAI(
            model=model_name,
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    if provider == COHERE_PROVIDER:
        return ChatOpenAI(
            model=model_name,
            api_key=os.getenv("COHERE_API_KEY"),
            base_url="https://api.cohere.ai/compatibility/v1",
        )
    if provider == DEEPSEEK_PROVIDER:
        return ChatOpenAI(
            model=model_name,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
    if provider == XAI_PROVIDER:
        return ChatOpenAI(
            model=model_name,
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
        )
    if provider == MISTRAL_PROVIDER:
        return ChatOpenAI(
            model=model_name,
            api_key=os.getenv("MISTRAL_API_KEY"),
            base_url="https://api.mistral.ai/v1",
        )
    if provider == OPENAI_PROVIDER:
        return ChatOpenAI(model=model_name)
    if provider == ANTHROPIC_PROVIDER:
        return ChatAnthropic(model_name=model_name)
    raise KeyError(f"Unsupported model provider: {provider}")


def is_provider_model_instance(llm: Any, provider: str) -> bool:
    if provider == OPENAI_PROVIDER:
        return isinstance(llm, ChatOpenAI)
    if provider == ANTHROPIC_PROVIDER:
        return isinstance(llm, ChatAnthropic)
    return False


def build_llm_for_stage(
    *,
    execution_context: AgentExecutionContext,
    llm: Any,
    agent: str,
    stage: str,
    reuse_llm_for_agent_scope: str | None = None,
) -> Any:
    provider = resolve_stage_provider_name(
        execution_context=execution_context,
        agent=agent,
        stage=stage,
    )
    model_name = resolve_stage_model_name(
        execution_context=execution_context,
        agent=agent,
        stage=stage,
    )
    if llm is None:
        return build_chat_model(provider=provider, model_name=model_name)
    if (
        not isinstance(llm, (ChatOpenAI, ChatAnthropic))
        and not hasattr(llm, "model")
        and not hasattr(llm, "model_name")
    ):
        return llm
    if (
        reuse_llm_for_agent_scope is not None
        and agent == reuse_llm_for_agent_scope
        and stage == PLANNER_STAGE
        and provider == default_planner_provider_for_agent_scope(reuse_llm_for_agent_scope)
        and model_name == default_planner_model_for_agent_scope(reuse_llm_for_agent_scope)
    ):
        return llm
    return build_chat_model(provider=provider, model_name=model_name)
