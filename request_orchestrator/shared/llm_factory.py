from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from conversation.models.conversation_model_config import ConversationModelConfig, PLANNER_STAGE
from request_orchestrator.models.agent_execution_context import AgentExecutionContext


def default_planner_model_for_agent_scope(agent_scope: str) -> str:
    return ConversationModelConfig.build_default().resolve(
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


def build_llm_for_stage(
    *,
    execution_context: AgentExecutionContext,
    llm: Any,
    agent: str,
    stage: str,
    reuse_llm_for_agent_scope: str | None = None,
) -> Any:
    model_name = resolve_stage_model_name(
        execution_context=execution_context,
        agent=agent,
        stage=stage,
    )
    if llm is None:
        return ChatOpenAI(model=model_name)
    if not isinstance(llm, ChatOpenAI):
        return llm
    if (
        reuse_llm_for_agent_scope is not None
        and agent == reuse_llm_for_agent_scope
        and stage == PLANNER_STAGE
        and model_name == default_planner_model_for_agent_scope(reuse_llm_for_agent_scope)
    ):
        return llm
    return ChatOpenAI(model=model_name)
