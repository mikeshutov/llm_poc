from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any
from uuid import UUID

from langchain_openai import ChatOpenAI
from conversation.models.conversation_model_config import ConversationModelConfig
from conversation.models.conversation_models import ConversationContext
from personalization.profile.models import UserProfile
from request_orchestrator.agent_runner.models.agent_profile import AgentProfile
from request_orchestrator.models.plan import Plan
from .agent_execution_context import AgentExecutionContext
from .agent_result import AgentResult
from .evidence import ToolResult
from request_orchestrator.shared.node_state import AgentNodeStates

@dataclass
class AgentState:
    task: str
    agent_profile: AgentProfile
    execution_context: AgentExecutionContext = field(default_factory=AgentExecutionContext)
    tool_category_names: list[str] = field(default_factory=list)
    node_states: AgentNodeStates = field(default_factory=AgentNodeStates)
    result: AgentResult = field(default_factory=AgentResult)
    llm: Any = field(
        default_factory=lambda: ChatOpenAI(
            model=ConversationModelConfig.default_main_agent_planner_model()
        ),
        repr=False,
    )

    @classmethod
    def new(
        cls,
        task: str,
        agent_profile: AgentProfile,
        max_turns: int | None = None,
        conversation_context: ConversationContext | None = None,
        user_profile: UserProfile | None = None,
        conversation_id: str | None = None,
        roundtrip_id: UUID | None = None,
        llm: Any | None = None,
        conversation_model_config: ConversationModelConfig | None = None,
    ) -> "AgentState":
        resolved_conversation_model_config = (
            ConversationModelConfig.build_default()
            if conversation_model_config is None
            else conversation_model_config
        )
        resolved_agent_profile = (
            replace(agent_profile, max_turns=max_turns)
            if max_turns is not None and max_turns != agent_profile.max_turns
            else agent_profile
        )
        resolved_agent_scope = resolved_agent_profile.scope
        return cls(
            task=task,
            execution_context=AgentExecutionContext(
                conversation_context=ConversationContext() if conversation_context is None else conversation_context,
                user_profile=UserProfile() if user_profile is None else user_profile,
                conversation_id=conversation_id,
                roundtrip_id=roundtrip_id,
                model_config=resolved_conversation_model_config,
            ),
            agent_profile=resolved_agent_profile,
            llm=(
                ChatOpenAI(
                    model=resolved_conversation_model_config.resolve(
                        agent=resolved_agent_scope,
                        stage="planner",
                    )
                )
                if llm is None
                else llm
            ),
        )

    def begin_plan(self, plan: Plan | None, *, needs_replan: bool = False) -> None:
        self.node_states.planner.plan = None if plan is None else plan.model_copy(deep=True)
        self.node_states.planner.needs_replan = needs_replan
        self.node_states.planner.plan_count += 1

    @property
    def max_turns(self) -> int:
        return self.agent_profile.max_turns

    def set_agent_inputs(
        self,
        *,
        task: str,
        tool_category_names: list[str] | None = None,
    ) -> None:
        self.task = task.strip()
        self.tool_category_names = [] if tool_category_names is None else [
            category.strip()
            for category in tool_category_names
            if isinstance(category, str) and category.strip()
        ]

    def resolve_agent_scope(self) -> str:
        return self.agent_profile.scope

    def gather_tool_results(self) -> list[ToolResult]:
        return [tool_result.model_copy(deep=True) for tool_result in self.result.tool_results]

    def gather_used_tools(self) -> list[str]:
        used_tools: list[str] = []
        seen: set[str] = set()
        for tool_result in self.gather_tool_results():
            tool_name = tool_result.tool_name.strip()
            if not tool_name or tool_name in seen:
                continue
            seen.add(tool_name)
            used_tools.append(tool_name)
        return used_tools

    def upsert_tool_result(self, tool_result: ToolResult) -> None:
        resolved_tool_result = tool_result.model_copy(deep=True)
        updated_tool_results = [existing.model_copy(deep=True) for existing in self.result.tool_results]
        if resolved_tool_result.step_id.strip():
            for index, existing in enumerate(updated_tool_results):
                if existing.step_id.strip() == resolved_tool_result.step_id.strip():
                    updated_tool_results[index] = resolved_tool_result
                    self.result = self.result.copy(tool_results=updated_tool_results)
                    return
        updated_tool_results.append(resolved_tool_result)
        self.result = self.result.copy(tool_results=updated_tool_results)
