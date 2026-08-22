from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm.chat_models import build_chat_model
from llm.conversation_model_config import ConversationModelConfig
from request_orchestrator.agent_runner.models.agent_profile import AgentProfile
from request_orchestrator.models.agent_inputs import AgentInputs
from request_orchestrator.models.plan import Plan
from .agent_execution_context import AgentExecutionContext
from .agent_result import AgentResult
from .evidence import ToolResult
from request_orchestrator.shared.node_state import AgentNodeStates

@dataclass
class AgentState:
    agent_profile: AgentProfile
    inputs: AgentInputs = field(default_factory=AgentInputs)
    execution_context: AgentExecutionContext = field(default_factory=AgentExecutionContext)
    node_states: AgentNodeStates = field(default_factory=AgentNodeStates)
    result: AgentResult = field(default_factory=AgentResult)
    llm: Any = field(
        default_factory=lambda: build_chat_model(
            provider=ConversationModelConfig.build_default().main_agent.planner.provider,
            model_name=ConversationModelConfig.default_main_agent_planner_model(),
        ),
        repr=False,
    )

    @classmethod
    def new(
        cls,
        agent_profile: AgentProfile,
        task: str = "",
        inputs: AgentInputs | None = None,
        execution_context: AgentExecutionContext | None = None,
        llm: Any | None = None,
    ) -> "AgentState":
        resolved_execution_context = (
            AgentExecutionContext.new()
            if execution_context is None
            else execution_context
        )
        resolved_agent_scope = agent_profile.scope
        return cls(
            inputs=AgentInputs.new(task=task) if inputs is None else inputs,
            execution_context=resolved_execution_context,
            agent_profile=agent_profile,
            llm=(
                build_chat_model(
                    provider=resolved_execution_context.model_config.resolve_provider(
                        agent=resolved_agent_scope,
                        stage="planner",
                    )
                    if agent_profile.model_selection_for_stage("planner") is None
                    else agent_profile.model_selection_for_stage("planner").provider,
                    model_name=resolved_execution_context.model_config.resolve(
                        agent=resolved_agent_scope,
                        stage="planner",
                    )
                    if agent_profile.model_selection_for_stage("planner") is None
                    else agent_profile.model_selection_for_stage("planner").model,
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

    @property
    def agent_name(self) -> str:
        return self.agent_profile.name

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
