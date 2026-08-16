from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from request_orchestrator.agent_runner.models.agent_profile import AgentProfile
from request_orchestrator.models.agent_execution_context import AgentExecutionContext
from request_orchestrator.models.agent_inputs import AgentInputs
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.request_analysis import RequestAnalysis, RequestAnalysisGoal


class AgentStratagy(Protocol):
    def run(self, agent_state: AgentState, *, thread_id: str) -> AgentState: ...


@dataclass(frozen=True)
class AgentRunner:
    profile: AgentProfile
    stratagy: AgentStratagy

    def _default_goal(self) -> str:
        return self.profile.request_analysis_goal.strip()

    def _default_tool_categories(self) -> list[str]:
        return sorted(self.profile.allowed_categories)

    def _build_default_request_analysis(self) -> RequestAnalysis:
        goal = self.profile.request_analysis_goal.strip()
        if not goal:
            return RequestAnalysis()
        return RequestAnalysis(
            goals=[
                RequestAnalysisGoal(
                    agent=self.profile.name,
                    goal=goal,
                    tool_categories=sorted(self.profile.allowed_categories),
                )
            ],
        )

    def _prepare_state(self, agent_state: AgentState) -> AgentState:
        agent_state.agent_profile = self.profile
        if not agent_state.inputs.task and self._default_goal():
            agent_state.inputs = AgentInputs.new(
                task=self._default_goal(),
                tool_category_names=self._default_tool_categories(),
            )
        return agent_state

    def run(
        self,
        agent_state: AgentState | None = None,
        *,
        user_query: str | None = None,
        execution_context: AgentExecutionContext | None = None,
        request_analysis: RequestAnalysis | None = None,
        llm: Any | None = None,
    ) -> AgentState:
        if agent_state is None:
            if user_query is None:
                raise ValueError("user_query is required when agent_state is not provided")
            if execution_context is None:
                raise ValueError("execution_context is required when agent_state is not provided")
            agent_state = AgentState.new(
                agent_profile=self.profile,
                inputs=AgentInputs.new(task=user_query),
                execution_context=execution_context,
                llm=llm,
            )
            if request_analysis is not None:
                agent_state.inputs = request_analysis.inputs_for_agent(self.profile.name)

        prepared_state = self._prepare_state(agent_state)
        return self.stratagy.run(
            prepared_state,
            thread_id=prepared_state.execution_context.conversation_id or "",
        )
