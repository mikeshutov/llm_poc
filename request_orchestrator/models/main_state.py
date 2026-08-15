from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any
from uuid import UUID

from conversation.models.conversation_model_config import ConversationModelConfig
from conversation.models.conversation_models import ConversationContext
from personalization.profile.models import UserProfile
from request_orchestrator.agent_runner.models.agent_profile import AgentProfile
from request_orchestrator.models.evidence import ToolResult
from request_orchestrator.models.agent_execution_context import AgentExecutionContext
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.orchestrator_result import OrchestratorResult
from request_orchestrator.models.request_analysis import RequestAnalysis, RequestAnalysisGoal

@dataclass
class MainState:
    task: str
    execution_context: AgentExecutionContext = field(default_factory=AgentExecutionContext)
    agent_profiles: list[AgentProfile] = field(default_factory=list)
    request_analysis: RequestAnalysis = field(default_factory=RequestAnalysis)
    agent_states: dict[str, AgentState] = field(default_factory=dict)
    result: OrchestratorResult = field(default_factory=OrchestratorResult)
    llm: Any = None

    @classmethod
    def new(
        cls,
        *,
        task: str,
        max_turns: int | None = None,
        conversation_context: ConversationContext | None = None,
        user_profile: UserProfile | None = None,
        conversation_id: str | None = None,
        roundtrip_id: UUID | None = None,
        llm: Any | None = None,
        conversation_model_config: ConversationModelConfig | None = None,
        agent_profiles: list[AgentProfile],
        initialize_agent_states: bool = True,
    ) -> "MainState":
        resolved_user_profile = UserProfile() if user_profile is None else user_profile
        resolved_agent_profiles = [
            replace(agent_profile, max_turns=max_turns)
            if max_turns is not None and max_turns != agent_profile.max_turns
            else agent_profile
            for agent_profile in agent_profiles
        ]
        state = cls(
            task=task,
            execution_context=AgentExecutionContext(
                conversation_context=ConversationContext() if conversation_context is None else conversation_context,
                user_profile=resolved_user_profile,
                conversation_id=conversation_id,
                roundtrip_id=roundtrip_id,
                model_config=ConversationModelConfig.build_default() if conversation_model_config is None else conversation_model_config,
            ),
            agent_profiles=resolved_agent_profiles,
            llm=llm,
        )
        if initialize_agent_states:
            state.initialize_agent_states()
        return state

    def initialize_agent_states(self) -> None:
        for agent_profile in self.agent_profiles:
            if agent_profile.name not in self.agent_states:
                self.agent_states[agent_profile.name] = AgentState.new(
                    task=self.task,
                    conversation_context=self.execution_context.conversation_context,
                    user_profile=self.execution_context.user_profile,
                    agent_profile=agent_profile,
                    conversation_id=self.execution_context.conversation_id,
                    roundtrip_id=self.execution_context.roundtrip_id,
                    llm=self.llm,
                    conversation_model_config=self.execution_context.model_config,
                )

        goals_by_agent = self._goals_by_agent()
        for agent_state in self.agent_states.values():
            goal_entry = goals_by_agent.get(agent_state.agent_profile.name)
            if goal_entry is None:
                agent_state.set_agent_inputs(task="")
                continue
            agent_state.set_agent_inputs(
                task=goal_entry.goal,
                tool_category_names=list(goal_entry.tool_categories),
            )

    def _goals_by_agent(self) -> dict[str, RequestAnalysisGoal]:
        goals_by_agent: dict[str, RequestAnalysisGoal] = {}
        for goal in self.request_analysis.goals:
            normalized_agent_name = goal.agent.strip()
            if not normalized_agent_name:
                continue
            goals_by_agent[normalized_agent_name] = goal.model_copy(deep=True)
        return goals_by_agent

    def gather_relevant_evidence_ids(self) -> list[str]:
        relevant_evidence_ids: list[str] = []
        seen_evidence_ids: set[str] = set()
        for agent_state in self.agent_states.values():
            for evidence_id in agent_state.result.relevant_evidence_ids:
                normalized = evidence_id.strip()
                if not normalized or normalized in seen_evidence_ids:
                    continue
                seen_evidence_ids.add(normalized)
                relevant_evidence_ids.append(normalized)
        return relevant_evidence_ids

    def gather_tool_results(self) -> list[ToolResult]:
        gathered: list[ToolResult] = []
        for agent_state in self.agent_states.values():
            gathered.extend(agent_state.gather_tool_results())
        return gathered

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
