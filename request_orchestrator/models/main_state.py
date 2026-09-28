from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from request_orchestrator.agent_runner.models.agent_profile import AgentProfile
from request_orchestrator.models.agent_inputs import AgentInputs
from uuid import UUID

from request_orchestrator.models.evidence import ToolResult
from request_orchestrator.models.agent_execution_context import AgentExecutionContext
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.orchestrator_result import OrchestratorResult
from request_orchestrator.shared.discovery.models import CapabilityDiscoveryResult
from request_orchestrator.models.evaluation_result import EvaluationStatus, EVALUATION_STATUS_RETRYABLE
from request_orchestrator.models.plan import Plan

@dataclass
class MainState:
    task: str
    execution_context: AgentExecutionContext = field(default_factory=AgentExecutionContext)
    agent_profiles: list[AgentProfile] = field(default_factory=list)
    discovered_capabilities: CapabilityDiscoveryResult | None = None
    discovered_attribute_types: list[str] = field(default_factory=list)
    evaluation_status: EvaluationStatus = EVALUATION_STATUS_RETRYABLE
    missing_information: list[str] = field(default_factory=list)
    relevant_evidence_ids: list[UUID] = field(default_factory=list)
    plan: Plan | None = None
    main_agent_state: AgentState | None = None
    direct_tool_results: list[ToolResult] = field(default_factory=list)
    agent_states: dict[str, AgentState] = field(default_factory=dict)
    result: OrchestratorResult = field(default_factory=OrchestratorResult)
    llm: Any = None

    @classmethod
    def new(
        cls,
        *,
        task: str,
        execution_context: AgentExecutionContext | None = None,
        llm: Any | None = None,
        agent_profiles: list[AgentProfile],
    ) -> "MainState":
        resolved_execution_context = (
            AgentExecutionContext.new()
            if execution_context is None
            else execution_context
        )
        state = cls(
            task=task,
            execution_context=resolved_execution_context,
            agent_profiles=list(agent_profiles),
            llm=llm,
        )
        state.initialize_agent_states()
        return state

    def initialize_agent_states(self) -> None:
        for agent_profile in self.agent_profiles:
            if agent_profile.name not in self.agent_states:
                self.agent_states[agent_profile.name] = AgentState.new(
                    agent_profile=agent_profile,
                    inputs=AgentInputs.new(task=self.task, request_task=self.task),
                    execution_context=replace(self.execution_context),
                    llm=self.llm,
                )

        for agent_state in self.agent_states.values():
            agent_state.inputs = AgentInputs.new(task="", request_task=self.task)

    def gather_relevant_evidence_ids(self) -> list[UUID]:
        relevant_evidence_ids: list[UUID] = []
        seen_evidence_ids: set[UUID] = set()
        agent_states = self._agent_states_for_aggregation()
        for agent_state in agent_states:
            for evidence_id in agent_state.result.relevant_evidence_ids:
                if evidence_id in seen_evidence_ids:
                    continue
                seen_evidence_ids.add(evidence_id)
                relevant_evidence_ids.append(evidence_id)
        return relevant_evidence_ids

    def gather_tool_call_ids(self) -> list[UUID]:
        tool_call_ids: list[UUID] = []
        seen_tool_call_ids: set[UUID] = set()
        agent_states = self._agent_states_for_aggregation()
        for agent_state in agent_states:
            for tool_call_id in agent_state.result.tool_call_ids:
                if tool_call_id in seen_tool_call_ids:
                    continue
                seen_tool_call_ids.add(tool_call_id)
                tool_call_ids.append(tool_call_id)
        return tool_call_ids

    def gather_tool_results(self) -> list[ToolResult]:
        gathered: list[ToolResult] = list(self.direct_tool_results)
        for agent_state in self._agent_states_for_aggregation():
            gathered.extend(agent_state.gather_tool_results())
        return gathered

    def _agent_states_for_aggregation(self) -> list[AgentState]:
        """Return each completed execution tree once for synthesis/persistence."""
        if self.main_agent_state is not None:
            return [self.main_agent_state]
        return list(self.agent_states.values())

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
