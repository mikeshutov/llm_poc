from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from conversation.models.conversation_model_config import ConversationModelConfig
from conversation.models.conversation_models import ConversationContext
from personalization.profile.models import UserProfile
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.agent_state import AgentState, AgentStateLog, IterationState, RequestAnalysis, RequestAnalysisGoal
from request_orchestrator.models.plan_step_ids import format_plan_step_id


def _get_main_agent_profile():
    from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE

    return MAIN_AGENT_PROFILE


def _build_profile_management_profile(user_profile: UserProfile | None):
    from request_orchestrator.agents.profile_management.profile import build_profile_management_profile

    return build_profile_management_profile(user_profile)


@dataclass
class MainState:
    task: str
    max_turns: int
    conversation_context: ConversationContext = field(default_factory=ConversationContext)
    user_profile: UserProfile = field(default_factory=UserProfile)
    conversation_id: str | None = None
    roundtrip_id: UUID | None = None
    request_analysis: RequestAnalysis = field(default_factory=RequestAnalysis)
    agent_states: list[AgentState] = field(default_factory=list)
    agent_log: AgentStateLog = field(default_factory=AgentStateLog)
    result: AgentResult = field(default_factory=lambda: AgentResult(answer=[]))
    llm: Any = None
    conversation_model_config: ConversationModelConfig = field(default_factory=ConversationModelConfig.build_default)

    @classmethod
    def new(
        cls,
        *,
        task: str,
        max_turns: int,
        conversation_context: ConversationContext | None = None,
        user_profile: UserProfile | None = None,
        conversation_id: str | None = None,
        roundtrip_id: UUID | None = None,
        llm: Any | None = None,
        conversation_model_config: ConversationModelConfig | None = None,
        initialize_agent_states: bool = True,
    ) -> "MainState":
        state = cls(
            task=task,
            max_turns=max_turns,
            conversation_context=ConversationContext() if conversation_context is None else conversation_context,
            user_profile=UserProfile() if user_profile is None else user_profile,
            conversation_id=conversation_id,
            roundtrip_id=roundtrip_id,
            llm=llm,
            conversation_model_config=ConversationModelConfig.build_default() if conversation_model_config is None else conversation_model_config,
        )
        if initialize_agent_states:
            state.initialize_agent_states()
        return state

    def initialize_agent_states(self) -> None:
        if self.agent_states:
            return

        self.upsert_agent_state(
            AgentState.new(
                task=self.task,
                max_turns=self.max_turns,
                conversation_context=self.conversation_context,
                user_profile=self.user_profile,
                agent_profile=_build_profile_management_profile(self.user_profile),
                conversation_id=self.conversation_id,
                roundtrip_id=self.roundtrip_id,
                llm=self.llm,
                conversation_model_config=self.conversation_model_config,
            )
        )
        self.upsert_agent_state(
            AgentState.new(
                task=self.task,
                max_turns=self.max_turns,
                conversation_context=self.conversation_context,
                user_profile=self.user_profile,
                agent_profile=_get_main_agent_profile(),
                conversation_id=self.conversation_id,
                roundtrip_id=self.roundtrip_id,
                llm=self.llm,
                conversation_model_config=self.conversation_model_config,
            )
        )

    def agent_state_map(self) -> dict[str, AgentState]:
        return {
            agent_state.agent_profile.name: agent_state
            for agent_state in self.agent_states
        }

    def get_agent_state(self, agent_name: str) -> AgentState:
        agent_state = self.agent_state_map().get(agent_name)
        if agent_state is not None:
            return agent_state
        raise KeyError(f"Unknown agent state: {agent_name}")

    def upsert_agent_state(self, updated_state: AgentState) -> None:
        updated_map = self.agent_state_map()
        updated_map[updated_state.agent_profile.name] = updated_state
        self.agent_states = list(updated_map.values())

    def routable_agent_states(self) -> list[AgentState]:
        return [
            agent_state
            for agent_state in self.agent_states
            if agent_state.agent_profile.request_analysis_selectable
        ]

    def request_analysis_agent_state(self) -> AgentState:
        routable_agent_states = self.routable_agent_states()
        if routable_agent_states:
            return routable_agent_states[0]
        if self.agent_states:
            return self.agent_states[0]
        raise KeyError("No agent states available")

    def synthesis_agent_state(self) -> AgentState:
        routable_agent_states = self.routable_agent_states()
        if routable_agent_states:
            return routable_agent_states[-1]
        if self.agent_states:
            return self.agent_states[-1]
        raise KeyError("No agent states available")

    def available_request_analysis_agents_payload(self) -> list[dict[str, Any]]:
        available_agents: list[dict[str, Any]] = []
        for agent_state in self.routable_agent_states():
            available_agents.append(
                {
                    "agent": agent_state.agent_profile.name,
                    "tool_categories": [
                        {
                            "name": name,
                            "description": category.description,
                        }
                        for name, category in sorted(agent_state.agent_profile.allowed_tool_categories().items())
                    ],
                }
            )
        return available_agents

    def resolve_synthesis_instruction(self) -> str:
        instruction = self.synthesis_agent_state().agent_profile.synthesis_instruction.strip()
        if instruction:
            return instruction
        return _get_main_agent_profile().synthesis_instruction

    def fan_out_shared_state(self) -> None:
        for agent_state in self.agent_states:
            agent_state.task = self.task
            agent_state.max_turns = self.max_turns
            agent_state.conversation_context = self.conversation_context
            agent_state.user_profile = self.user_profile
            agent_state.conversation_id = self.conversation_id
            agent_state.roundtrip_id = self.roundtrip_id
            if self.llm is not None:
                agent_state.llm = self.llm
            agent_state.conversation_model_config = self.conversation_model_config

    def build_agent_request_analysis(self, agent_name: str) -> RequestAnalysis:
        normalized_agent_name = agent_name.strip()
        matching_goals: list[RequestAnalysisGoal] = []

        for goal in self.request_analysis.goals:
            if goal.agent.strip() != normalized_agent_name:
                continue
            matching_goals.append(goal.model_copy(deep=True))

        return RequestAnalysis(
            goals=matching_goals,
            requested_user_attribute_types=list(self.request_analysis.requested_user_attribute_types),
        )

    def distribute_goals_to_agent_states(self) -> None:
        for agent_state in self.agent_states:
            agent_state.request_analysis = self.build_agent_request_analysis(agent_state.agent_profile.name)

    def collect_agent_outputs(self) -> None:
        if not self.agent_states:
            return

        relevant_evidence_ids: list[str] = []
        for agent_state in self.agent_states:
            relevant_evidence_ids.extend(agent_state.relevant_evidence_ids)

        deduped_evidence_ids: list[str] = []
        seen: set[str] = set()
        for evidence_id in relevant_evidence_ids:
            normalized = evidence_id.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped_evidence_ids.append(normalized)
        self.result = AgentResult(
            answer=list(self.result.answer),
            answer_blocks=list(self.result.answer_blocks),
            next_question=self.result.next_question,
            roundtrip_summary=self.result.roundtrip_summary,
            roundtrip_latency_ms=self.result.roundtrip_latency_ms,
            tool_summary=dict(self.result.tool_summary),
            agent_logs=self.build_agent_logs(),
            used_evidence_ids=deduped_evidence_ids,
            hydrated_evidence_by_id=dict(self.result.hydrated_evidence_by_id),
        )

    def build_final_result(self) -> AgentResult:
        return AgentResult(
            answer=list(self.result.answer),
            answer_blocks=list(self.result.answer_blocks),
            next_question=self.result.next_question,
            roundtrip_summary=self.result.roundtrip_summary,
            roundtrip_latency_ms=self.result.roundtrip_latency_ms,
            tool_summary=dict(self.result.tool_summary),
            agent_logs=self.build_agent_logs(),
            used_evidence_ids=list(self.result.used_evidence_ids),
            hydrated_evidence_by_id=dict(self.result.hydrated_evidence_by_id),
        )

    def build_agent_logs(self) -> dict[str, list[dict[str, Any]]]:
        logs = self.agent_log.to_grouped_dict()
        for agent_state in self.agent_states:
            for agent_name, entries in agent_state.build_agent_logs().items():
                logs.setdefault(agent_name, []).extend(entries)
        return logs

    def _build_synthesis_step_id_map(self) -> dict[str, str]:
        step_id_map: dict[str, str] = {}
        global_iteration_number = 1

        for agent_state in self.agent_states:
            for local_iteration_number, iteration in enumerate(agent_state.iteration_trace, start=1):
                if iteration.plan is not None:
                    for step in iteration.plan.steps:
                        step_id_map[
                            format_plan_step_id(local_iteration_number, step.id)
                        ] = format_plan_step_id(global_iteration_number, step.id)
                global_iteration_number += 1

        return step_id_map

    @staticmethod
    def _remap_evidence_id(evidence_id: str, step_id_map: dict[str, str]) -> str:
        normalized = evidence_id.strip()
        if not normalized:
            return ""

        for old_step_id, new_step_id in step_id_map.items():
            if normalized == old_step_id:
                return new_step_id
            if normalized.startswith(f"{old_step_id}R"):
                return f"{new_step_id}{normalized[len(old_step_id):]}"

        return normalized

    def gather_iteration_trace(self) -> list[IterationState]:
        step_id_map = self._build_synthesis_step_id_map()
        iterations: list[IterationState] = []
        global_iteration_number = 1

        for agent_state in self.agent_states:
            for local_iteration_number, iteration in enumerate(agent_state.iteration_trace, start=1):
                cloned_iteration = iteration.clone()
                remapped_results: dict[str, Any] = {}

                for result_step_id, result_value in cloned_iteration.results.items():
                    remapped_results[
                        self._remap_evidence_id(result_step_id, step_id_map)
                    ] = result_value

                if cloned_iteration.plan is not None:
                    for step in cloned_iteration.plan.steps:
                        old_step_id = format_plan_step_id(local_iteration_number, step.id)
                        new_step_id = format_plan_step_id(global_iteration_number, step.id)
                        if old_step_id in cloned_iteration.results and new_step_id not in remapped_results:
                            remapped_results[new_step_id] = cloned_iteration.results[old_step_id]

                cloned_iteration.results = remapped_results
                iterations.append(cloned_iteration)
                global_iteration_number += 1

        return iterations

    def gather_relevant_evidence_ids(self) -> list[str]:
        step_id_map = self._build_synthesis_step_id_map()
        evidence_ids: list[str] = []
        seen: set[str] = set()
        for agent_state in self.agent_states:
            for evidence_id in agent_state.relevant_evidence_ids:
                normalized = self._remap_evidence_id(evidence_id, step_id_map)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                evidence_ids.append(normalized)
        return evidence_ids
