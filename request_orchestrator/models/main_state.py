from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from conversation.models.conversation_model_config import ConversationModelConfig
from conversation.models.conversation_models import ConversationContext
from personalization.profile.models import UserProfile
from request_orchestrator.models.agent_profile import AgentProfile, DEFAULT_SYNTHESIS_INSTRUCTION
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.agent_state import AgentState, IterationState, RequestAnalysis, RequestAnalysisGoal
from request_orchestrator.models.plan_step_ids import format_plan_step_id

@dataclass
class MainState:
    task: str
    max_turns: int
    conversation_context: ConversationContext = field(default_factory=ConversationContext)
    user_profile: UserProfile = field(default_factory=UserProfile)
    synthesis_instruction: str = DEFAULT_SYNTHESIS_INSTRUCTION
    agent_profiles: list[AgentProfile] = field(default_factory=list)
    conversation_id: str | None = None
    roundtrip_id: UUID | None = None
    request_analysis: RequestAnalysis = field(default_factory=RequestAnalysis)
    agent_states: list[AgentState] = field(default_factory=list)
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
        synthesis_instruction: str = DEFAULT_SYNTHESIS_INSTRUCTION,
        agent_profiles: list[AgentProfile],
        initialize_agent_states: bool = True,
    ) -> "MainState":
        resolved_user_profile = UserProfile() if user_profile is None else user_profile
        state = cls(
            task=task,
            max_turns=max_turns,
            conversation_context=ConversationContext() if conversation_context is None else conversation_context,
            user_profile=resolved_user_profile,
            synthesis_instruction=synthesis_instruction,
            agent_profiles=list(agent_profiles),
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

        for agent_profile in self.agent_profiles:
            self.upsert_agent_state(self.new_agent_state(agent_profile))

    def new_agent_state(self, agent_profile: AgentProfile) -> AgentState:
        return AgentState.new(
            task=self.task,
            max_turns=self.max_turns,
            conversation_context=self.conversation_context,
            user_profile=self.user_profile,
            agent_profile=agent_profile,
            conversation_id=self.conversation_id,
            roundtrip_id=self.roundtrip_id,
            llm=self.llm,
            conversation_model_config=self.conversation_model_config,
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

    def distribute_goals_to_agent_states(self) -> None:
        goals_by_agent: dict[str, list[RequestAnalysisGoal]] = {}
        for goal in self.request_analysis.goals:
            normalized_agent_name = goal.agent.strip()
            if not normalized_agent_name:
                continue
            goals_by_agent.setdefault(normalized_agent_name, []).append(
                goal.model_copy(deep=True)
            )

        for agent_state in self.agent_states:
            agent_state.request_analysis = RequestAnalysis(
                goals=goals_by_agent.get(agent_state.agent_profile.name, []),
                requested_user_attribute_types=list(self.request_analysis.requested_user_attribute_types),
            )

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
        self.result = self.result.copy(used_evidence_ids=deduped_evidence_ids)

    def build_final_result(self) -> AgentResult:
        return self.result.copy()

    def _build_synthesis_view(self) -> tuple[list[IterationState], list[str]]:
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

        def remap_evidence_id(evidence_id: str) -> str:
            normalized = evidence_id.strip()
            if not normalized:
                return ""

            for old_step_id, new_step_id in step_id_map.items():
                if normalized == old_step_id:
                    return new_step_id
                if normalized.startswith(f"{old_step_id}R"):
                    return f"{new_step_id}{normalized[len(old_step_id):]}"

            return normalized

        iterations: list[IterationState] = []
        relevant_evidence_ids: list[str] = []
        seen_evidence_ids: set[str] = set()
        global_iteration_number = 1

        for agent_state in self.agent_states:
            for local_iteration_number, iteration in enumerate(agent_state.iteration_trace, start=1):
                cloned_iteration = iteration.clone()
                remapped_results: dict[str, Any] = {}

                for result_step_id, result_value in cloned_iteration.results.items():
                    remapped_results[remap_evidence_id(result_step_id)] = result_value

                if cloned_iteration.plan is not None:
                    for step in cloned_iteration.plan.steps:
                        old_step_id = format_plan_step_id(local_iteration_number, step.id)
                        new_step_id = format_plan_step_id(global_iteration_number, step.id)
                        if old_step_id in cloned_iteration.results and new_step_id not in remapped_results:
                            remapped_results[new_step_id] = cloned_iteration.results[old_step_id]

                cloned_iteration.results = remapped_results
                iterations.append(cloned_iteration)
                global_iteration_number += 1

            for evidence_id in agent_state.relevant_evidence_ids:
                normalized = remap_evidence_id(evidence_id)
                if not normalized or normalized in seen_evidence_ids:
                    continue
                seen_evidence_ids.add(normalized)
                relevant_evidence_ids.append(normalized)

        return iterations, relevant_evidence_ids

    def gather_iteration_trace(self) -> list[IterationState]:
        iterations, _ = self._build_synthesis_view()
        return iterations

    def gather_relevant_evidence_ids(self) -> list[str]:
        _, relevant_evidence_ids = self._build_synthesis_view()
        return relevant_evidence_ids
