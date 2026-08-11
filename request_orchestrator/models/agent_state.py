from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from common.serialization import sanitize_for_json_storage
from conversation.models.conversation_model_config import (
    ConversationModelConfig,
    MAIN_AGENT_MODEL_SCOPE,
    PLANNER_STAGE,
    PROFILE_AGENT_MODEL_SCOPE,
)
from conversation.models.conversation_models import ConversationContext
from personalization.profile.models import GeoLocation, GeoMetadata, UserProfile
from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE
from request_orchestrator.agents.models.agent_profile import AgentProfile
from .agent_result import AgentResult
from .evaluation_result import EvaluationStatus, EVALUATION_STATUS_RETRYABLE
from .plan import Plan


class RequestAnalysis(BaseModel):
    goal: str = ""
    applicable_tool_categories: list[str] = []
    requested_user_attribute_types: list[str] = []



def build_geometadata(
    *,
    timezone: str | None = "America/Toronto",
    location: GeoLocation | None = None,
) -> GeoMetadata:
    resolved_timezone = (timezone or "").strip()
    if not resolved_timezone and location is not None:
        resolved_timezone = (location.timezone or "").strip()
    if not resolved_timezone:
        resolved_timezone = "America/Toronto"

    now = datetime.now(ZoneInfo(resolved_timezone))
    return GeoMetadata(
        current_datetime=now.isoformat(),
        current_weekday=now.strftime("%A"),
        timezone=resolved_timezone,
        location=location,
    )


@dataclass
class AgentStateLogElement:
    agent_name: str
    kind: str = "event"
    title: str = ""
    summary: str = ""
    details: str = ""
    status: str = ""
    tool_name: str = ""
    step_id: str = ""
    iteration: int | None = None
    request: Any | None = None
    response: Any | None = None
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "agent_name": self.agent_name,
            "kind": self.kind,
        }
        if self.title:
            payload["title"] = self.title
        if self.summary:
            payload["summary"] = self.summary
        if self.details:
            payload["details"] = self.details
        if self.status:
            payload["status"] = self.status
        if self.tool_name:
            payload["tool_name"] = self.tool_name
        if self.step_id:
            payload["step_id"] = self.step_id
        if self.iteration is not None:
            payload["iteration"] = self.iteration
        if self.request is not None:
            payload["request"] = sanitize_for_json_storage(self.request)
        if self.response is not None:
            payload["response"] = sanitize_for_json_storage(self.response)
        if self.error:
            payload["error"] = self.error
        if self.data:
            payload["data"] = sanitize_for_json_storage(self.data)
        if self.metadata:
            payload["metadata"] = sanitize_for_json_storage(self.metadata)
        return payload


@dataclass
class AgentStateLog:
    entries: list[AgentStateLogElement] = field(default_factory=list)

    def add(
        self,
        *,
        agent_name: str,
        kind: str = "event",
        title: str = "",
        summary: str = "",
        details: str = "",
        status: str = "",
        tool_name: str = "",
        step_id: str = "",
        iteration: int | None = None,
        request: Any | None = None,
        response: Any | None = None,
        error: str = "",
        data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.entries.append(
            AgentStateLogElement(
                agent_name=agent_name,
                kind=kind,
                title=title,
                summary=summary,
                details=details,
                status=status,
                tool_name=tool_name,
                step_id=step_id,
                iteration=iteration,
                request=request,
                response=response,
                error=error,
                data={} if data is None else dict(data),
                metadata={} if metadata is None else dict(metadata),
            )
        )

    def clone(self) -> AgentStateLog:
        return AgentStateLog(
            entries=[
                AgentStateLogElement(
                    agent_name=entry.agent_name,
                    kind=entry.kind,
                    title=entry.title,
                    summary=entry.summary,
                    details=entry.details,
                    status=entry.status,
                    tool_name=entry.tool_name,
                    step_id=entry.step_id,
                    iteration=entry.iteration,
                    request=entry.request,
                    response=entry.response,
                    error=entry.error,
                    data=dict(entry.data),
                    metadata=dict(entry.metadata),
                )
                for entry in self.entries
            ]
        )

    def to_grouped_dict(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for entry in self.entries:
            grouped.setdefault(entry.agent_name, []).append(entry.to_payload())
        return grouped


@dataclass
class IterationState:
    plan: Plan | None = None
    results: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        *,
        plan: Plan | None = None,
        results: dict[str, Any] | None = None,
    ) -> "IterationState":
        return cls(
            plan=plan,
            results={} if results is None else results,
        )

    def clone(self) -> IterationState:
        return IterationState(
            plan=None if self.plan is None else self.plan.model_copy(deep=True),
            results=dict(self.results),
        )


@dataclass
class SubagentState:
    task: str
    max_turns: int
    agent_profile: AgentProfile
    request_analysis: RequestAnalysis = field(default_factory=RequestAnalysis)
    iteration_trace: list[IterationState] = field(default_factory=list)
    agent_log: AgentStateLog = field(default_factory=AgentStateLog)
    result: AgentResult = field(default_factory=lambda: AgentResult(answer=[]))
    relevant_evidence_ids: list[str] = field(default_factory=list)
    evaluation_status: EvaluationStatus = EVALUATION_STATUS_RETRYABLE
    goal_reached: bool = False

    def to_runtime_state(self, parent_state: AgentState) -> AgentState:
        return AgentState(
            task=self.task,
            max_turns=self.max_turns,
            conversation_context=parent_state.conversation_context,
            user_profile=parent_state.user_profile,
            agent_profile=self.agent_profile,
            conversation_id=parent_state.conversation_id,
            roundtrip_id=parent_state.roundtrip_id,
            request_analysis=self.request_analysis.model_copy(deep=True),
            iteration_trace=[iteration.clone() for iteration in self.iteration_trace],
            agent_log=self.agent_log.clone(),
            result=self.result,
            relevant_evidence_ids=list(self.relevant_evidence_ids),
            evaluation_status=self.evaluation_status,
            goal_reached=self.goal_reached,
            llm=parent_state.llm,
            conversation_model_config=parent_state.conversation_model_config.model_copy(deep=True),
        )

    def update_from_runtime_state(self, runtime_state: AgentState) -> SubagentState:
        self.task = runtime_state.task
        self.max_turns = runtime_state.max_turns
        self.agent_profile = runtime_state.agent_profile
        self.request_analysis = runtime_state.request_analysis.model_copy(deep=True)
        self.iteration_trace = [iteration.clone() for iteration in runtime_state.iteration_trace]
        self.agent_log = runtime_state.agent_log.clone()
        self.result = runtime_state.result
        self.relevant_evidence_ids = list(runtime_state.relevant_evidence_ids)
        self.evaluation_status = runtime_state.evaluation_status
        self.goal_reached = runtime_state.goal_reached
        return self

    def clone(self) -> SubagentState:
        return SubagentState(
            task=self.task,
            max_turns=self.max_turns,
            agent_profile=self.agent_profile,
            request_analysis=self.request_analysis.model_copy(deep=True),
            iteration_trace=[iteration.clone() for iteration in self.iteration_trace],
            agent_log=self.agent_log.clone(),
            result=self.result,
            relevant_evidence_ids=list(self.relevant_evidence_ids),
            evaluation_status=self.evaluation_status,
            goal_reached=self.goal_reached,
        )


@dataclass
class AgentState:
    task: str
    max_turns: int
    conversation_context: ConversationContext = field(default_factory=ConversationContext)
    user_profile: UserProfile = field(default_factory=UserProfile)
    agent_profile: AgentProfile = field(default_factory=lambda: MAIN_AGENT_PROFILE)
    conversation_id: str | None = None
    roundtrip_id: UUID | None = None
    request_analysis: RequestAnalysis = field(default_factory=RequestAnalysis)
    iteration_trace: list[IterationState] = field(default_factory=list)
    subagent_states: dict[str, SubagentState] = field(default_factory=dict)
    agent_log: AgentStateLog = field(default_factory=AgentStateLog)
    result: AgentResult = field(default_factory=lambda: AgentResult(answer=[]))
    relevant_evidence_ids: list[str] = field(default_factory=list)
    evaluation_status: EvaluationStatus = EVALUATION_STATUS_RETRYABLE
    goal_reached: bool = False
    llm: Any = field(default_factory=lambda: ChatOpenAI(model=ConversationModelConfig.default_main_agent_planner_model()), repr=False)
    conversation_model_config: ConversationModelConfig = field(default_factory=ConversationModelConfig.build_default)

    @classmethod
    def new(
        cls,
        task: str,
        max_turns: int,
        conversation_context: ConversationContext | None = None,
        user_profile: UserProfile | None = None,
        agent_profile: AgentProfile | None = None,
        conversation_id: str | None = None,
        roundtrip_id: UUID | None = None,
        llm: Any | None = None,
        conversation_model_config: ConversationModelConfig | None = None,
    ) -> "AgentState":
        return cls(
            task=task,
            max_turns=max_turns,
            conversation_context=ConversationContext() if conversation_context is None else conversation_context,
            user_profile=UserProfile() if user_profile is None else user_profile,
            agent_profile=MAIN_AGENT_PROFILE if agent_profile is None else agent_profile,
            conversation_id=conversation_id,
            roundtrip_id=roundtrip_id,
            llm=ChatOpenAI(model=ConversationModelConfig.default_main_agent_planner_model()) if llm is None else llm,
            conversation_model_config=ConversationModelConfig.build_default() if conversation_model_config is None else conversation_model_config,
        )

    def add_iteration(self, iteration: IterationState) -> IterationState:
        self.iteration_trace.append(iteration)
        return iteration

    def get_subagent_state(
        self,
        agent_profile: AgentProfile,
        *,
        task: str | None = None,
        max_turns: int | None = None,
    ) -> SubagentState:
        existing_state = self.subagent_states.get(agent_profile.name)
        if existing_state is not None:
            if task is not None:
                existing_state.task = task
            if max_turns is not None:
                existing_state.max_turns = max_turns
            existing_state.agent_profile = agent_profile
            return existing_state

        subagent_state = SubagentState(
            task=self.task if task is None else task,
            max_turns=self.max_turns if max_turns is None else max_turns,
            agent_profile=agent_profile,
        )
        self.subagent_states[agent_profile.name] = subagent_state
        return subagent_state

    def clone_for_parallel(self) -> AgentState:
        return AgentState(
            task=self.task,
            max_turns=self.max_turns,
            conversation_context=self.conversation_context,
            user_profile=self.user_profile,
            agent_profile=self.agent_profile,
            conversation_id=self.conversation_id,
            roundtrip_id=self.roundtrip_id,
            request_analysis=self.request_analysis.model_copy(deep=True),
            iteration_trace=[iteration.clone() for iteration in self.iteration_trace],
            subagent_states={name: subagent_state.clone() for name, subagent_state in self.subagent_states.items()},
            agent_log=self.agent_log.clone(),
            result=self.result,
            relevant_evidence_ids=list(self.relevant_evidence_ids),
            evaluation_status=self.evaluation_status,
            goal_reached=self.goal_reached,
            llm=self.llm,
            conversation_model_config=self.conversation_model_config.model_copy(deep=True),
        )

    def log_status(
        self,
        *,
        agent_name: str,
        kind: str = "event",
        title: str = "",
        summary: str = "",
        details: str = "",
        status: str = "",
        tool_name: str = "",
        step_id: str = "",
        iteration: int | None = None,
        request: Any | None = None,
        response: Any | None = None,
        error: str = "",
        data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.agent_log.add(
            agent_name=agent_name,
            kind=kind,
            title=title,
            summary=summary,
            details=details,
            status=status,
            tool_name=tool_name,
            step_id=step_id,
            iteration=iteration,
            request=request,
            response=response,
            error=error,
            data=data,
            metadata=metadata,
        )

    def build_agent_logs(self) -> dict[str, list[dict[str, Any]]]:
        logs = self.agent_log.to_grouped_dict()
        for subagent_state in self.subagent_states.values():
            for agent_name, entries in subagent_state.agent_log.to_grouped_dict().items():
                logs.setdefault(agent_name, []).extend(entries)
        return logs

    def resolve_model_for_stage(self, *, agent: str, stage: str) -> str:
        return self.conversation_model_config.resolve(agent, stage)

    def resolve_agent_scope(self) -> str:
        if self.agent_profile.name == "profile_management":
            return PROFILE_AGENT_MODEL_SCOPE
        return MAIN_AGENT_MODEL_SCOPE

    def build_llm_for_stage(self, *, stage: str, agent: str | None = None) -> Any:
        if not isinstance(self.llm, ChatOpenAI):
            return self.llm
        resolved_agent = self.resolve_agent_scope() if agent is None else agent
        model_name = self.resolve_model_for_stage(agent=resolved_agent, stage=stage)
        if resolved_agent == MAIN_AGENT_MODEL_SCOPE and stage == PLANNER_STAGE and model_name == ConversationModelConfig.default_main_agent_planner_model():
            return self.llm
        return ChatOpenAI(model=model_name)



