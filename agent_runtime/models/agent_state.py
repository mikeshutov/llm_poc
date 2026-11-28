from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from agent_runtime.compaction import should_compact_tool
from agent_runtime.working_memory import (
    MAX_WORKING_MEMORY_BYTES,
    bound_working_memory,
    render_working_memory_template,
    working_memory_path,
)
from common.config import UTF8_ENCODING

from llm.chat_models import build_chat_model, resolve_stage_model_selection
from llm.conversation_model_config import ConversationModelConfig
from agent_runtime.models.agent_profile import AgentProfile
from agent_runtime.models.agent_inputs import AgentInputs
from agent_runtime.models.plan import Plan
from agent_runtime.models.synthesized_result import SynthesisResult
from .agent_execution_context import AgentExecutionContext
from .agent_result import AgentResult
from retrieval.evidence import ToolResult
from agent_runtime.models.node_state import AgentNodeStates
from capabilities.registry.repository import message_repository

@dataclass
class AgentState:
    agent_profile: AgentProfile
    inputs: AgentInputs = field(default_factory=AgentInputs)
    available_agent_states: dict[str, "AgentState"] | None = None
    direct_tool_results: list[ToolResult] = field(default_factory=list)
    executed_request_signatures: list[str] = field(default_factory=list)
    execution_context: AgentExecutionContext = field(default_factory=AgentExecutionContext)
    node_states: AgentNodeStates = field(default_factory=AgentNodeStates)
    result: AgentResult = field(default_factory=AgentResult)
    reasoner_result: SynthesisResult | None = None
    reasoner_message_history: list[Any] = field(default_factory=list)
    working_memory_file_name: str | None = None
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
        planner_selection = resolve_stage_model_selection(
            execution_context=resolved_execution_context,
            agent=resolved_agent_scope,
            stage="planner",
            agent_profile=agent_profile,
        )
        return cls(
            inputs=AgentInputs.new(task=task) if inputs is None else inputs,
            execution_context=resolved_execution_context,
            agent_profile=agent_profile,
            llm=(
                build_chat_model(
                    provider=planner_selection.provider,
                    model_name=planner_selection.model,
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

    def render_working_memory(self) -> str:
        """Render the profile's runtime memory template for this task."""
        return bound_working_memory(self.read_working_memory() or render_working_memory_template(self))

    def read_working_memory(self) -> str:
        """Return the initialized runtime memory file without template decoration."""
        path = working_memory_path(self)
        if path is not None:
            try:
                with path.open(encoding=UTF8_ENCODING) as memory_file:
                    return bound_working_memory(memory_file.read(MAX_WORKING_MEMORY_BYTES + 1)).strip()
            except (OSError, UnicodeError):
                return ""
        return ""

    def condense_reasoner_history(self, results: list[ToolResult]) -> None:
        """Persist compaction and remove compactable exchanges from active history."""
        omitted_call_ids = {
            str(call.get("id", ""))
            for message in self.reasoner_message_history
            if isinstance(message, AIMessage)
            for call in message.tool_calls or []
            if (
                call.get("id")
                and should_compact_tool(str(call.get("name", "")))
            )
        }
        if not omitted_call_ids:
            return

        if self.result.tool_call_ids:
            repository = message_repository.MessageToolRepository()
            repository.mark_compacted([
                result.tool_call_id
                for result in results
                if result.tool_call_id is not None and should_compact_tool(result.tool_name)
            ])

        condensed_history: list = []
        for message in self.reasoner_message_history:
            if isinstance(message, ToolMessage):
                if str(message.tool_call_id) in omitted_call_ids:
                    continue
                condensed_history.append(message)
                continue
            if isinstance(message, AIMessage) and message.tool_calls:
                remaining_calls = [
                    call for call in message.tool_calls
                    if str(call.get("id", "")) not in omitted_call_ids
                ]
                if not remaining_calls:
                    continue
                if len(remaining_calls) != len(message.tool_calls):
                    condensed_history.append(message.model_copy(update={"tool_calls": remaining_calls}))
                    continue
            condensed_history.append(message)
        self.reasoner_message_history[:] = condensed_history

    def gather_tool_results(self) -> list[ToolResult]:
        gathered = list(self.direct_tool_results)
        if self.result.tool_call_ids:
            gathered.extend(message_repository.MessageToolRepository().get_tool_results(self.result.tool_call_ids))
        return gathered

    def gather_tool_calls(self):
        gathered = []
        if self.result.tool_call_ids:
            gathered.extend(message_repository.MessageToolRepository().get_tool_calls(self.result.tool_call_ids))
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
