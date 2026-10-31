from __future__ import annotations

from dataclasses import dataclass

from personalization.profile.service import build_user_profile
from agent_runtime.models.agent_execution_context import AgentExecutionContext
from agent_runtime.models.agent_state import AgentState
from agent_runtime.models.plan import Plan
from agent_runtime.planning.planner import invoke_planner
from agent_runtime.planning.prompts.planner_prompt import build_planner_prompt
from agents.main.request import TOP_LEVEL_PROFILE



@dataclass(frozen=True)
class JobOrchestrator:
    """orchestrator for job plan creation."""

    def generate_plan(self, *, user_id: str, prompt: str) -> tuple[Plan | None, str | None]:
        if not user_id.strip():
            raise ValueError("user_id is required")
        if not prompt.strip():
            raise ValueError("prompt is required")

        profile = build_user_profile(user_id=user_id)
        execution_context = AgentExecutionContext.new(user_profile=profile)
        state = AgentState.new(
            TOP_LEVEL_PROFILE,
            task=prompt.strip(),
            execution_context=execution_context,
        )
        planner_prompt, tools = build_planner_prompt(state)
        planner_messages = planner_prompt.build_messages()

        try:
            planning_result, _ = invoke_planner(
                state,
                planner_messages,
                tools=tools,
                prompt_input_object={"messages": planner_messages},
            )
        except Exception as exc:
            return None, f"plan generation failed: {exc}"

        if planning_result.status != "ready" or planning_result.needs_replan or not planning_result.steps:
            reason = planning_result.reason or "planner did not produce a usable plan"
            return None, reason
        return Plan(steps=planning_result.steps), None
