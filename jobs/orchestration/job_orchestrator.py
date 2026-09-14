from __future__ import annotations

from dataclasses import dataclass

from personalization.profile.service import build_user_profile
from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE
from request_orchestrator.models.agent_execution_context import AgentExecutionContext
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.plan import Plan
from request_orchestrator.shared.planner.planner import invoke_planner
from request_orchestrator.shared.planner.prompts.planner_prompt import build_planner_prompt



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
            MAIN_AGENT_PROFILE,
            task=prompt.strip(),
            execution_context=execution_context,
        )
        planner_prompt = build_planner_prompt(state)

        try:
            planning_result, _ = invoke_planner(
                state,
                planner_prompt.build(),
                prompt_input_object=planner_prompt.to_log_input_object(),
            )
        except Exception as exc:
            return None, f"plan generation failed: {exc}"

        if planning_result.status != "ready" or planning_result.needs_replan or not planning_result.steps:
            reason = planning_result.reason or "planner did not produce a usable plan"
            return None, reason
        return Plan(steps=planning_result.steps), None
