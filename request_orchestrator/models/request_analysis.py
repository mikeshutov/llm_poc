from __future__ import annotations

from pydantic import BaseModel


class RequestAnalysisGoal(BaseModel):
    agent: str = ""
    goal: str = ""
    tool_categories: list[str] = []


class RequestAnalysis(BaseModel):
    goals: list[RequestAnalysisGoal] = []
    requested_user_attribute_types: list[str] = []

    def _goal_entry_for_agent(self, agent_name: str) -> RequestAnalysisGoal | None:
        normalized_agent_name = agent_name.strip()
        for goal_entry in self.goals:
            if goal_entry.agent.strip() == normalized_agent_name:
                return goal_entry
        return None

    def goal_for_agent(self, agent_name: str, default: str = "") -> str:
        goal_entry = self._goal_entry_for_agent(agent_name)
        if goal_entry is None:
            return default
        return goal_entry.goal.strip()

    def tool_categories_for_agent(self, agent_name: str) -> list[str]:
        goal_entry = self._goal_entry_for_agent(agent_name)
        if goal_entry is None:
            return []
        return [
            category
            for category in goal_entry.tool_categories
            if isinstance(category, str) and category.strip()
        ]

    def set_goal_for_agent(self, agent_name: str, goal: str, *, tool_categories: list[str] | None = None) -> None:
        goal_entry = self._goal_entry_for_agent(agent_name)
        normalized_agent_name = agent_name.strip()
        normalized_goal = goal.strip()
        normalized_tool_categories = [] if tool_categories is None else [
            category.strip()
            for category in tool_categories
            if isinstance(category, str) and category.strip()
        ]
        if goal_entry is not None:
            goal_entry.goal = normalized_goal
            if tool_categories is not None:
                goal_entry.tool_categories = normalized_tool_categories
            return
        self.goals.append(
            RequestAnalysisGoal(
                agent=normalized_agent_name,
                goal=normalized_goal,
                tool_categories=normalized_tool_categories,
            )
        )
