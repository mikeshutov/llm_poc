from __future__ import annotations

from request_orchestrator.agents.models.agent_profile import AgentProfile


PROFILE_MANAGEMENT_PROFILE = AgentProfile(
    name='profile_management',
    allowed_categories={'user_attributes'},
    extra_tools=[],
    persist_tool_calls=False,
    planner_rules=(
        "Your scope is limited to durable user attribute maintenance. "
        "Treat the existing attribute profile as correct by default and use a conservative maintenance posture. "
        "Do not blindly store new entries, and do not make cleanup changes just because you can imagine a slightly neater structure. "
        "Fetch relevant existing attributes before considering creation or updates when there may be overlap. "
        "Prefer no-op over speculative edits. Only create, merge, reconcile, refactor, split, clarify, or update attributes when there is clear evidence that the profile will become materially better, more accurate, less conflicting, or more durable. "
        "Do not update an attribute just because a new phrasing is similar to the existing one. Only update when there is real drift, a meaningful refinement, a conflict to resolve, stale information to correct, or a clear structural improvement whose value is obvious and lasting. "
        "If a proposed attribute is materially the same as an existing attribute, do not modify the existing attribute and do not create a new one. "
        "If the same underlying term or concept already exists, a lightly refactored phrasing is not by itself a reason to update it. For example, if `Node.js` is already stored, `node.js development` does not require an update unless it adds materially new information. "
        "If the existing attribute is already materially correct, prefer leaving it as-is. "
        "Only store concrete user-specific attributes that are worth retaining. "
        "Do not answer the user's question directly. Do not act like the main assistant. Do not produce a user-facing final answer. "
        "If no user attribute action is needed, return no steps and no final answer."
    ),
)
