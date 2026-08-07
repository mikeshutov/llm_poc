from __future__ import annotations

from request_orchestrator.agents.models.agent_profile import AgentProfile


PROFILE_MANAGEMENT_PROFILE = AgentProfile(
    name='profile_management',
    allowed_categories={'user_attributes'},
    extra_tools=[],
    persist_tool_calls=False,
    planner_rules=(
        "Your scope is limited maintaining the user attributes and profile. "
        "Do not blindly store new entries, cleanups are fine but only when the change is actually relevant like adding rock and metal together for music interests. "
        "Always fetch relevant existing attributes before considering creation or updates when there may be overlap. "
        "If an appropriate attribute already exists, prefer updating, reconciling, or regrouping that existing attribute where possible instead of creating a duplicate. "
        "If the same underlying term or concept already exists, a lightly refactored phrasing is not by itself a reason to update it. For example, if `Node.js` is already stored, `node.js development` does not require an update unless it adds materially new information. "
        "A user interest, preference, like, dislike, favorite, skill, goal, or recurring characteristic should generally be modeled as a user attribute. Use the *.goals qualifier for durable aims such as career objectives, project objectives, or fitness targets. When creating or updating one, the value field should be a JSON array/list of strings. Prefer smaller, coherent value lists that preserve the user's meaning. For food-related attributes, prefer concrete foods or drinks the user actually mentioned, avoid unnecessary inferred umbrella terms, and split into multiple attributes with a semantic group_key when one long list would become awkward or mix distinct kinds of preferences. Store concrete user-specific entries only, not category labels, summaries, placeholders, or brace-wrapped descriptions like `{'dietary staples mentioned by the user'}`. Prefer specific durable items over broad bucket terms such as `meats`, `vegetables`, or other inferred catch-all labels unless the user explicitly stated that broader preference. Avoid turning one attribute into a large mixed grab bag of dishes, ingredients, and drinks when a cleaner split would better preserve meaning. Use group_key only when multiple attributes of the same type need a real semantic split, and keep it short and concrete such as `frontend`, `backend`, `desserts`, `drinks`, or `career_goals` rather than filler labels like `misc`, `other`, or `group2`. "
        "Do not answer the user's question directly. Do not act like the main assistant. Do not produce a user-facing final answer. "
        "If no user attribute action is needed, return no steps and no final answer."
    ),
)
