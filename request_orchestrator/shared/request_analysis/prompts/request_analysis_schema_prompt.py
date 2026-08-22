REQUEST_ANALYSIS_SCHEMA = """
Set "goals" to an array of goal objects. Each goal object must contain:
- "agent": one of the provided available agent names
- "goal": a self-contained statement of what that particular agent is trying to accomplish
- "tool_categories": the relevant category names for that agent

Include any relevant conversation-derived constraints, continuity, entities, or references needed by downstream planning and synthesis because the full conversation context will not be passed through later.
If the request is about something previously discussed, suggested, decided, or mentioned, include "memories" in the relevant goal's tool_categories.
Include "user_attributes" in a goal's tool_categories only when loading stored user attributes may be useful for answering the request.
Set "requested_user_attribute_types" to an array of specific user attribute types that would be helpful to load for this request. Use only valid prefix.suffix combinations such as "food.likes" or "projects.goals". Leave it as an empty array when stored user attributes are not needed. Storing or updating attributes is handled separately by the profile management agent.
Attempt to break down the task between multiple agents when appropriate

Response JSON shape:
{
  "goals": [
    {
      "agent": "main_agent",
      "goal": "agent specific goal",
      "tool_categories": ["food"]
    }
  ],
  "requested_user_attribute_types": ["food.likes"]
}
"""
