# Request Analysis
Request analysis exists to narrow the problem before planning starts.

## Why It Exists
As the number of tools grows, passing the full tool list to the planner becomes less useful and more expensive. Request analysis is the first pass that decides what the request is really about and what downstream context is actually needed.

## Current Responsibilities
Request analysis currently does three main things:
1. Determines the user's goal in a self-contained way so downstream steps do not need the full conversation history to understand the task.
2. Selects the applicable tool categories for the current request.
3. Requests any durable user attribute types that would materially help with the request.

## Practical Effect
The flow is:
1. `request_analysis` reads the latest prompt plus structured conversation context and a lightweight user profile.
2. It produces a refined goal, tool-category selection, and requested attribute types.
3. We then load only the requested profile attributes.
4. The planner receives a narrower tool set and a more relevant profile slice.

## Why The Goal Matters
The refined goal is intentionally self-contained. It should carry forward:
1. The real objective.
2. Any important conversation-derived constraints.
3. Named entities, references, or continuity that later planning and synthesis still need.

That matters because the full conversation context is not passed unchanged through every later step.

## Why Attribute Selection Matters
Durable user attributes are not preloaded in full. Instead, request analysis asks for specific types such as `food.likes` or `projects.goals` only when they are likely to help.

This keeps prompts smaller and lets profile hydration stay selective rather than eager.
