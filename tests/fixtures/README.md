# Shared test fixtures

Fixtures are grouped by topic here and registered in `tests/conftest.py`.
Every test section can request any fixture by argument name: capability tests can
use agent fixtures, product tests can use retrieval fixtures, and so on. No fixture
imports or imports from other test modules are needed.

Find available fixtures with `python -m pytest --fixtures tests/capabilities`, or
search this directory for the topic or fixture name.

## Topics

| Module | Fixtures |
| --- | --- |
| `agents.py` | `agent_profile_factory`, `user_agent_profile_factory`, `execution_context_factory`, `agent_state_factory`, `react_agent_state_factory`, `main_state_factory`, `db_agent_factory`, `set_plan_results` |
| `planning.py` | `plan_step_factory`, `plan_factory`, `default_plan`, `planning_result_factory` |
| `capabilities.py` | `tool_factory`, `cache_repository_factory` |
| `tools.py` | `registered_tool_registry`, `tool_repository_factory`, `recording_tool_repository` |
| `conversation.py` | `conversation_message_factory`, `conversation_roundtrip_factory`, `context_repository_factory`, `conversation_repository_factory`, `recording_conversation_repository_factory`, `request_conversation_repository_factory`, `conversation_model_config_repository_factory` |
| `database.py` | `database_connection_factory` |
| `llm.py` | `llm_factory`, `langchain_response_factory`, `openai_response_factory`, `openai_client_factory`, `embedding_client_factory`, `tracking_openai_model_factory`, `tracking_anthropic_model_factory` |
| `evidence.py` | `evidence_factory`, `evidence_step_factory` |
| `evaluation.py` | `completion_client_factory` |
| `retrieval.py` | `reranker_factory`, `static_reranker_factory` |
| `products.py` | `product_factory`, `product_search_results_factory` |
| `personalization.py` | `user_attribute_factory` |
| `attribute_classifier.py` | `attribute_classifier_factory` |
| `assets.py` | `file_asset_factory`, `writing_catalog_factory` |
| `http.py` | `http_response_factory` |
| `jobs.py` | `job_planner_factory` |
| `games.py` | `commander_page_factory` |

## Compose defaults and overrides

Factories return fresh objects. Use a default for a small test, or compose builders
and override fields for a larger scenario. An explicit empty collection stays empty;
for example, `plan_factory(steps=[])` does not create a default step.

```python
def test_capability(agent_state_factory, plan_factory, plan_step_factory):
    step = plan_step_factory(id="lookup", tool="web_fetch", args={"url": "https://example.com"})
    plan = plan_factory(steps=[step])
    state = agent_state_factory(
        plan=plan,
        task="Read the source",
        profile_overrides={"max_turns": 5},
    )
    assert state.node_states.planner.plan.steps[0].tool == "web_fetch"
```

`plan_factory()` and `default_plan` provide a one-step web-search plan.
`plan_step_factory` and `plan_factory` copy supplied mutable data, so assigning
step indices or changing one plan's arguments does not mutate the caller's steps.

`agent_state_factory` accepts an existing `profile`, `inputs`, `execution_context`,
and `llm`; `profile_overrides` copies the supplied profile instead of mutating it.
Without a profile, a supplied plan provides the static tool names. The default LLM
is an inert object, so constructing a test state does not initialize a provider.
Use `agent_state_factory(user_id="alice")` for user-scoped runtime context. When
given an existing execution context, the owner override copies its user profile
without changing the supplied context. `user_agent_profile_factory(name="helper",
description="Research helper")` defaults to a user-agent profile in the main scope
and accepts the same profile overrides as `agent_profile_factory`.
Use `execution_context_factory(user_id="alice", conversation_id="conversation-1",
roundtrip_id=uuid4())` when context identifiers matter. Omitted identifiers stay
unset; user profiles and conversation contexts are fresh on every call.
`db_agent_factory` supplies an ID, name, and instruction; specify ownership fields
explicitly, such as `user_id="alice"` or `agent_type=AgentType.SYSTEM`.

`llm_factory` accepts response content, tool calls, and token counts, and
records received prompts in `prompts`.

## Scope and isolation

All fixtures are function-scoped. Tool-registry isolation is automatic; repository, HTTP, and provider substitution is opt-in. Repository factories that
install executor/state replacements make the latest created instance active.

Keep constructor/default/validation tests explicit when that constructor is the
behavior under test. Keep scenario-specific setup and assertions local; do not turn
fixtures into a second implementation of production behavior. Path/security tests should exercise real path resolution.
