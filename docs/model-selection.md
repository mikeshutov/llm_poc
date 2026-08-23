# Model Selection
Model selection in this repo is set up to allow you select a model per step and there is an edit section that lets you select the models. Additionally when you click on replay you may select different models before you replay that particular message.

## Current Shape
The system can use different models for different parts of the loop. Today the configured stages are:
1. `main_agent.request_analysis`
2. `main_agent.planner`
3. `main_agent.synthesis`
4. `profile_agent.planner`
5. `shared.evaluator`
6. `shared.reranker`

That means request analysis, planning, synthesis, evaluation, and reranking do not have to share the same model.

## Defaults
The defaults currently live in `llm/conversation_model_config.py`.

The main pattern is:
1. `gpt-5.6-luna` is used by default for every stage.

Current default variables include:
1. `LLM_MODEL`
2. `MAIN_AGENT_PLANNER_MODEL`
3. `MAIN_AGENT_SYNTHESIS_MODEL`

If stage-specific environment variables are not set, the config falls back to the default values defined in `ConversationModelConfig`.

## Per-Conversation Configuration
Model selection is resolved per conversation, not just per process.

The flow is:
1. The repository loads any persisted `conversation_model_config` entries for the conversation.
2. Missing stages are filled with defaults.
3. The final resolved config becomes a `ConversationModelConfig` object.
4. That resolved config is passed into the orchestrator for the turn.

This lets different conversations use different stage/model combinations without requiring one global setting for everything.

## How Stages Resolve Models
`AgentState` resolves the model for a stage through the conversation model config.

In practice:
1. Request analysis asks for the configured `main_agent.request_analysis` model.
2. Main planning asks for `main_agent.planner`.
3. Synthesis asks for `main_agent.synthesis`.
4. Profile management planning asks for `profile_agent.planner`.
5. Evaluator asks for `shared.evaluator`.
6. Reranker asks for `shared.reranker`.

This keeps model choice close to the purpose of the stage rather than treating the whole request as one uniform LLM call.

## Persistence And Metadata
At the start of a roundtrip, the resolved conversation model config is attached to the pending roundtrip metadata.

That matters because:
1. the turn can later be inspected with the exact resolved stage configuration
2. observability can tie costs and latencies back to the model choices actually used for that roundtrip

## Pricing And Cost Tracking
Model pricing also lives alongside the conversation model config.

`ConversationModelConfig` maintains a pricing registry and resolves pricing by model name. It also normalizes snapshot-style model names so cost tracking still works when a model name includes a dated suffix.

That pricing data is used by `llm/usage.py` to compute:
1. input cost
2. output cost
3. total cost

for each recorded LLM call.

## Why This Exists
Basically by having this as a feature I can easily see the costs per step. I can also see how well it works out in terms of input/output. I can utilize replay as well as model selection to see how different models effect results of different steps.

Separating model selection by stage makes it easier to tune for:
1. latency
2. cost
3. answer quality
4. reasoning quality
