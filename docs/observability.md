# Observability
The repo now has a fairly deep observability layer across prompts, LLM calls, tool execution, roundtrips, and final evidence attribution.

## What We Record
At a high level, each turn can produce observable data in a few layers:
1. Agent-stage logs such as request analysis, profile hydration, planning, evaluation, synthesis, and tool calls.
2. Stored prompt rows for important LLM stages.
3. Structured LLM call records with usage, cost, latency, and prompt metadata.
4. Roundtrip-level payloads with summaries, tool summaries, evidence attribution, and total turn duration.
5. Hydrated evidence attached to final results so source rendering can stay typed.

## Agent Logs
The Streamlit debug experience renders structured agent logs rather than raw text blobs.

Current log types include:
1. `request_analysis`
2. `profile_load`
3. `plan`
4. `evaluator`
5. `tool_call`
6. `synthesis`

These logs are shaped into typed payloads in `rendering/debug.py`, which keeps the UI consistent and makes the logs easier to scan as JSON.

## Prompt Observability
Prompt construction is observable at the section level.

`AgentPrompt` currently records:
1. The full rendered prompt text.
2. Total prompt token count.
3. Per-section prompt metadata including:
   - section key
   - heading
   - text
   - token count

That means we can inspect not just the final prompt, but also which sections were included and how expensive each section was.

Important LLM-stage prompts are also stored as roundtrip prompt rows, including:
1. request analysis
2. planner
3. evaluator
4. synthesis

## LLM Call Records
Each recorded LLM call can include:
1. `agent`
2. `stage`
3. `callsite`
4. `model`
5. token usage
6. computed input, output, and total cost
7. `latency_ms`
8. normalized `input_object`
9. normalized `output_object`
10. extra metadata

This is handled through `llm/usage.py`.

At the roundtrip level, these calls are also rolled up into an `llm_usage` payload with:
1. total input, output, cached, and overall tokens
2. total latency across recorded LLM calls
3. total computed cost
4. per-call serialized records

## Tool Execution Observability
Tool execution is also tracked in a structured way.

Current tool-call observability includes:
1. the plan step and tool name
2. step id and iteration
3. tool request payload
4. tool response payload
5. error details when a tool fails
6. `latency_ms`
7. extra metadata

The executor records per-tool latency around the actual wrapped tool call, so the timing reflects the real invocation path rather than just the raw client call.

## Rate Limits And Retries
The tool registry also contains lightweight runtime protections that are worth treating as part of observability:
1. per-key rate limiting
2. retry policies
3. emitted debug exceptions when tool execution fails

This makes it easier to understand whether a tool failure was caused by bad inputs, external 429s, timeouts, or 5xx-style issues.

## Roundtrip Observability
At the end of a turn, the roundtrip payload can include:
1. final response text
2. structured answer blocks
3. `next_question`
4. `roundtrip_summary`
5. `tool_summary`
6. `agent_logs`
7. `used_evidence_ids`
8. `hydrated_evidence_by_id`
9. `llm_usage`
10. `roundtrip_latency_ms`

That gives one place to inspect both the user-visible outcome and the supporting execution details behind it.

## Evidence Attribution
Observability now also extends into answer attribution.

The final synthesized result can carry:
1. `used_evidence_ids` for the evidence referenced by the answer
2. `hydrated_evidence_by_id` for the typed full evidence objects behind those ids

That supports:
1. source rendering in the UI
2. source inspection through the sources modal
3. later work around paragraph-level attribution and richer rendering

## UI Surfaces
Today the main observability surfaces are:
1. debug log rendering in the Streamlit UI
2. turn-level usage, cost, and duration display
3. sources rendering backed by hydrated evidence
4. persisted roundtrip payloads and prompt rows for replay or inspection

## Why It Matters
This observability layer exists so we can answer questions like:
1. What prompt was actually sent?
2. Which sections contributed most to prompt size?
3. How long did each model call take?
4. How long did the whole roundtrip take?
5. Which tools ran and how long did they take?
6. What evidence supported the final answer?
7. Why did the system replan, retry, or stop?
