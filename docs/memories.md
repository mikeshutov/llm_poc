# Memories
One interesting area in the repo is allowing the agent to recall prior conversations or previously discussed topics. The current approach is intentionally straightforward and uses embeddings in a similar way to file retrieval.

## Current Shape
1. We store conversation summaries and an embedding for each summary.
2. We also store roundtrip summaries and embeddings for those individual back-and-forths.

That gives us two different memory levels:
1. A conversation-level memory that acts like a persistent high-level state for a conversation.
2. A roundtrip-level memory that captures a specific exchange.

## Tool
`memory.search` supports both retrieval levels. Call it without `conversation_ids` to discover relevant conversations; pass the returned IDs to search detailed roundtrips within those conversations. Use `memory.detail` when the exact prompt, response, or structured payload is needed.

## Why It Exists
The goal is to let the agent find relevant information from prior conversations on demand rather than injecting large amounts of historical data into every prompt.

The pattern is:
1. Summaries are converted into embeddings.
2. Queries are embedded at runtime.
3. An unscoped search retrieves the most relevant conversations; a scoped search retrieves matching turn-level memories.
4. Results are ordered by relevance and can include a score to help identify the best matches.

This keeps memory retrieval closer to a retrieval problem than a giant prompt-history problem.
