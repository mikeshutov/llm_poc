# Memories
One interesting area in the repo is allowing the agent to recall prior conversations or previously discussed topics. The current approach is intentionally straightforward and uses embeddings in a similar way to file retrieval.

## Current Shape
1. We store conversation summaries and an embedding for each summary.
2. We also store roundtrip summaries and embeddings for those individual back-and-forths.

That gives us two different memory levels:
1. A conversation-level memory that acts like a persistent high-level state for a conversation.
2. A roundtrip-level memory that captures a specific exchange.

## Tools
This currently comes together through two tools:
1. `search_memories` searches across past conversations using conversation summaries and returns a small set of relevant conversation memories.
2. `search_roundtrip_memories` searches detailed roundtrips after a relevant conversation is identified and returns matching turn-level memories.

## Why It Exists
The goal is to let the agent find relevant information from prior conversations on demand rather than injecting large amounts of historical data into every prompt.

The pattern is:
1. Summaries are converted into embeddings.
2. Queries are embedded at runtime.
3. Similarity search retrieves the most relevant memories.
4. Results are ordered by relevance and can include a score to help identify the best matches.

This keeps memory retrieval closer to a retrieval problem than a giant prompt-history problem.
