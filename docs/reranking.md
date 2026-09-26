# Reranking
The repo uses a dedicated local `BAAI/bge-reranker-v2-m3` service to improve ordering after retrieval without forcing every retrieval source to own its own ranking logic.

## Current Shape
1. Retrieval produces `Candidate` objects plus the original domain models such as `ProductResult`.
2. Each source maps its entity into a `Candidate` with semantic content and attributes.
3. The application sends a query and richer, token-budgeted candidate evidence to the local `/rerank` endpoint.
4. The reranker returns numeric scores in the form `{ "results": [{"id": "...", "score": 0.8}] }`.
5. The reranker service rebuilds the ranked output from those returned ids.
6. If the number of candidates is already at or below the configured top-k limit, the reranker call is skipped entirely.

## Important Details
1. The top-k limit is standardized in `reranker/constants.py` and is currently `10`.
2. Candidate evidence construction is handled through the generic evidence builder, while each mapper supplies its domain fields.
3. URLs, images, retrieval metadata, and embeddings are excluded from semantic reranker input.
4. Candidate evidence is deterministically constrained by `RERANKER_CANDIDATE_TOKEN_BUDGET`.
5. Product web results currently use their URL as the id when no better external identifier is available yet.

The Docker service is exposed on `localhost:5433` for the local application. Set `RERANKER_SERVICE_URL=http://reranker:8080` when the application itself runs inside the Compose network.

TODO: replace the approximate application-side character budget with tokenizer-aware budgeting that reserves space for the query, reports truncation, and aligns exactly with `RERANKER_MAX_LENGTH`.

## Separation Of Concerns
The design keeps three concerns separate:
1. Retrieval can stay domain-specific.
2. The reranker can stay generic.
3. Downstream code can still work with the original result objects after ranking.

That means more candidate sources can plug into the same reranking contract over time.
