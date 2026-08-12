# Reranking
The repo has a lightweight reranker layer for candidate results. It is used to improve ordering after retrieval without forcing every retrieval source to own its own ranking logic.

## Current Shape
1. Retrieval produces `Candidate` objects plus the original domain models such as `ProductResult`.
2. We send only a reduced candidate payload to the reranker prompt rather than the full object.
3. The reranker returns structured JSON in the form `{ "ranked_ids": ["...", "..."] }`.
4. The reranker service rebuilds the ranked output from those returned ids.
5. If the number of candidates is already at or below the configured top-k limit, the reranker call is skipped entirely.

## Important Details
1. The top-k limit is standardized in `reranker/constants.py` and is currently `6`.
2. Prompt construction is handled through a dedicated `RerankerPrompt` model rather than being assembled ad hoc in the service.
3. Candidate text is intentionally condensed before it is sent to the model.
4. Product web results currently use their URL as the id when no better external identifier is available yet.

## Separation Of Concerns
The design keeps three concerns separate:
1. Retrieval can stay domain-specific.
2. The reranker can stay generic.
3. Downstream code can still work with the original result objects after ranking.

That means more candidate sources can plug into the same reranking contract over time.
