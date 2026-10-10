# Completion evaluator gate

The evaluator has an optional local completion/answerability fast path backed by the natural-language-inference model `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.

After tool execution, the application sends the evaluator evidence as the premise and the current request as an answerability hypothesis. A high-confidence entailment produces a `SATISFIED` evaluation directly. Neutral, contradictory, low-confidence, malformed, or unavailable NLI results fall back to the existing LLM evaluator.

Start the service with:

```text
docker compose up -d completion-evaluator
```

Configuration:

- `COMPLETION_EVALUATOR_SERVICE_URL` — application URL, default `http://localhost:5435`
- `COMPLETION_EVALUATOR_TIMEOUT_SECONDS` — client timeout, default `15`
- `COMPLETION_EVALUATOR_ENTAILMENT_THRESHOLD` — fast-path threshold, default `0.85`
- `COMPLETION_EVALUATOR_MODEL` — service model name, default `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
