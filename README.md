# Chat powered by an agentic harness
This project was created with the idea of exploring how to build things that utilize LLMs. Over time it has grown from just a simple chatbot that looks at a local product catalog to what it is today. It is now a chat powered by an agentic harness with new capabilities constantly being added as experimenting goes on.

This is not meant to be a production piece of code. It is mostly a place to explore the topic and keep iterating on ideas.

## Table of Contents
- [Project Summary](#project-summary)
- [Main App Flow](#main-app-flow)
- [Documentation](#documentation)
- [Observability](docs/observability.md)
- [Setup](#setup)
- [Notes](#notes)

## Project Summary
At a high level this repo explores:
- Agent orchestration with capability discovery, planning, execution, evaluation, and synthesis
- User-profile hydration and durable attribute and profile management
- Retrieval and reranking across products, web/news, files, and memories
- Streamlit-based UX is realy just for simplicity so we can experiment quickly but a React UI might be needed
- Tool registry and tool policies for agents

## Main App Flow
Rough breakdown of the current agent loop flow.
1. Prompt comes in and we assemble conversation context.
2. We pass that context plus the latest user prompt into agent state.
3. We load likely useful user-created agents for the main planner to consider.
4. We prepare a small `User Profile` section that starts with geo/location-aware metadata plus an initially empty user-attributes section.
5. Capability discovery identifies useful tools and delegatable agents, while attribute discovery identifies user-profile data that may help.
6. We load the discovered profile attributes and condense overlapping records for prompt efficiency.
7. The main agent plans a mixture of direct tool calls and delegated agent tasks.
8. The main agent executes its plan; delegated agents maintain their own planner/executor/evaluator state beneath the main execution state.
9. After execution, the evaluator decides whether another useful planning pass is needed or whether the request can be synthesized.
10. Synthesis generates the final response, roundtrip summary, and tool summary from the completed main-agent state and explicit evidence.

We also store conversations, roundtrips, prompt rows, summaries, and tool calls for future prompts.

```mermaid
flowchart TD
    A[User Prompt] --> B[Build Context And Agent State]
    B --> LUA[Load Likely Useful User Agents]
    LUA --> C[Discover Capabilities And Attributes]
    C --> D[Hydrate Profile]
    subgraph MAR[Main Agent]
        P1[Main Planner] --> X[Main Agent Executor]
        X --> EV[Main Agent Evaluator]
        EV -->|retryable| P1
    end
    D --> P1
    X -.->|delegates| DA[Profile/User Agent State]
    DA --> DX[Delegated Agent Execution]
    DX --> EV
    EV -.->|satisfied or terminal| S[Synthesis]
    PMV -.->|satisfied or terminal| S
    UCA -.->|Agent Result| S
    S --> L[Response]
```

## Documentation
The more detailed documentation now lives in the [`docs/`](docs/) folder.

- [Context Assembly](docs/context-assembly.md)
- [File Searching](docs/file-searching.md)
- [Memories](docs/memories.md)
- [Model Selection](docs/model-selection.md)
- [Observability](docs/observability.md)
- [Request Analysis](docs/request-analysis.md)
- [Reranking](docs/reranking.md)
- [Attribute relevance classifier](docs/attribute-classifier.md)

## Setup
### Prereqs
- Docker + Docker Compose
- Python 3.11+ (uses local `.venv`)
- Environment variables in `.env`

Example `.env`:
```text
DATABASE_URL=postgresql://app:app@localhost:5432/products
OPENAI_API_KEY=...
BRAVE_SEARCH_API_KEY=...
```

### API Keys
- `OPENAI_API_KEY`
- `BRAVE_SEARCH_API_KEY`
- `DEEPSEEK_API_KEY`
- `XAI_API_KEY`
- `MISTRAL_API_KEY`
- `GEMINI_API_KEY`
- `COHERE_API_KEY`

Example with optional provider keys:
```text
DATABASE_URL=postgresql://app:app@localhost:5432/products
OPENAI_API_KEY=...
BRAVE_SEARCH_API_KEY=...
DEEPSEEK_API_KEY=...
XAI_API_KEY=...
MISTRAL_API_KEY=...
GEMINI_API_KEY=...
COHERE_API_KEY=...
```

## Quick Start
1. Start DB
```text
docker compose up -d
```

2. Run DB setup (extensions + schemas + migrations)
```text
python scripts/setup_db.py
```

If you already have a local database from an older clone and just need to move it forward, run:
```text
python scripts/migrate_db.py
```

3. (Optional) Seed products + embeddings
```text
python scripts/seed_products.py
```

4. Start the app
```text
streamlit run main.py
```

## Image Backfill (Optional)
If you already seeded the DB and want to backfill images:
```text
setx ALLOW_IMAGE_BACKFILL 1
python scripts/seed_products.py
```

To force refresh existing images:
```text
setx FORCE_IMAGE_REFRESH 1
python scripts/seed_products.py
```

Product images are stored in `db/images/` for now.
Uploaded files (PDFs, DOCX, images, etc.) are stored in `static/files/`.

## Notes
### Product Catalog
Initially the repo was just about searching a product catalog with an LLM. That is why the catalog still has a central place in the project history.

The early goal was to understand embeddings and how prompt-based product search could work against a local dataset. Over time that grew into a broader agentic chat app, but the product catalog remained as one of the internal retrieval tools alongside web, news, files, and memory search.
