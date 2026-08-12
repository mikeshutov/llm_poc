# LLM Powered agentic chat with a bunch of tooling
This project was created with the idea of exploring how to build things that utilize LLMs. Over time it has grown from just a simple chatbot that looks at a local product catalog to what it is today. The product catalog used is open data loaded into the database to act as a pretend store, which is where embeddings started to come into the picture for product searching.

This is not meant to be a production piece of code. It is mostly a place to explore the topic and keep iterating on ideas.

## Table of Contents
- [Project Summary](#project-summary)
- [Main App Flow](#main-app-flow)
- [Notes](#notes)
- [Documentation](#documentation)
- [Setup](#setup)

## Project Summary
At a high level this repo explores:
- agent orchestration with request analysis, planning, execution, and synthesis
- staged user-profile hydration and durable attribute management
- retrieval and reranking across products, web/news, files, and memories
- a Streamlit-based UX for experimenting with the full loop

## Main App Flow
Rough breakdown of the current agent loop flow.
1. Prompt comes in and we assemble conversation context.
2. We pass that context plus the latest user prompt into agent state.
3. We prepare a small `User Profile` section that starts with geo/location-aware metadata plus an initially empty user-attributes section.
4. `request_analysis` infers the user's goal, selects relevant tool categories, and requests any specific user attribute types that would be helpful for the request.
5. We load only the requested user attribute types into the profile and condense overlapping records for prompt efficiency.
6. After profile loading, the main flow fans out into separate agent paths.
7. The main agent path handles planning, execution, replanning, and synthesis, while the profile-management agent path can work on durable attribute maintenance separately.
8. The executor executes the tool calls in the plan in parallel and stores the results in state.
9. After the initial fanout, a collect/barrier step brings the active paths back together before the validator decides whether we should execute or synthesize.
10. After execution, the loop can either replan immediately when the planner marked `needs_replan`, run the evaluator to decide whether another useful step remains, or synthesize if the goal is reached or limits are hit.
11. Synthesis generates the final response, roundtrip summary, and tool summary. It works from explicit evidence plus narrowed conversation context rather than planner history payloads.

We also store conversations, roundtrips, prompt rows, summaries, and tool calls for future prompts.

```mermaid
flowchart TD
    A[User Prompt] --> B[Build Conversation Context]
    B --> C[Request Analysis]
    C --> D[Load Requested Profile Attributes]
    D --> E[Fanout]

    E --> PM[Profile Management Agent]
    E --> P1[Initial Planner]
    PM --> K[Collect]
    P1 --> K

    K --> V{Validator}
    V -->|No plan or goal reached| S[Synthesis]
    V -->|Plan has steps| X[Execute Tools in Parallel]

    X --> R{Post-Execution Router}
    R -->|goal reached or max turns| S
    R -->|planner set needs_replan| P2[Planner]
    R -->|results ready for evaluation| EV[Evaluator]

    EV --> ER{Evaluator Router}
    ER -->|satisfied or terminal| S
    ER -->|retryable| P2

    P2 --> K
    S --> L[Response]
```

## Notes
### Product Catalog
Initially the repo was just about searching a product catalog with an LLM. That is why the catalog still has a central place in the project history.

The early goal was to understand embeddings and how prompt-based product search could work against a local dataset. Over time that grew into a broader agentic chat app, but the product catalog remained as one of the internal retrieval tools alongside web, news, files, and memory search.

## Documentation
The more detailed documentation now lives in the [`docs/`](docs/) folder.

- [Context Assembly](docs/context-assembly.md)
- [File Searching](docs/file-searching.md)
- [Memories](docs/memories.md)
- [Request Analysis](docs/request-analysis.md)
- [Reranking](docs/reranking.md)

## Setup
### Prereqs
- Docker + Docker Compose
- Python 3.11+ (uses local `.venv`)
- `DATABASE_URL`, `OPENAI_API_KEY`, `BRAVE_SEARCH_API_KEY` in `.env`

Example `.env`:
```text
DATABASE_URL=postgresql://app:app@localhost:5432/products
OPENAI_API_KEY=...
BRAVE_SEARCH_API_KEY=...
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
