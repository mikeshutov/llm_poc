# LLM Powered agentic chat with a bunch of tooling
This project was created with the idea of exploring how to build things that utilize LLM's. Over time it has grown from just a simple chat bot that looks at a local product catalog to what it is today.


## Flows of the App
Currently the flow is when a prompt comes in we assemble the context and begin our agentic flow. if the classifier determines we already have enough context to answer we just go to the answer rigth away, this is for cases where the question may be applicable to existing context. Otherwise we enter our hybrid loop where we plan tool calls and execute said tool calls. The planner then determines if additional tool calls are required or if we have enough data. Additionally we have a limiter on the number of iterations we allow the agent to go through.

```mermaid
flowchart TD
    A[User Prompt] --> B[Context Assembly]
    B --> C[Classification]
    C --> D{Can Respond With Context?}
    D -->|Yes| F[Response Generation]
    D -->|No| E[Planner]
    E --> H[Executor]
    H --> G{Goal Reached or Limit Reached?}
    G -->|No| E
    G -->|Yes| F
    F --> I[Response]
```


## How is Context Assembled
Currently it is assemebled by combining the summary and all messages after the index of that summary. That means that if the last summary was at message 30 we will include messages after 30. Not perfect but provides a way to understand context assembly.
```mermaid
flowchart TD
    A[Context Builder] -->B 
    subgraph B[Conversation Context]
        direction TB
        M1["Conversation Summary with Summary Index"]
        M2["Array of Previous Roundtrips Past Summary Index"]
        M3["Latest Prompt"]
        M1 --> M2 --> M3 
    end

    subgraph C[Roundtrip Element]
        direction TB
        N1["User Prompt"]
        N2["Assistant Response"]
        N3["Tool Call Data"]
        N1-->N2-->N3
    end

 M2 --> C
```


# Setup Information
## Prereqs
- Docker + Docker Compose
- Python 3.11+ (uses local `.venv`)
- `DATABASE_URL`, `OPENAI_API_KEY`, `BRAVE_SEARCH_API_KEY` in `.env`

Example `.env`:
```
DATABASE_URL=postgresql://app:app@localhost:5432/products
OPENAI_API_KEY=...
BRAVE_SEARCH_API_KEY=...
```

## Quick Start
1. Start DB
```
docker compose up -d
```

2. Run DB setup (extensions + schemas)
```
python scripts/setup_db.py
```

3. (Optional) Seed products + embeddings
```
python db/seed_products.py
```

4. Start the app
```
streamlit run main.py
```

## Image Backfill (Optional)
If you already seeded the DB and want to backfill images:
```
setx ALLOW_IMAGE_BACKFILL 1
python db/seed_products.py
```

To force refresh existing images:
```
setx FORCE_IMAGE_REFRESH 1
python db/seed_products.py
```

Images are stored in `db/images/` (ignored by git).
