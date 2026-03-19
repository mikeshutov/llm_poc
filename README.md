# LLM Powered agentic chat with a bunch of tooling
This project was created with the idea of exploring how to build things that utilize LLM's. Over time it has grown from just a simple chat bot that looks at a local product catalog to what it is today. The product catalog used is just open data we have fed into the DB to have a pretend store (which is how this started) which is where we utilize embeddings for product searching.


## Flows of the App
Rough Breakdown of the agentic loop flow.
1. Prompt comes in and we assemble our context.
2. Pass the context to the state and set up an agent state to track execution.
3. Classifier checks if the context can satisfy the users request (using previous tool call data etc)
   a. If theres sufficient context we just go to answering right away.
   b. Categorize the request, this will determine which tools need to be provided.
4. We call our planner and provide tools matching the categories from classifier.
   a. The idea here is we could likely have thousands of tools and it seems like a good idea to only pass whats needed.
   b. There are tool specific rules which get added depending on the categories and tools (minimal but seems to be useful long term)
5. Executor executes the tool calls in the plan and stores them in state.
6. Planner checks if we have sufficient context to answer. 
   a. If goal reached or we reach our iteration limit create our response.
   b. If not we replan with tool calls made.
7. Response gets generated

We also do store the conversation/tool calls and so on for the next prompt.
The diagrams provided are just to illustrate at a high level what this looks like.

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

Simple diagram to illustrate what this looks like.

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
