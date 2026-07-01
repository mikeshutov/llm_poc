# LLM Powered agentic chat with a bunch of tooling
This project was created with the idea of exploring how to build things that utilize LLM's. Over time it has grown from just a simple chat bot that looks at a local product catalog to what it is today. The product catalog used is just open data we have fed into the DB to have a pretend store (which is how this started) which is where we utilize embeddings for product searching.
This is not meant to be a production piece of code. More just a way to explore the topic.


## Flows of the App
Rough breakdown of the current agent loop flow.
1. Prompt comes in and we assemble conversation context.
2. We pass that context plus the latest user prompt into agent state.
3. `request_analysis` infers the user's goal, decides whether tools are required, and selects the relevant tool categories.
   - If the existing context is strong enough, we can skip planning and go straight to synthesis.
   - If tools are needed, the selected categories determine which tool groups and rules are loaded.
4. We call the planner and give it the goal, prior tool-use context, available tools, and previous planner iterations.
   - The idea here is we could likely have thousands of tools and it seems like a good idea to only pass what is needed.
   - There are also tool-specific rules which get added depending on the active categories.
5. The executor executes the tool calls in the plan and stores the results in state.
6. The planner replans if needed.
   - If the goal is reached or we hit the iteration limit, we move to synthesis.
   - Otherwise we loop with the newly gathered evidence.
7. Synthesis generates the final response, roundtrip summary, and tool summary.

We also store conversations, roundtrips, prompt rows, summaries, and tool calls for future prompts.
The diagrams provided are just to illustrate the high-level shape of the flow.

```mermaid
flowchart TD
    A[User Prompt] --> B[Context Assembly]
    B --> C[Request Analysis]
    C --> D{Needs Tools?}
    D -->|No| F[Synthesis]
    D -->|Yes| E[Planner]
    E --> H[Executor]
    H --> G{Goal Reached or Limit Reached?}
    G -->|No| E
    G -->|Yes| F
    F --> I[Response]
```


## How is Context Assembled
Roundtrips have this flow:
1. We create a pending roundtrip with the user's prompt and the model being used.
2. We execute the agent logic.
3. We update the pending roundtrip with the response data, roundtrip summary, tool summary, and any related metadata.

For subsequent prompts, the context builder pulls together a few layers of history:
1. `conversation.summary`, which is the continually refreshed top-level summary of the overall conversation.
2. The latest rolling `conversation_summary` row, which summarizes an older batch window and carries its tool summary.
3. Recent unsummarized roundtrip summaries and tool summaries after the latest batch cutoff.

The latest user prompt is not embedded inside the stored conversation context anymore. It is passed separately into the prompt as the final section so the live request stays distinct from historical context.

The idea is to give the request analysis and synthesis steps reusable historical context while still keeping the freshest user request explicit and easy to reason about.

Simple diagram to illustrate what this looks like.

```mermaid
flowchart TD
    A[Context Builder] --> B
    subgraph B[Conversation Context]
        direction TB
        M1["conversation.summary"]
        M2["Latest rolling conversation_summary summary + tool summary + cutoff"]
        M3["Recent unsummarized roundtrip summaries"]
        M4["Recent unsummarized tool summaries"]
        M1 --> M2 --> M3 --> M4
    end

    A --> D[Latest User Prompt passed separately]

    subgraph C[Roundtrip Element]
        direction TB
        N1["User Prompt"]
        N2["Assistant Response"]
        N3["Roundtrip Summary"]
        N4["Tool Summary"]
        N1 --> N2 --> N3 --> N4
    end

    M3 --> C
    M4 --> C
```

## How File Searching with Uploads and Large files Works here
The implementation is fairly simple its intentionally not async to keep things simple but one could imagine at scale you would want to make part of the processing async. idea for what happens with file uploads/searches is explained in the diagrams but essentially:
1. Files are uploaded and chunked into 500 token sized chunks and embeddings are created.
2. When the file tools are utilized we convert the query into an embedding and perform an embedding search to find chunks which are semantically close/likely the data we look for. Note: For images we generate a description of the image with an LLM and then generate an embedding for that descrption. This way we can allow for easy contextual searches of images as well.
```mermaid
flowchart TD
    A[File Uploaded] --> B[File Type Check]
    B -->|Image| C[Generate Image Description via LLM]
    B -->|Text / PDF / DOCX| E[Extract Text]
    C --> F[Single Chunk from Description]
    E --> F2[Split into 500-Token Chunks]
    F --> G[Create Embedding per Chunk]
    F2 --> G
    G --> H[Save File Information + Chunks to DB]
```

## Interesting Notes/Decisions
### Why Request Analysis
With the number of tools growing I wanted to solve for the scaling problem of passing a large tool list to the planner. The idea here is:
1. Request analysis determines the user's goal and the category or categories that are applicable to the request.
2. The planner prompt then injects only the tools and rules which fall under those categories plus any rules that are always present.
This results in sending only the tools that are relevant, at least that is the idea. If we had thousands of tools we could reduce them to a much smaller number, although I am certain that if the number of tools grows this problem will need another refactor.

### What's with the Product Catalog
Initially the goal was just to build a way to search through a catalog by using an LLM. So the first thing that I added was a catalog. As part of that I added embeddings and the ability to search through the catalog utilizing user input that was just a prompt. The goal was to understand embeddings and how those would work. This has since become just another tool on the agent/chat which checks for products in the internal catalog or looks for products on the search index (Brave used here).


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

Product images are stored in `db/images/` for now. 
Uploaded files (PDFs, DOCX, images, etc.) are stored in `static/files/`.
