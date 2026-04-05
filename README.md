# LLM Powered agentic chat with a bunch of tooling
This project was created with the idea of exploring how to build things that utilize LLM's. Over time it has grown from just a simple chat bot that looks at a local product catalog to what it is today. The product catalog used is just open data we have fed into the DB to have a pretend store (which is how this started) which is where we utilize embeddings for product searching.
This is not meant to be a production piece of code. More just a way to explore the topic.


## Flows of the App
Rough Breakdown of the agentic loop flow.
1. Prompt comes in and we assemble our context.
2. Pass the context to the state and set up an agent state to track execution.
3. Classifier checks if the context can satisfy the users request (using previous tool call data etc)
   - If theres sufficient context we just go to answering right away.
   - Categorize the request, this will determine which tools need to be provided.
4. We call our planner and provide tools matching the categories from classifier.
   - The idea here is we could likely have thousands of tools and it seems like a good idea to only pass whats needed.
   - There are tool specific rules which get added depending on the categories and tools (minimal but seems to be useful long term)
5. Executor executes the tool calls in the plan and stores them in state.
6. Planner checks if we have sufficient context to answer. 
   - If goal reached or we reach our iteration limit create our response.
   - If not we replan with tool calls made.
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
Roundtrips all have this flow:
1. We create a pending roundtrip with the users prompt.
2. We execute the agentic logic.
3. We update the pending roundtrip with our response data.

During subsequent prompts these roundtrips are used to assemble the context by combining the summary and all messages after the index of that summary. That means that if the last summary was at message 30 we will include messages after 30. Not perfect but provides a way to understand context assembly.
An important note is when we assemble the context we make sure to also include tool call data as part of the context (longterm would be a tool call summary of sorts). 

The idea is to provide that additional context so that the LLM could potentially reuse the data during classification and skip lookups entirely. 

Simple diagram to illustrate what this looks like.

```mermaid
flowchart TD
    A[Context Builder] -->B 
    subgraph B[Conversation Context]
        direction TB
        M1["Conversation Summary + Tool Call Summary\n with Summary Index"]
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
    G --> H[Save File + Chunks to DB]
```

## Interesting Notes/Decisions
### Why Classifier
With the number of tools growing I wanted to solve for the scaling problem of passing a bunch of tools to the planner which was going to happen. The idea here is:
1. The classifier determines a category or categories which are applicable to the request.
2. The planner prompt then handles injection of only tools and rules which fall under this category + any rules that are always present.
This results in sending only the tools that are relevant at least thats the idea. If we had thousands of tools we could reduce them to a much smaller number, although I am certain that if the number of tools grows this problem will need another refactor.

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
