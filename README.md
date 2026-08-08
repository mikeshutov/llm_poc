# LLM Powered agentic chat with a bunch of tooling
This project was created with the idea of exploring how to build things that utilize LLM's. Over time it has grown from just a simple chat bot that looks at a local product catalog to what it is today. The product catalog used is just open data we have fed into the DB to have a pretend store (which is how this started) which is where we utilize embeddings for product searching.
This is not meant to be a production piece of code. More just a way to explore the topic.


## Flows of the App
Rough breakdown of the current agent loop flow.
1. Prompt comes in and we assemble conversation context.
2. We pass that context plus the latest user prompt into agent state.
3. We prepare a small `User Profile` section that starts with geo/location-aware metadata plus an initially empty user-attributes section.
4. `request_analysis` infers the user's goal, decides whether tools are required, selects the relevant tool categories, and requests any specific user attribute types that would be helpful for the request.
   - If the existing context is strong enough, we can skip planning and go straight to synthesis.
   - If tools are needed, the selected categories determine which tool groups and rules are loaded.
   - If durable profile context would help, request analysis asks for specific attribute types such as `food.likes` or `projects.goals` rather than receiving the full stored attribute set up front.
5. We load only the requested user attribute types into the profile and condense overlapping records for prompt efficiency.
   - This keeps the profile lightweight during request analysis and only hydrates the slice that is actually useful.
6. After profile loading, the main flow fans out into separate agent paths.
   - The main agent path handles planning, execution, replanning, and synthesis.
   - The profile-management agent path can work on durable attribute maintenance separately.
7. The executor executes the tool calls in the plan and stores the results in state.
8. The planner replans if needed.
   - If the goal is reached or we hit the iteration limit, we move to synthesis.
   - Otherwise we loop with the newly gathered evidence.
9. A collect/barrier step brings the active paths back together before the validator decides whether we should execute again or synthesize.
10. Synthesis generates the final response, roundtrip summary, and tool summary. It receives explicit plan evidence plus a small recent-context window rather than the planner context or tool-summary history payloads.

We also store conversations, roundtrips, prompt rows, summaries, and tool calls for future prompts.
The diagrams provided are just to illustrate the high-level shape of the flow.

```mermaid
flowchart TD
    A[User Prompt] --> B[Context Assembly]
    B --> C[Request Analysis]
    C --> D[Load Requested Profile Attributes]
    D --> E[Fanout to Agents]
    E --> P[Profile Management Agent]
    E --> G[Initial Planner]
    P --> K[Collect]
    G --> K
    K --> F{Execute More Tools?}
    F -->|No| J[Synthesis]
    F -->|Yes| H[Executor]
    H --> I[Replan]
    I --> K
    J --> L[Response]
```


## How is Context Assembled
The more accurate mental model now is not just “we build one context blob and pass it everywhere.” What we actually do is build a fairly rich shared `AgentState`, and then each edge or node chooses the specific parts of that state that it needs when constructing its own prompt.

### AgentState First
At the start of a turn we assemble a reusable `AgentState` that can hold:
1. The latest user prompt as the active task.
2. Conversation context, including high-level summaries and recent roundtrips.
3. A lightweight user profile with geo/location-aware metadata.
4. Request-analysis output such as the refined goal, requested attribute types, and applicable tool categories.
5. Iteration state, including plans and tool results gathered so far.
6. Subagent state for any secondary agent paths such as profile management.
7. Final result fields, logs, and any other runtime data needed across the graph.

This means the system can keep one shared runtime picture of the turn without forcing every prompt to receive every field.

### Context Per Edge
After that shared state exists, each edge builds its own prompt context from the smallest useful slice of the state.

For example:
1. `request_analysis` gets the latest user prompt, lightweight profile metadata, and selected conversation context so it can infer goal, tool categories, and useful stored attribute types.
2. `load_user_profile` does not create an LLM prompt, but it uses the request-analysis output in state to hydrate only the requested attribute types.
3. The planner gets the refined goal, the narrowed tool set, the requested slice of the user profile, and previous iteration evidence rather than the full conversation context.
4. The executor does not build a planning prompt. It uses the current plan plus accumulated results in state to execute tool calls.
5. The profile-management agent can use the same shared state model, but with its own narrower prompt and rules focused on durable attribute maintenance.
6. Synthesis gets the latest result state, explicit plan evidence, the hydrated profile slice, and only a small recent-context window rather than the full planner context.

So in practice the flow is:
1. Build `AgentState`.
2. Move through the graph.
3. At each edge or node, choose the relevant state elements.
4. Build the local prompt or execution context for that step.

### Conversation Context Layer
Roundtrips have this storage flow:
1. We create a pending roundtrip with the user's prompt and the model being used.
2. We execute the agent logic.
3. We update the pending roundtrip with the response data, roundtrip summary, tool summary, and any related metadata.

For subsequent prompts, the conversation-context portion of state is built from a few layers of history:
1. `conversation.summary`, which is the continually refreshed top-level summary of the overall conversation.
2. The latest rolling `conversation_summary` row, which summarizes an older batch window and carries its tool summary.
3. Recent unsummarized roundtrips after the latest batch cutoff, where each entry includes the original `user_prompt` plus the `roundtrip_summary`.
4. Recent unsummarized structured tool summaries after the latest batch cutoff.

The latest user prompt is not embedded inside the stored conversation context anymore. It is passed separately as the live task so the current request stays distinct from historical context.

### Profile Hydration Layer
We also prepare a separate `User Profile` section inside state. That profile always starts with geo/location-aware metadata, but stored user attributes are loaded in a staged way:
1. `request_analysis` sees the lightweight profile and decides whether stored user attributes would help.
2. It requests specific attribute types such as `food.likes`, `technology.skills`, or `projects.goals`.
3. We load only those active attribute types, condense overlapping records by attribute type and `group_key`, and then place that hydrated profile slice back into state.
4. After that hydration step, the later agent paths work from the smaller relevant profile slice rather than the full stored attribute set.

That means `request_analysis` does not receive the full stored attribute set up front, the planner does not receive full conversation context, and synthesis receives only a narrow recent-context window plus explicit plan evidence.

Overall, the idea is that each step in the flow has access to a large `AgentState` object containing the contextual information it may need. When a step needs to make a planning, analysis, or synthesis request, we then choose the most appropriate subset of that state for the prompt. This keeps requests smaller while still giving the LLM enough context to do the job well.

Simple diagram to illustrate what this looks like.

```mermaid
flowchart TD
    A[Context Builder] --> B[Build Shared AgentState]

    subgraph C[AgentState]
        direction TB
        S1[Latest User Prompt / Task]
        S2[Conversation Context]
        S3[Lightweight User Profile]
        S4[Request Analysis Output]
        S5[Iteration Trace / Tool Results]
        S6[Subagent States]
    end

    B --> C
    C --> D[Request Analysis selects needed state fields]
    C --> E[Profile Loader hydrates requested attributes]
    C --> F[Planner selects refined goal, tools, profile slice, prior evidence]
    C --> G[Executor uses current plan plus results]
    C --> H[Synthesis selects recent context plus final evidence]
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
### Memories
As more topics are explored one interesting topic is the idea of allowing the agent to recall previous conversations or topics discussed. The way it is done here is fairly straight forward and achieved via embeddings similar to how we handle rag through files. The data was restructured slightly to achive this but now what we have is the following:

1. We have conversation summaries as well as an embedding for this summary generated on every single roundtrip. This means that we have what could be described as a persistant current state for the conversation.
2. We have conversation roundtrip summaries with embeddings as well. This summary simply summarizes this particular back and forth. So a particular prompt being responded to.

This comes together with the addition of two tools: search_memories, search_roundtrip_memories
1. search_memories - Is used to search through past conversations using the conversation summary and returns up to 3 relevant conversation memories by default.
2. search_roundtrip_memories - Is used to search through detailed roundtrips once we have found a relevant conversation and returns up to 3 matching roundtrips by default.

The idea here is to essentially allow the agent to on demand find data that is present in other conversations and to do that we use a similar approach to files where summaries are converted to embeddings and the data can be found using an embedding search. We order the data by relevance of course and provide a relevance score as well to help identify the correct data.

### Why Request Analysis
With the number of tools growing I wanted to solve for the scaling problem of passing a large tool list to the planner. The idea here is:
1. Request analysis determines the user's goal, the category or categories that are applicable to the request, and any specific user attribute types worth loading. This lets the planner focus on a refined goal instead of carrying full conversation context.
2. We then load only the requested profile attributes before the later agent paths run, so both the main agent and the profile-management path can work from a smaller relevant profile slice.
3. The planner prompt then injects only the tools and rules which fall under those categories plus any rules that are always present.
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

2. Run DB setup (extensions + schemas + migrations)
```
python scripts/setup_db.py
```

If you already have a local database from an older clone and just need to move it forward, run:
```
python scripts/migrate_db.py
```

3. (Optional) Seed products + embeddings
```
python scripts/seed_products.py
```

4. Start the app
```
streamlit run main.py
```

## Image Backfill (Optional)
If you already seeded the DB and want to backfill images:
```
setx ALLOW_IMAGE_BACKFILL 1
python scripts/seed_products.py
```

To force refresh existing images:
```
setx FORCE_IMAGE_REFRESH 1
python scripts/seed_products.py
```

Product images are stored in `db/images/` for now. 
Uploaded files (PDFs, DOCX, images, etc.) are stored in `static/files/`.
