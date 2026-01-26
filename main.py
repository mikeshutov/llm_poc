import json
from dotenv import load_dotenv
from uuid import UUID

from conversation.conversation import generate_conversation_title
from db import get_conversation_repo
from intent_processing.generic_search import generic_web_search
from llm.llm_message_builder import compose_messages_from_roundtrips
from websearch.web_search import process_search_intent

load_dotenv()
import streamlit as st

from intent_layer.intent_layer import parse_query
from intent_processing.product_retrieval import find_products
from response_layer.response_layer import generate_response
from models.intent import Intent

#Page title
st.set_page_config(page_title="Product Finder", page_icon="🛒")
#get query parameters
qp = st.query_params
cid = qp.get("cid")


conversation_repository = get_conversation_repo()

#util to move
def render_message(msg):
    role = msg["role"]
    content = msg["content"]
    contentTitle = msg.get("title", "Debug")
    if role == "debug":
        debug_render_message(content,contentTitle)
    else:
        with st.chat_message(role):
            if role == "assistant":
                st.info(content)
            else:
                st.write(content)


def debug_render_message(content, contentTitle):
    with st.chat_message("assistant", avatar="🧪"):
        st.markdown(contentTitle)
        with st.expander("Debug"):
            st.json(content)

if cid:
    st.session_state.conversation_id = cid
else:
    conv = conversation_repository.create_conversation(user_id="anonymous", metadata={"source": "streamlit"})
    st.session_state.conversation_id = str(conv.id)
    st.query_params["cid"] = st.session_state.conversation_id  # persist in URL

# sidebar to do hold conversation list and title
with st.sidebar:
    st.title("LLM Powered Store/Searcher")
    st.caption("Conversation")
    st.code(st.session_state.conversation_id)

    if st.button("🗑️ Delete this conversation", type="secondary"):
        conversation_repository.delete_conversation(UUID(st.session_state.conversation_id),user_id="anonymous")
        latest = conversation_repository.get_latest_conversation("anonymous")

        if latest:
            st.session_state.conversation_id = str(latest.id)
        else:
            conv = conversation_repository.create_conversation(
                user_id="anonymous",
                metadata={"source": "streamlit"},
            )
            st.session_state.conversation_id = str(conv.id)

        st.query_params["cid"] = st.session_state.conversation_id
        st.session_state.loaded_cid = None
        st.session_state.messages = []
        st.rerun()

    if st.button("➕ New chat"):
        conv = conversation_repository.create_conversation(user_id="anonymous", metadata={"source": "streamlit"})
        st.session_state.conversation_id = str(conv.id)
        st.session_state.messages = []
        st.session_state.debug_turns = []
        st.rerun()

    st.divider()
    st.caption("Conversations")
    conversations = conversation_repository.list_conversations(user_id="anonymous", limit=50)

    # Build labels + stable mapping
    items = []
    id_by_label = {}
    for c in conversations:
        title = (c.title or "Untitled").strip()
        label = f"{title}  ·  {str(c.id)[:8]}"
        items.append(label)
        id_by_label[label] = str(c.id)

    # Pick current index
    current_id = st.session_state.conversation_id
    current_index = 0
    for i, label in enumerate(items):
        if id_by_label[label] == current_id:
            current_index = i
            break

    selected = st.radio(
        label="Conversations",
        options=items,
        index=current_index if items else 0,
        label_visibility="collapsed",
    )

    if items:
        new_id = id_by_label[selected]
        if new_id != current_id:
            st.session_state.conversation_id = new_id
            st.query_params["cid"] = new_id
            st.session_state.loaded_cid = None
            st.rerun()

if "messages" not in st.session_state or st.session_state.get("loaded_cid") != st.session_state.conversation_id:
    roundtrips = conversation_repository.list_roundtrips(
        UUID(st.session_state.conversation_id),
        limit=10,
    )
    st.session_state.messages = []
    for rt in roundtrips:
        st.session_state.messages.append({"role": "user", "content": rt.user_prompt})
        st.session_state.messages.append({"role": "assistant", "content": rt.generated_response})
    st.session_state.loaded_cid = st.session_state.conversation_id



# output whole chat
for msg in st.session_state.messages:
    render_message(msg)

# prompt area
# needs to be heavily refactored but it was a good way to get started on figuring this thing out
userQuery = st.chat_input("What are you looking for? e.g. red red shirts under 50")
if userQuery:
    # store the promopt
    st.session_state.messages.append({"role": "user", "content": userQuery})
    with st.chat_message("user"):
        st.write(userQuery)

    #prepare to submit the query by getting all roundtrips
    conversation_roundtrips = conversation_repository.list_roundtrips(UUID(st.session_state.conversation_id), limit=10)
    roundtrips_with_latest = [*compose_messages_from_roundtrips(conversation_roundtrips),
                              {"role": "user", "content": userQuery}]

    parsedQuery = parse_query(roundtrips_with_latest)
    queryDebugTitle = "**Debug: Parsed intent and tools**"
    st.session_state.messages.append({"role": "debug", "content": parsedQuery.model_dump(), "title": queryDebugTitle})
    with st.chat_message("assistant", avatar="🧪"):
        st.markdown(queryDebugTitle)
        with st.expander("Debug"):
            st.json(parsedQuery.model_dump())

    match parsedQuery.intent:
        case Intent.FIND_PRODUCTS:
            productResultTitle = "Product Results"
            productResponse = find_products(parsedQuery.product_query)
            st.session_state.messages.append(
                {"role": "debug", "content": productResponse.to_dict(orient="records"), "title": productResultTitle})
            debug_render_message(productResponse.to_dict(orient="records"), productResultTitle)

            # pass records to response layer (exclude embedding and ID; were keeping distance in case we want to use it)
            records = productResponse.drop(columns=["embedding", "id"], errors="ignore").to_dict(orient="records")
            answer = generate_response(
                parsedRequest=roundtrips_with_latest,
                query_results=json.dumps(records),
            )
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.info(answer)
            appendResult = conversation_repository.append_roundtrip(st.session_state.conversation_id,userQuery,answer)
            if appendResult.message_index == 0:
                generated_title = generate_conversation_title(userQuery)
                conversation_repository.set_conversation_title(st.session_state.conversation_id, generated_title)
                st.rerun()

            summary_row = conversation_repository.get_summary(UUID(cid))
        # general information intent to demonstrate web data lookups
        case Intent.GENERAL_INFORMATION:
            search_results_title = "Search Results"
            search_intent_results = process_search_intent(userQuery)
            st.session_state.messages.append(
                {"role": "debug", "content": search_intent_results, "title": search_results_title})
            search_results = generic_web_search(q=search_intent_results.search_query, search_type=search_intent_results.search_type)
            answer = generate_response(
                parsedRequest=roundtrips_with_latest,
                query_results=json.dumps(search_results),
            )
            # we need to parse out some of the data for the searches we perform but its a start
            st.session_state.messages.append({"role": "assistant", "content": answer})
            appendResult = conversation_repository.append_roundtrip(st.session_state.conversation_id, userQuery, answer)
            if appendResult.message_index == 0:
                generated_title = generate_conversation_title(userQuery)
                conversation_repository.set_conversation_title(st.session_state.conversation_id, generated_title)
                st.rerun()
            with st.chat_message("assistant"):
                st.info(answer)

        # unknown intent just in case we cant figure out what the user wants
        case Intent.UNKNOWN:
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I don't understand you."})
            st.error("Sorry, I don't understand you.")
