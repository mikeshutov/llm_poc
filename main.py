import json
from dataclasses import asdict

from dotenv import load_dotenv
from uuid import UUID
from conversation.conversation import generate_conversation_title, generate_conversation_summary
from conversation.repository.repo_factory import get_conversation_repo
from intent_processing.generic_search import generic_web_search
from conversation.context_builder import build_roundtrip_context
from rendering.rendering import render_message, debug_render_message
from rendering.sidebar import render_sidebar
from rendering.messages import ensure_messages_loaded, render_messages, append_assistant_response


def setup_conversation(conversation_repository, cid):
    if cid:
        st.session_state.conversation_id = cid
    else:
        conv = conversation_repository.create_conversation(user_id="anonymous", metadata={"source": "streamlit"})
        st.session_state.conversation_id = str(conv.id)
        st.query_params["cid"] = st.session_state.conversation_id  # persist in URL


load_dotenv()
import streamlit as st

from intent_layer.intent_layer import parse_query
from intent_processing.product_retrieval import find_products
from response_layer.response_layer import generate_response
from intent_layer.models.intent import Intent
from response_layer.models.response_payload import ResponsePayload
from websearch.models.search_type import SearchType
from response_layer.cards_mapper import news_results_to_cards, product_results_to_cards
from personalization.tone import update_tone_state
from intent_layer.models.parsed_request import QueryDetails

#Page title
st.set_page_config(page_title="Product Finder", page_icon="🛒")
#get query parameters
qp = st.query_params
cid = qp.get("cid")


conversation_repository = get_conversation_repo()

setup_conversation(conversation_repository, cid)
current_conversation = conversation_repository.get_conversation(UUID(st.session_state.conversation_id))
if current_conversation and current_conversation.tone_state:
    st.session_state.tone_state = current_conversation.tone_state

# sidebar to do hold conversation list and title
with st.sidebar:
    render_sidebar(conversation_repository)

ensure_messages_loaded(
    conversation_repository,
    st.session_state.conversation_id,
    limit=10,
)



# output whole chat
render_messages(st.session_state.messages, render_message)

# prompt area
# needs to be heavily refactored but it was a good way to get started on figuring this thing out
userQuery = st.chat_input("What are you looking for? e.g. red red shirts under 50")
if userQuery:
    # store the promopt
    st.session_state.messages.append({"role": "user", "content": userQuery})
    with st.chat_message("user"):
        st.write(userQuery)

    with st.spinner("Thinking..."):
        #prepare to submit the query by getting all roundtrips
        roundtrips_with_latest = build_roundtrip_context(
            conversation_repository,
            st.session_state.conversation_id,
            userQuery,
            limit=5,
        )

        parsedQuery = parse_query(roundtrips_with_latest)
        st.session_state.tone_state = update_tone_state(
            st.session_state.get("tone_state"),
            parsedQuery.tone,
        )
        if st.session_state.get("tone_state"):
            tone_state_dict = st.session_state.tone_state.model_dump()
            conversation_repository.update_tone_state(
                UUID(st.session_state.conversation_id),
                tone_state_dict,
            )
            tone_label = tone_state_dict.get("label")
        else:
            tone_label = None
        queryDebugTitle = "**Debug: Parsed intent and tools**"
        st.session_state.messages.append({"role": "debug", "content": parsedQuery.model_dump(), "title": queryDebugTitle})
        debug_render_message(parsedQuery.model_dump(), queryDebugTitle)

        match parsedQuery.intent:
            case Intent.FIND_PRODUCTS:
                productResultTitle = "Product Results"
                product_query_text = parsedQuery.query_details.query_text if parsedQuery.query_details else ""
                productResponse = find_products(product_query_text, parsedQuery.common_properties)
                debug_render_message(
                    {
                        "internal_results": [asdict(p) for p in productResponse.internal_results],
                        "external_results": [asdict(p) for p in productResponse.external_results],
                    },
                    productResultTitle,
                )

                answer = generate_response(
                    conversation_entries=roundtrips_with_latest,
                    query_results=json.dumps(
                        {
                            "internal_results": productResponse.internal_results,
                            "external_results": productResponse.external_results,
                        },
                        default=str,
                    ),
                    tone_label=tone_label,
                )
                
                if not answer.cards:
                    answer = ResponsePayload(
                        response=answer.response,
                        cards=product_results_to_cards(productResponse, limit=10),
                        follow_up=answer.follow_up,
                    )

                print(answer)
                append_assistant_response(
                    conversation_repository,
                    st.session_state.conversation_id,
                    userQuery,
                    answer,
                    generate_conversation_title,
                    generate_conversation_summary,
                    parsed_query=parsedQuery.model_dump(),
                )

            # general information intent to demonstrate web data lookups
            case Intent.GENERAL_INFORMATION:
                search_results_title = "Search Results"
                if not parsedQuery.query_details:
                    clarification = ResponsePayload(
                        response="Can you clarify what you want to search for?",
                        cards=[],
                        follow_up="For example: the topic, timeframe, and source type (news or web).",
                    )
                    append_assistant_response(
                        conversation_repository,
                        st.session_state.conversation_id,
                        userQuery,
                        clarification,
                        generate_conversation_title,
                        generate_conversation_summary,
                        parsed_query=parsedQuery.model_dump(),
                    )
                else:
                    search_results = generic_web_search(parsedQuery.query_details)
                    answer = generate_response(
                        conversation_entries=roundtrips_with_latest,
                        query_results=json.dumps(search_results),
                        tone_label=tone_label,
                    )
                    search_type = parsedQuery.query_details.search_type
                    debug_render_message(parsedQuery.query_details.model_dump(), search_results_title)
                    if search_type == SearchType.NEWS_SEARCH and not answer.cards:
                        news_cards = news_results_to_cards(search_results, limit=5)
                        if news_cards:
                            answer = ResponsePayload(
                                response=answer.response,
                                cards=news_cards,
                                follow_up=answer.follow_up,
                            )
                    # we need to parse out some of the data for the searches we perform but its a start
                    append_assistant_response(
                        conversation_repository,
                        st.session_state.conversation_id,
                        userQuery,
                        answer,
                        generate_conversation_title,
                        generate_conversation_summary,
                        parsed_query=parsedQuery.model_dump(),
                    )

            # unknown intent just in case we cant figure out what the user wants
            case Intent.UNKNOWN:
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, I don't understand you."})
                st.error("Sorry, I don't understand you.")
