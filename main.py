import json

import streamlit as st

from intent_processing.db_product_search import search_products
from intent_processing.product_embeddings import embed_text
from intent_layer.intent_layer import parse_query
from intent_processing.product_retrieval import find_products
from response_layer.response_layer import generate_response
from models.intent import Intent


st.set_page_config(page_title="Product Finder", page_icon="🛒")

st.title("LLM Powered Product Search")

user_query = st.text_input("What are you looking for?", placeholder="e.g. red gym shirts under 50 bucks")


if user_query:
    st.subheader("1️⃣ Parsed intent / filters")
    parsedQuery = parse_query(user_query)
    st.json(parsedQuery)

    match parsedQuery.intent:
        case Intent.FIND_PRODUCTS:
            st.subheader("2️⃣ Retrieved products")
            df = find_products(parsedQuery.product_query)
            st.dataframe(df)

            # pass records to response layer (exclude embedding and ID; were keeping distance in case we want to use it)
            records = df.drop(columns=["embedding", "id"], errors="ignore").to_dict(orient="records")
            answer = generate_response(
                user_query=user_query,
                parsedRequest=parsedQuery,
                query_results=json.dumps(records),
            )
            st.subheader("3️⃣ LLM recommendation")
            st.write(answer)

        case Intent.UNKNOWN:
            st.write("Sorry, I don't understand you.")


