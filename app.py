"""
RAG Travel Assistant Streamlit Frontend

Provides a conversational Streamlit interface for interacting with RAG Travel Assistant.

Author: Sebastian Escobar-Mesa
Date: 5-9-2026
"""

import streamlit as st
from travel_rag import run_answer

st.set_page_config(
    page_title="RAG Travel Assistant",
    layout="centered"
)

st.markdown("""
<style>
.user-message {
    background-color: #DCF8C6;
    color: black;
    padding: 12px 16px;
    border-radius: 16px;
    max-width: 75%;
    margin-left: auto;
    margin-bottom: 12px;
}

.assistant-message {
    background-color: #F1F0F0;
    color: black;
    padding: 12px 16px;
    border-radius: 16px;
    max-width: 75%;
    margin-right: auto;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("RAG Travel Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f'<div class="user-message">{message["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="assistant-message">{message["content"]}</div>',
            unsafe_allow_html=True
        )

query = st.chat_input("Ask a travel question...")

if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.spinner("Generating answer..."):
        answer = run_answer(
            query=query,
            top_k=8,
            db_path="./kb",
            collection_name="travel",
            embed_model="text-embedding-3-small",
            chat_model="gpt-4o-mini",
            alpha=0.5
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()