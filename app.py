"""
RAG-Based AI Search System — starter interface.

Run with:
    streamlit run app.py

This gives you a working, end-to-end demo today: document loading, TF-IDF based
retrieval, and an extractive answer — all wired into a real web interface. Build
your final project by upgrading each piece (see the TODOs in rag/embed_store.py
and rag/generate.py) without needing to touch this file's overall structure.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st

import time  
from google.genai import errors

from rag.ingest import load_documents, build_chunk_records
from rag.embed_store import VectorStore
from rag.generate import generate_answer

DATA_FOLDER = "data/sample_docs"

st.markdown(
    """
    <style>

    [data-testid="stAppViewContainer"] {
        direction: rtl;
    }
    [data-testid="stAppViewContainer"] > * {
        direction: ltr;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.set_page_config(page_title="Cake Master <3", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading and indexing documents...")
def load_store():
    docs = load_documents(DATA_FOLDER)
    chunks = build_chunk_records(docs)
    store = VectorStore()
    store.build(chunks)
    return store, docs, chunks


store, docs, chunks = load_store()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)
    mode = st.radio("Answer mode", ["extractive", "llm"], index=0,
                     help="Extractive works with no setup. LLM mode needs ANTHROPIC_API_KEY set.")
    st.divider()
    st.caption(f"Indexed **{len(docs)}** documents \u2192 **{len(chunks)}** chunks")
    with st.expander("Documents in this index"):
        for d in docs:
            st.write(f"- {d['title']}")

st.title("🔎 The Cake Master ")
st.caption("Ask your cake related questions!")

query = st.text_input("Your question", placeholder="e.g. Tips for baking cakes?")
search_clicked = st.button("Search", type="primary")

if search_clicked and query.strip():
    retrieved = store.query(query, top_k=top_k)
    
    # --- ADDED AUTO-RETRY LOGIC FOR 429 ERRORS ---
    answer = None
    max_retries = 3
    retry_delay = 5  # Seconds to wait before trying again
    
    for attempt in range(max_retries):
        try:
            with st.spinner("Generating answer..."):
                answer = generate_answer(query, retrieved, mode=mode)
            break  # Success! Break out of the retry loop.
        except errors.ClientError as e:
            if e.code == 429:
                if attempt < max_retries - 1:
                    st.warning(f"Rate limit hit (429). Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Double the wait time for the next attempt
                else:
                    st.error("Google Gemini API Free Tier rate limit exceeded. Please try again in a minute or switch to 'extractive' mode.")
            elif e.code in (400, 401, 403):
                st.error("Your Gemini API key was rejected (it may be missing, "
                    "expired, or mistyped). Double-check GEMINI_API_KEY in your "
                    ".env file, or switch to 'extractive' mode in the sidebar."
                )
                break
            else:
                st.error(f"An API error occurred: {e.message}")
                break
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            break

    # Only render results if an answer was successfully retrieved
    if answer:
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for chunk, score in retrieved:
            with st.expander(f"{chunk.doc_title}  \u00b7  similarity {score:.2f}"):
                st.write(f"\"{chunk.text}\"")
                st.write(f"Citation: {chunk.citation}")
                st.write(f"URL: {chunk.source_url}")
                
elif search_clicked:
    st.warning("Type a question first.")


