"""
Phase 4 Hybrid RAG Chatbot
PressGaney Internship - Shifa Sulthana O
 
Streamlit app that wraps the hybrid RAG pipeline.
Run with: streamlit run app.py

"""
 
# ============================================================
# SQLite Fix for Ubuntu 20.04 — MUST be first
# ============================================================
import sys
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
 
# ============================================================
# Imports
# ============================================================
import streamlit as st
import rag_pipeline
 
# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Phase 4 RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)
 
# ============================================================
# Header
# ============================================================
st.title("🤖 Phase 4 Hybrid RAG Chatbot")
st.info(
    "This chatbot answers questions only from the uploaded knowledge base. "
    "If the information is not available, it will say so instead of guessing."
)
st.caption(
    "Powered by Gemini Embeddings · BM25 · ChromaDB · Groq Llama 3.3 70B"
)
st.markdown(
    "Ask any question about: **AI, Machine Learning, Deep Learning, "
    "NLP, Computer Vision, Neural Networks, Reinforcement Learning, "
    "Large Language Models, and more.**"
)
st.divider()
 
# ============================================================
# Chat History (session state)
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Display existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(f"📄 View source chunks ({msg['num_docs']} retrieved)"):
                for i, source in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.caption(source[:400] + ("..." if len(source) > 400 else ""))
                    st.divider()
 
# ============================================================
# Chat Input
# ============================================================
if question := st.chat_input("Ask a question about the documents..."):
 
    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
 
    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
               result = rag_pipeline.ask_question(question)

            except Exception as e:
               st.error(f"An error occurred:\n\n{e}")
               st.stop()
 
        answer   = result["answer"]
        sources  = result["sources"]
        num_docs = result["num_docs"]
 
        st.markdown(answer)
 
        if sources:
            with st.expander(f"📄 View source chunks ({num_docs} retrieved)"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.caption(source[:400] + ("..." if len(source) > 400 else ""))
                    st.divider()
        else:
            st.info("No source chunks were retrieved for this question.")
 
    # Save assistant message to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
        "num_docs": num_docs
    })
 
# ============================================================
# Sidebar — System Info
# ============================================================
with st.sidebar:
    st.header("⚙️ System Info")
    st.markdown(f"**Embedding Model:** `gemini-embedding-001`")
    st.markdown(f"**LLM:** `llama-3.3-70b-versatile`")
    st.markdown(f"**Vector DB:** ChromaDB")
    st.markdown(f"**Keyword Search:** BM25")
    st.markdown(f"**Chunks in DB:** {rag_pipeline.vectordb._collection.count()}")
    st.markdown(f"**Retrieval:** 3 vector + 2 BM25")
    st.divider()
    st.markdown("**📚 Knowledge Base:**")
    st.markdown(
        "- Artificial Intelligence\n"
        "- Machine Learning\n"
        "- Deep Learning\n"
        "- Neural Networks\n"
        "- Natural Language Processing\n"
        "- Computer Vision\n"
        "- Large Language Models\n"
        "- Reinforcement Learning\n"
        "- Generative AI\n"
        "- Retrieval-Augmented Generation"
    )
    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
 