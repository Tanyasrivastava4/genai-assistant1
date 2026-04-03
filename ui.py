import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.set_page_config(page_title="GenAI Knowledge Assistant", layout="wide")

st.title("🤖 GenAI Knowledge Assistant")

# Sidebar for document upload
with st.sidebar:
    st.header(" Document Upload")
    uploaded_file = st.file_uploader("Upload PDF, TXT, or MD", type=["txt", "pdf", "md"])
    
    if uploaded_file is not None:
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{API_URL}/ingest", files=files)
        if response.status_code == 200:
            st.success(f" Ingested: {response.json()['chunks_stored']} chunks")
        else:
            st.error(f"Failed: {response.text}")
    
    st.divider()
    st.caption("Cache Status: Redis + Qdrant")
    st.caption("LLM: Groq Llama 3.1-8B")

# Main area for queries
st.header("💬 Ask a Question")

query = st.text_input("Enter your question:", placeholder="e.g., What is Artificial Intelligence?")
use_llm = st.checkbox("Use LLM (uncheck to return raw chunks)", value=True)

if st.button("Submit", type="primary") and query:
    with st.spinner("Processing..."):
        response = requests.post(
            f"{API_URL}/query",
            json={"query": query, "use_llm": use_llm}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Display answer
            st.subheader("📝 Answer")
            st.write(data["answer"])
            
            # Display metadata
            col1, col2, col3 = st.columns(3)
            col1.metric("Cache Hit", data["cache_hit"] or "Miss")
            col2.metric("Source", data["source"])
            col3.metric("Latency", f"{data['latency_ms']} ms")
            
            # Display retrieved chunks (expandable)
            with st.expander("🔍 Retrieved Chunks"):
                for i, chunk in enumerate(data["retrieved_chunks"]):
                    st.markdown(f"**Chunk {i+1}**")
                    st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                    st.divider()
        else:
            st.error(f"Error: {response.text}")