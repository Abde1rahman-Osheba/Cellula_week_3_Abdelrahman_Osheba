import streamlit as st
import os

from rag_pipeline import (
    load_document, 
    chunk_text, 
    generate_embeddings, 
    build_vector_store, 
    retrieve, 
    generate_response
)

# Streamlit Page Config
st.set_page_config(
    page_title="Personalized RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

# Cache the heavy initialization steps so they don't rerun on every interaction
@st.cache_resource
def initialize_pipeline():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(script_dir, "knowledge_base.txt")
    
    with st.spinner("Loading knowledge base..."):
        text = load_document(kb_path)
        chunks = chunk_text(text)
    
    with st.spinner("Generating embeddings (this may take a moment)..."):
        model, embeddings = generate_embeddings(chunks)
    
    with st.spinner("Building vector database..."):
        index = build_vector_store(embeddings)
        
    return model, index, chunks

# Main App UI
st.title("🤖 Personalized RAG Assistant")
st.markdown("Ask anything about Abdelrahman Osheba based on the local knowledge base.")

try:
    model, index, chunks = initialize_pipeline()
    st.success("Pipeline initialized successfully!", icon="✅")
except Exception as e:
    st.error(f"Error initializing pipeline: {str(e)}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Retrieve and Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context & generating answer..."):
            retrieved = retrieve(prompt, model, index, chunks, top_k=3)
            response = generate_response(prompt, retrieved)
            
            st.markdown(response)
            
            with st.expander("Show Retrieved Context"):
                for i, r in enumerate(retrieved):
                    st.info(f"**Chunk {i+1}:** {r}")
                    
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
