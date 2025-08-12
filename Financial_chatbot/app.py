import streamlit as st
import os
from query_assistant import create_rag_chain # Import the function from your query_assistant.py

# --- Streamlit App ---
st.set_page_config(page_title="Financial Insights Assistant", page_icon="🏦")

st.title("🏦 AI Financial Insights Assistant")
st.markdown("Ask questions about your financial documents and get context-aware answers.")

# Use Streamlit's caching to avoid re-initializing LLM and loading vector store on every rerun
@st.cache_resource
def initialize_rag_chain():
    """Initializes the RAG chain and caches it."""
    with st.spinner("Initializing AI Assistant (this may take a moment)..."):
        return create_rag_chain()

rag_chain = initialize_rag_chain()

if rag_chain is None:
    st.error("Failed to initialize the AI Assistant. Please check your AWS setup and ensure 'faiss_index' exists.")
    st.stop() # Stop the app if initialization fails

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Show Sources"):
                for i, source_doc in enumerate(message["sources"]):
                    st.text(f"Chunk {i+1} from {source_doc.metadata.get('source')}:")
                    st.code(source_doc.page_content[:200] + "...", language='text')

# React to user input
if prompt := st.chat_input("Ask about your financial documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        sources = []

        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({"input": prompt})
                full_response = response["answer"]
                sources = response["context"]

                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"An error occurred: {e}. Please check your model access or input."
                message_placeholder.error(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": sources})

        if sources:
            with st.expander("Show Sources"):
                for i, source_doc in enumerate(sources):
                    st.text(f"Chunk {i+1} from {source_doc.metadata.get('source')}:")
                    st.code(source_doc.page_content[:200] + "...", language='text')