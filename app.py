# File: app.py
import os
import streamlit as st
from dotenv import load_dotenv
import time
from src.rag_system import RAGSystem, USE_CLAUDE

# Load environment variables
load_dotenv()

# Check if API keys are set
if USE_CLAUDE and not os.getenv("ANTHROPIC_API_KEY"):
    st.error("Error: ANTHROPIC_API_KEY environment variable not set but you're trying to use Claude.")
    st.stop()
elif not USE_CLAUDE and not os.getenv("OPENAI_API_KEY"):
    st.error("Error: OPENAI_API_KEY environment variable not set but you're trying to use OpenAI.")
    st.stop()

# Check if OpenAI API key exists (needed for embeddings regardless of LLM choice)
if not os.getenv("OPENAI_API_KEY"):
    st.error("Error: OPENAI_API_KEY environment variable not set. This is needed for embeddings.")
    st.stop()

# Set page config
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

# Title and description
st.title("📚 PDF RAG Assistant")
model_name = "Claude 3.7 Sonnet" if USE_CLAUDE else "GPT-3.5 Turbo"
st.markdown(f"Ask questions about your PDF documents using {model_name}.")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    
    # Show current LLM provider
    st.info(f"Currently using: {model_name}\n\nTo switch LLM provider, change the USE_CLAUDE variable in src/rag_system.py")
    
    # PDF folder selection
    pdf_folder = st.text_input("PDF Folder Path", value="data")
    
    # Vector store path
    vector_store = st.text_input("Vector Store Path (optional)", value="")
    
    # Temperature setting
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    
    # Rebuild vector store button
    rebuild = st.button("Rebuild Vector Store")
    
    # Clear conversation button
    clear_conversation = st.button("Clear Conversation")
    
    # Save conversation
    save_path = st.text_input("Save conversation to file", value="conversation.txt")
    save_conversation = st.button("Save Conversation")
    
    # About section
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        f"This app uses {model_name} and vector search to answer questions about your PDF documents. "
        "Upload PDFs to the specified folder and ask away!"
    )

# Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Initialize RAG system
@st.cache_resource(show_spinner=False)
def initialize_rag(pdf_folder, vector_store_path, temperature, force_rebuild=False):
    with st.spinner(f"Initializing RAG system with {model_name}..."):
        rag = RAGSystem(
            pdf_folder=pdf_folder,
            vector_store_path=vector_store_path if vector_store_path else None,
            temperature=temperature
        )
        
        try:
            rag.process_pdfs(force_rebuild=force_rebuild)
            return rag
        except Exception as e:
            st.error(f"Error initializing RAG system: {str(e)}")
            return None

# Initialize or reinitialize RAG system
if not st.session_state.initialized or rebuild:
    with st.spinner(f"Initializing RAG system with {model_name}..."):
        st.session_state.rag = initialize_rag(pdf_folder, vector_store, temperature, force_rebuild=rebuild)
        if st.session_state.rag:
            st.session_state.initialized = True
            if rebuild:
                st.success("Vector store rebuilt successfully!")

# Handle clear conversation
if clear_conversation:
    if st.session_state.rag:
        st.session_state.rag.clear_conversation()
        st.session_state.messages = []
        st.success("Conversation cleared!")

# Handle save conversation
if save_conversation:
    if st.session_state.rag and save_path:
        st.session_state.rag.save_conversation(save_path)
        st.success(f"Conversation saved to {save_path}!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Sources"):
                st.markdown(message["sources"])

# Chat input
if st.session_state.initialized:
    if prompt := st.chat_input("Ask a question about your PDFs"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Show a spinner while processing
            with st.spinner(f"Searching documents and generating answer with {model_name}..."):
                try:
                    # Query RAG system
                    result = st.session_state.rag.direct_query(prompt)
                    answer = result["result"]
                    sources = st.session_state.rag.format_sources(result["source_documents"])
                    
                    # Simulate typing
                    for chunk in answer.split():
                        full_response += chunk + " "
                        time.sleep(0.01)
                        message_placeholder.markdown(full_response + "▌")
                    
                    # Display final response
                    message_placeholder.markdown(full_response)
                    
                    # Show sources
                    with st.expander("Sources"):
                        st.markdown(sources)
                    
                    # Add AI response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
else:
    st.info("Initialize the RAG system to start asking questions.")