# File: src/rag_system.py
import os
import glob
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

# Import single provider variable to control which LLM to use
USE_CLAUDE = False  # Set to False to use OpenAI

class RAGSystem:
    def __init__(self, 
                 pdf_folder: str,
                 vector_store_path: str = None,
                 temperature: float = 0.0):
        """
        Initialize a simple RAG system with a single toggle between Claude and OpenAI.
        
        Args:
            pdf_folder: Path to the folder containing PDFs
            vector_store_path: Path to save/load the vector store
            temperature: Temperature for response generation
        """
        self.pdf_folder = pdf_folder
        self.vector_store_path = vector_store_path or os.path.join(pdf_folder, "faiss_index")
        self.temperature = temperature
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Initialize embeddings (using OpenAI for embeddings regardless of LLM choice)
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize the LLM based on the USE_CLAUDE toggle
        if USE_CLAUDE:
            # Check if Anthropic API key is set
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
                
            # Initialize Claude
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(
                model="claude-3-7-sonnet-20250219",
                temperature=temperature,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            print("Using Claude 3.7 Sonnet for responses")
        else:
            # Check if OpenAI API key is set
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable not set")
                
            # Initialize OpenAI
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo", 
                temperature=temperature
            )
            print("Using GPT-3.5 Turbo for responses")
        
        # Initialize vector store
        self.vector_store = None
        
        # Load vector store if it exists
        if os.path.exists(self.vector_store_path):
            print(f"Loading existing vector store from {self.vector_store_path}")
            self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)

    def process_pdfs(self, force_rebuild: bool = False) -> None:
        """Process PDFs and build vector store."""
        if self.vector_store is not None and not force_rebuild:
            print("Vector store already exists. Use force_rebuild=True to rebuild.")
            return
            
        # Check if folder exists
        if not os.path.exists(self.pdf_folder):
            raise FileNotFoundError(f"Folder '{self.pdf_folder}' not found")
            
        # Get all PDF files in the folder
        pdf_files = glob.glob(os.path.join(self.pdf_folder, "*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in '{self.pdf_folder}'")
        
        print(f"Found {len(pdf_files)} PDF files in {self.pdf_folder}")
        
        # Load all PDFs
        all_documents = []
        for pdf_path in pdf_files:
            print(f"Processing: {os.path.basename(pdf_path)}")
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                for doc in documents:
                    doc.metadata["source"] = os.path.basename(pdf_path)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
        
        if not all_documents:
            raise ValueError("No documents were successfully loaded")
            
        print(f"Loaded {len(all_documents)} total pages from all PDFs")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(all_documents)
        print(f"Split into {len(chunks)} chunks")
        
        # Create vector store
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # Save the vector store
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        self.vector_store.save_local(self.vector_store_path)
        print(f"Created and saved vector store to {self.vector_store_path}")
        
        print("RAG system ready for queries")

    def direct_query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with a question and return result with sources."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Please process PDFs first.")
        
        # Add question to conversation history
        self.conversation_history.append({"role": "user", "content": question})
        
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(question, k=4)
        
        # Format context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Universal prompt that works well for both models
        prompt_template = """
        You are a helpful assistant answering questions based on the provided documents.

        Context from documents:
        {context}

        Previous conversation:
        {chat_history}

        Question: {question}

        Answer based only on the information in the context. If the information isn't available in the context, say "I don't have enough information about that in the documents." Include specific references to sources when possible.
        """
        
        # Format conversation history for the prompt
        chat_history_text = ""
        for i in range(0, len(self.conversation_history) - 1, 2):
            if i+1 < len(self.conversation_history):
                user_msg = self.conversation_history[i]["content"]
                assistant_msg = self.conversation_history[i+1]["content"] if i+1 < len(self.conversation_history) else ""
                chat_history_text += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
        
        # Generate the answer
        full_prompt = prompt_template.format(
            chat_history=chat_history_text,
            context=context,
            question=question
        )
        
        response = self.llm.invoke(full_prompt)
        answer = response.content
        
        # Add answer to conversation history
        self.conversation_history.append({"role": "assistant", "content": answer})
        
        # Return formatted result
        return {
            "result": answer,
            "source_documents": docs
        }

    def format_sources(self, source_docs: List[Any]) -> str:
        """Format source documents for display."""
        sources_text = ""
        for i, doc in enumerate(source_docs):
            source_file = doc.metadata.get("source", "Unknown")
            page_num = doc.metadata.get("page", 0) + 1
            sources_text += f"Source {i+1} ({source_file}, Page {page_num}):\n"
            sources_text += f"{doc.page_content[:150]}...\n\n"
        return sources_text

    def save_conversation(self, file_path: str) -> None:
        """Save conversation history to a file."""
        if not self.conversation_history:
            print("No conversation to save")
            return
            
        with open(file_path, 'w') as f:
            for i in range(0, len(self.conversation_history), 2):
                if i < len(self.conversation_history):
                    user_msg = self.conversation_history[i]["content"]
                    f.write(f"Human: {user_msg}\n\n")
                    
                if i+1 < len(self.conversation_history):
                    assistant_msg = self.conversation_history[i+1]["content"]
                    f.write(f"AI: {assistant_msg}\n\n")
        
        print(f"Conversation saved to {file_path}")
        
    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        print("Conversation history cleared")