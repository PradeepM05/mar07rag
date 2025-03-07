import os
import sys
from glob import glob
from typing import List, Dict, Any, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

class RAGSystem:
    def __init__(self, 
                 pdf_folder: str,
                 vector_store_path: str = None,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.0):
        """
        Initialize the RAG system.
        
        Args:
            pdf_folder: Path to the folder containing PDFs
            vector_store_path: Path to save/load the vector store (if None, defaults to pdf_folder/faiss_index)
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks to maintain context
            model_name: OpenAI model to use
            temperature: Temperature for response generation (0.0 = deterministic)
        """
        self.pdf_folder = pdf_folder
        self.vector_store_path = vector_store_path or os.path.join(pdf_folder, "faiss_index")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize memory for conversation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Custom prompt template
        self.prompt_template = """
        You are an AI assistant that answers questions based on provided documents.
        
        Chat History:
        {chat_history}
        
        Context from documents:
        {context}
        
        Question: {question}
        
        Please provide a detailed answer based only on the context provided. If the information is not in the context, 
        say "I don't have enough information about that in the documents." Include specific references to the source 
        documents when possible.
        
        Answer:
        """
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature
        )
        
        # Check if vector store exists or needs to be created
        if os.path.exists(self.vector_store_path):
            print(f"Loading existing vector store from {self.vector_store_path}")
            self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings,allow_dangerous_deserialization=True)
            self.qa_chain = self._create_qa_chain()
        else:
            print(f"Vector store not found at {self.vector_store_path}")
            self.vector_store = None
            self.qa_chain = None

    def process_pdfs(self, force_rebuild: bool = False) -> None:
        """
        Process all PDFs in the folder and build the vector store.
        
        Args:
            force_rebuild: Whether to rebuild the vector store even if it exists
        """
        # Skip if vector store exists and no rebuild is requested
        if self.vector_store is not None and not force_rebuild:
            print("Vector store already exists. Use force_rebuild=True to rebuild.")
            return
            
        # Check if folder exists
        if not os.path.exists(self.pdf_folder):
            raise FileNotFoundError(f"Folder '{self.pdf_folder}' not found")
            
        # Get all PDF files in the folder
        pdf_files = glob(os.path.join(self.pdf_folder, "*.pdf"))
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
                # Add source filename to metadata
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
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
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
        
        # Create QA chain
        self.qa_chain = self._create_qa_chain()
        
        print("RAG system ready for queries")

    def _create_qa_chain(self) -> RetrievalQA:
        """Create a question-answering chain."""
        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name=self.model_name, temperature=self.temperature),
            chain_type="stuff",  # This chain type expects a 'query' input
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={
                "memory": self.memory
            }
        )
        
        return qa_chain

    def query(self, question):
        try:
            # Use the 'query' key as expected by the chain
            result = self.qa_chain.invoke({"query": question})
            return result
        except Exception as e:
            print(f"Error in query: {str(e)}")
            
            # If that fails, try with the deprecated __call__ method
            try:
                # Use the 'query' key as expected by the chain
                result = self.qa_chain({"query": question})
                return result
            except Exception as e2:
                print(f"Error in alternative query: {str(e2)}")
                
                # As a last resort, try with both input formats
                try:
                    # Get documents from the retriever
                    documents = self.vector_store.similarity_search(question)
                    
                    # Try with a combined format
                    result = self.qa_chain.invoke({
                        "query": question,
                        "input_documents": documents,
                        "question": question
                    })
                    return result
                except Exception as e3:
                    print(f"Error in combined query: {str(e3)}")
                    raise e3

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
        """Save the conversation history to a file."""
        if not self.memory.chat_memory.messages:
            print("No conversation to save")
            return
            
        with open(file_path, 'w') as f:
            for message in self.memory.chat_memory.messages:
                role = "Human" if message.type == "human" else "AI"
                f.write(f"{role}: {message.content}\n\n")
        
        print(f"Conversation saved to {file_path}")
        
    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.memory.clear()
        print("Conversation history cleared")