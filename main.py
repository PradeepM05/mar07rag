# File: main.py
import os
import sys
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Add the project root directory to the Python path
#sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#from src.rag_system import SimpleRAGSystem as RAGSystem


from src.rag_system import RAGSystem

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RAG System for PDF documents")
    parser.add_argument("--pdf_folder", type=str, default="data",
                        help="Path to the folder containing PDFs (default: data)")
    parser.add_argument("--vector_store", type=str, default=None,
                        help="Path to save/load the vector store (default: {pdf_folder}/faiss_index)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild of the vector store even if it exists")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="OpenAI model to use (default: gpt-3.5-turbo)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for response generation (default: 0.0)")
    parser.add_argument("--save_conversation", type=str, default=None,
                        help="Path to save the conversation history after exit")
    return parser.parse_args()

def display_help():
    """Display available commands."""
    print("\nAvailable commands:")
    print("  ?                : Show this help message")
    print("  !save FILENAME   : Save conversation to a file")
    print("  !clear           : Clear conversation history")
    print("  !rebuild         : Rebuild vector store from PDFs")
    print("  !exit            : Exit the program")
    print("  Any other input  : Interpreted as a question for the RAG system\n")

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file with your OpenAI API key like this:")
        print("OPENAI_API_KEY=your-api-key-here")
        sys.exit(1)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create RAG system
    rag = RAGSystem(
        pdf_folder=args.pdf_folder,
        vector_store_path=args.vector_store,
        model_name=args.model,
        temperature=args.temperature
    )
    
    # Process PDFs
    try:
        rag.process_pdfs(force_rebuild=args.rebuild)
    except Exception as e:
        print(f"Error processing PDFs: {str(e)}")
        sys.exit(1)
    
    # Interactive query loop
    print("\n=== RAG System initialized and ready ===")
    print("You can now ask questions about your PDFs.")
    print("Type '?' for help or '!exit' to quit.")
    
    while True:
        query = input("\nQuestion: ")
        
        # Check for special commands
        if query.lower() in ["exit", "quit", "q", "!exit"]:
            break
        elif query == "?":
            display_help()
            continue
        elif query.startswith("!save "):
            filename = query[6:].strip()
            if not filename:
                print("Please specify a filename")
                continue
            rag.save_conversation(filename)
            continue
        elif query == "!clear":
            rag.clear_conversation()
            continue
        elif query == "!rebuild":
            print("Rebuilding vector store...")
            rag.process_pdfs(force_rebuild=True)
            continue
            
        # Get answer
        try:
            result = rag.direct_query(query)
            
            # Display answer
            print("\nAnswer:", result["result"])
            
            # Display sources
            print("\nSources:")
            print(rag.format_sources(result["source_documents"]))
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Save conversation if requested
    if args.save_conversation:
        rag.save_conversation(args.save_conversation)
        
    print("Goodbye!")

if __name__ == "__main__":
    main()