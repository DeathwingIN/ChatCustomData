import os
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredPDFLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

# Configure paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATHS = [
    os.path.join(BASE_DIR, "data/pdf"),
    os.path.join(BASE_DIR, "uploads/pdf")
]
TXT_PATHS = [
    os.path.join(BASE_DIR, "data/txt"),
    os.path.join(BASE_DIR, "uploads/txt")
]
CHROMA_PATH = os.path.join(BASE_DIR, "vector_db")

def load_documents():
    """Load documents from all configured directories"""
    all_docs = []
    
    # Load PDF documents
    for pdf_path in PDF_PATHS:
        if os.path.exists(pdf_path) and os.listdir(pdf_path):
            loader = DirectoryLoader(
                pdf_path,
                glob="**/*.pdf",
                loader_cls=UnstructuredPDFLoader,
                loader_kwargs={"mode": "single", "strategy": "fast"},
                show_progress=True
            )
            try:
                all_docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading PDFs from {pdf_path}: {str(e)}")
    
    # Load TXT documents
    for txt_path in TXT_PATHS:
        if os.path.exists(txt_path) and os.listdir(txt_path):
            loader = DirectoryLoader(
                txt_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True
            )
            try:
                all_docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading TXTs from {txt_path}: {str(e)}")
    
    return all_docs

def process_documents():
    """Process documents into vector DB with enhanced error handling"""
    try:
        documents = load_documents()
        if not documents:
            print("No documents found in any configured directories")
            return None
        
        # Clean metadata
        filtered_docs = filter_complex_metadata(documents)
        for doc in filtered_docs:
            doc.metadata = {
                "source": os.path.basename(doc.metadata.get("source", "unknown")),
                "page": doc.metadata.get("page_number", 0)
            }
        
        # Text splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )
        chunks = splitter.split_documents(filtered_docs)
        if not chunks:
            raise ValueError("No valid chunks created - check document contents")
        
        # Embedding and DB creation
        embedding = OllamaEmbeddings(model="nomic-embed-text")
        filtered_chunks = filter_complex_metadata(chunks)
        
        # Chroma DB configuration
        db = Chroma.from_documents(
            documents=filtered_chunks,
            embedding=embedding,
            persist_directory=CHROMA_PATH,
            collection_metadata={"hnsw:space": "cosine"},
        )
        db.persist()
        return db
    
    except Exception as e:
        print(f"Critical error in processing: {str(e)}")
        return None