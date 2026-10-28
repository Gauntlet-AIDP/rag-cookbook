"""
Naive RAG - Ingestion Pipeline
Loads PDFs from the letters directory, chunks them, and stores vectors in MongoDB.
"""

import os
import certifi
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # sandbox
    from llm_shim import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

try:
    from llm_shim import ensure_vector_index  # sandbox only: creates the index Atlas UI would
except ImportError:
    ensure_vector_index = None

# Load environment variables
load_dotenv()

# Configuration
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
# Atlas (mongodb+srv, mongodb.net or tls/ssl=true) needs the certifi CA bundle; a plain local URL does not
_TLS_HINTS = ("mongodb+srv", "mongodb.net", "tls=true", "ssl=true")
MONGO_KWARGS = {"tlsCAFile": certifi.where()} if any(h in (MONGO_DB_URL or "").lower() for h in _TLS_HINTS) else {}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB configuration
DB_NAME = "rag_playbook"
COLLECTION_NAME = "naive_rag"
INDEX_NAME = "naive"

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def get_pdf_files(letters_dir: str) -> list[Path]:
    """Get all PDF files from the letters directory."""
    letters_path = Path(letters_dir)
    pdf_files = sorted(letters_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    return pdf_files


def load_and_chunk_pdfs(pdf_files: list[Path]) -> list:
    """Load PDFs and split into chunks."""
    all_documents = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata["source_file"] = pdf_path.name
                doc.metadata["year"] = pdf_path.stem.replace("ltr", "")
            
            # Split documents into chunks
            chunks = text_splitter.split_documents(documents)
            all_documents.extend(chunks)
            print(f"  -> Created {len(chunks)} chunks")
            
        except Exception as e:
            print(f"  -> Error processing {pdf_path.name}: {e}")
    
    print(f"\nTotal chunks created: {len(all_documents)}")
    return all_documents


def setup_mongodb_collection():
    """Set up MongoDB collection for vector storage."""
    client = MongoClient(MONGO_DB_URL, **MONGO_KWARGS)
    db = client[DB_NAME]
    
    # Create collection if it doesn't exist
    if COLLECTION_NAME not in db.list_collection_names():
        db.create_collection(COLLECTION_NAME)
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        # Clear existing documents for fresh ingestion
        db[COLLECTION_NAME].delete_many({})
        print(f"Cleared existing documents in: {COLLECTION_NAME}")
    
    return client, db[COLLECTION_NAME]


def create_vector_store(collection, documents: list):
    """Create embeddings and store in MongoDB Atlas Vector Search."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    print("\nCreating embeddings and storing in MongoDB...")
    
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=documents,
        embedding=embeddings,
        collection=collection,
        index_name=INDEX_NAME
    )
    
    print(f"Successfully stored {len(documents)} document chunks in MongoDB")
    return vector_store


def print_vector_search_index_instructions():
    """Print instructions for creating the vector search index in MongoDB Atlas."""
    print("\n" + "=" * 70)
    print("IMPORTANT: Create Vector Search Index in MongoDB Atlas")
    print("=" * 70)
    print("""
To enable vector search, create an index in MongoDB Atlas with this definition:

1. Go to MongoDB Atlas → Your Cluster → Atlas Search → Create Search Index
2. Select "JSON Editor" and use this configuration:

{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }
  ]
}

3. Set the index name to: naive
4. Select database: rag_playbook
5. Select collection: naive_rag

After creating the index, wait for it to become "Active" before running queries.
""")
    print("=" * 70)


def main():
    """Main ingestion pipeline."""
    print("=" * 50)
    print("Naive RAG - Ingestion Pipeline")
    print("=" * 50)
    
    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Get path to letters directory (relative to this script)
    script_dir = Path(__file__).parent
    letters_dir = script_dir.parent / "letters"
    
    if not letters_dir.exists():
        raise FileNotFoundError(f"Letters directory not found: {letters_dir}")
    
    # Step 1: Get PDF files
    pdf_files = get_pdf_files(letters_dir)
    
    if not pdf_files:
        raise ValueError("No PDF files found in letters directory")
    
    # Step 2: Load and chunk PDFs
    documents = load_and_chunk_pdfs(pdf_files)
    
    # Step 3: Setup MongoDB
    client, collection = setup_mongodb_collection()
    
    try:
        # Step 4: Create embeddings and store in vector database
        vector_store = create_vector_store(collection, documents)
        
        print("\n✅ Ingestion complete!")
        print(f"   Database: {DB_NAME}")
        print(f"   Collection: {COLLECTION_NAME}")
        print(f"   Documents stored: {len(documents)}")
        
        # Print instructions for creating the vector search index
        if ensure_vector_index:
            ensure_vector_index(collection, INDEX_NAME)
        else:
            print_vector_search_index_instructions()
        
    finally:
        client.close()


if __name__ == "__main__":
    main()
