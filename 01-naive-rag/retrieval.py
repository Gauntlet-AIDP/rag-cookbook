"""
Naive RAG - Retrieval Module
Performs vector search against MongoDB to retrieve relevant document chunks.
"""

import os
import certifi
from dotenv import load_dotenv

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # sandbox
    from llm_shim import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Configuration
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
# Atlas (mongodb+srv, mongodb.net or tls/ssl=true) needs the certifi CA bundle; a plain local URL does not
_TLS_HINTS = ("mongodb+srv", "mongodb.net", "tls=true", "ssl=true")
MONGO_KWARGS = {"tlsCAFile": certifi.where()} if any(h in (MONGO_DB_URL or "").lower() for h in _TLS_HINTS) else {}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB configuration (must match ingestion.py)
DB_NAME = "rag_playbook"
COLLECTION_NAME = "naive_rag"
INDEX_NAME = "naive"

# Retrieval configuration
DEFAULT_TOP_K = 5


def get_vector_store():
    """Connect to the MongoDB vector store."""
    client = MongoClient(MONGO_DB_URL, **MONGO_KWARGS)
    collection = client[DB_NAME][COLLECTION_NAME]
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    
    return vector_store, client


def retrieve_documents(query: str, top_k: int = DEFAULT_TOP_K) -> list:
    """
    Retrieve the top-k most relevant documents for a given query.
    
    Args:
        query: The search query string
        top_k: Number of documents to retrieve (default: 5)
    
    Returns:
        List of relevant document chunks with metadata
    """
    vector_store, client = get_vector_store()
    
    try:
        # Perform similarity search
        results = vector_store.similarity_search(
            query=query,
            k=top_k
        )
        return results
    finally:
        client.close()


def retrieve_documents_with_scores(query: str, top_k: int = DEFAULT_TOP_K) -> list:
    """
    Retrieve the top-k most relevant documents with similarity scores.
    
    Args:
        query: The search query string
        top_k: Number of documents to retrieve (default: 5)
    
    Returns:
        List of tuples (document, score)
    """
    vector_store, client = get_vector_store()
    
    try:
        # Perform similarity search with scores
        results = vector_store.similarity_search_with_score(
            query=query,
            k=top_k
        )
        return results
    finally:
        client.close()


def format_retrieved_context(documents: list) -> str:
    """
    Format retrieved documents into a context string for the LLM.
    
    Args:
        documents: List of retrieved document objects
    
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_file", "Unknown")
        year = doc.metadata.get("year", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        
        context_parts.append(
            f"[Document {i}]\n"
            f"Source: {source} (Year: {year}, Page: {page})\n"
            f"Content:\n{doc.page_content}\n"
        )
    
    return "\n---\n".join(context_parts)


def debug_collection():
    """Debug function to check MongoDB collection status."""
    client = MongoClient(MONGO_DB_URL, **MONGO_KWARGS)
    collection = client[DB_NAME][COLLECTION_NAME]
    
    print("=" * 50)
    print("DEBUG: MongoDB Collection Status")
    print("=" * 50)
    
    # Check document count
    doc_count = collection.count_documents({})
    print(f"\nTotal documents in collection: {doc_count}")
    
    if doc_count > 0:
        # Get a sample document to check structure
        sample = collection.find_one()
        print(f"\nSample document fields: {list(sample.keys())}")
        
        # Check if embedding field exists
        if "embedding" in sample:
            print(f"Embedding field exists with {len(sample['embedding'])} dimensions")
        else:
            print("WARNING: 'embedding' field NOT found!")
            print(f"Available fields: {list(sample.keys())}")
    
    # List indexes
    print("\nIndexes on collection:")
    for index in collection.list_indexes():
        print(f"  - {index['name']}: {index.get('key', 'N/A')}")
    
    # Try to list search indexes (Atlas Vector Search)
    try:
        search_indexes = list(collection.list_search_indexes())
        print(f"\nVector Search Indexes: {len(search_indexes)}")
        for idx in search_indexes:
            print(f"  - Name: {idx.get('name')}")
            print(f"    Status: {idx.get('status', 'unknown')}")
            print(f"    Definition: {idx.get('latestDefinition', idx.get('definition', 'N/A'))}")
    except Exception as e:
        print(f"\nCould not list search indexes: {e}")
    
    client.close()
    return doc_count


def main():
    """Test retrieval with a sample query."""
    print("=" * 50)
    print("Naive RAG - Retrieval Test")
    print("=" * 50)
    
    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # First, debug the collection
    doc_count = debug_collection()
    
    if doc_count == 0:
        print("\n❌ No documents found! Run ingestion.py first.")
        return
    
    print("\n" + "=" * 50)
    print("Running Retrieval Test")
    print("=" * 50)
    
    # Test query
    test_query = "What are Warren Buffett's thoughts on investment philosophy?"
    print(f"\nQuery: {test_query}")
    print("-" * 50)
    
    # Retrieve documents with scores
    results = retrieve_documents_with_scores(test_query, top_k=3)
    
    print(f"\nRetrieved {len(results)} documents:\n")
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"--- Document {i} (Score: {score:.4f}) ---")
        print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
        print(f"Year: {doc.metadata.get('year', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"Content preview: {doc.page_content[:300]}...")
        print()
    
    # Show formatted context
    documents = [doc for doc, _ in results]
    context = format_retrieved_context(documents)
    print("\n" + "=" * 50)
    print("Formatted Context for LLM:")
    print("=" * 50)
    print(context[:1000] + "..." if len(context) > 1000 else context)


if __name__ == "__main__":
    main()
