"""
Hybrid Search RAG - Retrieval Module
Combines BM25 (keyword search) with vector search (semantic search).

NO INGESTION REQUIRED - Reuses existing vectorized documents from:
- naive_rag collection (from naive/ ingestion)
- metadata_filtered_rag collection (from metadata-filtered/ ingestion)

Uses Reciprocal Rank Fusion (RRF) to combine results from both methods.
"""

import os
import certifi
from dotenv import load_dotenv
from typing import Optional
from collections import defaultdict

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # sandbox
    from llm_shim import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Configuration
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
# Atlas (mongodb+srv, mongodb.net or tls/ssl=true) needs the certifi CA bundle; a plain local URL does not
_TLS_HINTS = ("mongodb+srv", "mongodb.net", "tls=true", "ssl=true")
MONGO_KWARGS = {"tlsCAFile": certifi.where()} if any(h in (MONGO_DB_URL or "").lower() for h in _TLS_HINTS) else {}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB configuration - Choose which collection to use
# Option 1: Use naive RAG collection
# DB_NAME = "rag_playbook"
# COLLECTION_NAME = "naive_rag"
# INDEX_NAME = "naive"

# Option 2: Use metadata-filtered collection (default - has richer metadata)
DB_NAME = "rag_playbook"
COLLECTION_NAME = "metadata_filtered_rag"
INDEX_NAME = "metadata_filtered_index"

# Hybrid search configuration
DEFAULT_TOP_K = 5
DEFAULT_BM25_WEIGHT = 0.5  # Weight for BM25 results
DEFAULT_VECTOR_WEIGHT = 0.5  # Weight for vector results
RRF_K = 60  # Reciprocal Rank Fusion constant


def get_mongo_client():
    """Get MongoDB client."""
    return MongoClient(MONGO_DB_URL, **MONGO_KWARGS)


def get_vector_store():
    """Connect to the MongoDB vector store."""
    client = get_mongo_client()
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


def load_documents_for_bm25(limit: Optional[int] = None) -> list[Document]:
    """
    Load documents from MongoDB for BM25 indexing.
    
    Note: BM25 requires all documents in memory. For very large collections,
    consider using a subset or MongoDB's text search instead.
    """
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    try:
        cursor = collection.find()
        if limit:
            cursor = cursor.limit(limit)
        
        documents = []
        for doc in cursor:
            text = doc.get("text", "")
            if not text:
                continue
            
            metadata = {
                "source_file": doc.get("source_file", "Unknown"),
                "year": doc.get("year", 0),
                "page": doc.get("page", 0),
                "_id": str(doc.get("_id", "")),  # For deduplication
            }
            
            if "topic_buckets" in doc:
                metadata["topic_buckets"] = doc["topic_buckets"]
            if "companies_mentioned" in doc:
                metadata["companies_mentioned"] = doc["companies_mentioned"]
            if "decade" in doc:
                metadata["decade"] = doc["decade"]
            
            documents.append(Document(
                page_content=text,
                metadata=metadata
            ))
        
        return documents
    finally:
        client.close()


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    weights: list[float],
    k: int = RRF_K
) -> list[Document]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    
    RRF Score = sum(weight_i / (k + rank_i)) for each list
    
    Args:
        result_lists: List of ranked document lists
        weights: Weight for each list
        k: RRF constant (default 60)
    
    Returns:
        Combined and reranked list of documents
    """
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Calculate RRF scores
    doc_scores = defaultdict(float)
    doc_map = {}  # Map content hash to document
    
    for list_idx, doc_list in enumerate(result_lists):
        weight = weights[list_idx]
        for rank, doc in enumerate(doc_list):
            # Use content as key for deduplication
            doc_key = hash(doc.page_content)
            
            # RRF formula
            rrf_score = weight / (k + rank + 1)
            doc_scores[doc_key] += rrf_score
            
            # Store document (keep first occurrence)
            if doc_key not in doc_map:
                doc_map[doc_key] = doc
    
    # Sort by RRF score
    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    
    # Return reranked documents
    return [doc_map[key] for key in sorted_keys]


def hybrid_search(
    query: str,
    k: int = DEFAULT_TOP_K,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    bm25_k: int = None,
    vector_k: int = None,
    documents: list[Document] = None,
) -> list[Document]:
    """
    Perform hybrid search combining BM25 and vector search.
    
    Args:
        query: Search query
        k: Number of final results
        bm25_weight: Weight for BM25 results
        vector_weight: Weight for vector results
        bm25_k: Number of BM25 candidates
        vector_k: Number of vector candidates
        documents: Pre-loaded documents (optional, for caching)
    
    Returns:
        List of top-k documents from hybrid search
    """
    # Set candidate counts
    if bm25_k is None:
        bm25_k = k * 3
    if vector_k is None:
        vector_k = k * 3
    
    # Load documents for BM25 if not provided
    if documents is None:
        documents = load_documents_for_bm25()
    
    if not documents:
        raise ValueError("No documents found. Run ingestion first!")
    
    # BM25 search
    bm25_retriever = BM25Retriever.from_documents(documents, k=bm25_k)
    bm25_results = bm25_retriever.invoke(query)
    
    # Vector search
    vector_store, client = get_vector_store()
    try:
        vector_results = vector_store.similarity_search(query=query, k=vector_k)
    finally:
        client.close()
    
    # Combine with RRF
    combined = reciprocal_rank_fusion(
        result_lists=[bm25_results, vector_results],
        weights=[bm25_weight, vector_weight]
    )
    
    return combined[:k]


def retrieve_and_compare(
    query: str,
    k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Retrieve using all three methods and compare results.
    """
    print(f"\n🔍 Query: {query}")
    print("-" * 50)
    
    # Load documents once
    print("📚 Loading documents...")
    documents = load_documents_for_bm25()
    print(f"   Loaded {len(documents)} documents")
    
    # BM25 only
    print("📝 Running BM25 search...")
    bm25_retriever = BM25Retriever.from_documents(documents, k=k)
    bm25_results = bm25_retriever.invoke(query)
    
    # Vector only
    print("🔢 Running vector search...")
    vector_store, client = get_vector_store()
    try:
        vector_results = vector_store.similarity_search(query=query, k=k)
    finally:
        client.close()
    
    # Hybrid
    print("🔀 Running hybrid search...")
    hybrid_results = hybrid_search(query, k=k, documents=documents)
    
    return {
        "query": query,
        "bm25": {
            "documents": bm25_results,
            "sources": [d.metadata.get("source_file") for d in bm25_results]
        },
        "vector": {
            "documents": vector_results,
            "sources": [d.metadata.get("source_file") for d in vector_results]
        },
        "hybrid": {
            "documents": hybrid_results,
            "sources": [d.metadata.get("source_file") for d in hybrid_results]
        }
    }


def format_retrieved_context(documents: list) -> str:
    """Format retrieved documents into context string for LLM."""
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
    """Check collection status."""
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    print("=" * 50)
    print("Hybrid Search - Collection Status")
    print("=" * 50)
    print(f"\nUsing collection: {DB_NAME}.{COLLECTION_NAME}")
    
    doc_count = collection.count_documents({})
    print(f"Total documents: {doc_count}")
    
    if doc_count == 0:
        print("\n⚠️  No documents found!")
        print("   Run ingestion in naive/ or metadata-filtered/ first.")
    else:
        sample = collection.find_one()
        print(f"Sample fields: {list(sample.keys())}")
    
    client.close()
    return doc_count


def main():
    """Test hybrid retrieval and compare with individual methods."""
    print("=" * 60)
    print("Hybrid Search RAG - Retrieval Test")
    print("=" * 60)
    print("\n📌 NOTE: No ingestion required!")
    print(f"   Using existing collection: {DB_NAME}.{COLLECTION_NAME}")
    
    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Check collection
    doc_count = debug_collection()
    if doc_count == 0:
        return
    
    # Test queries
    test_queries = [
        {
            "query": "GEICO insurance operations",
            "expected_strength": "BM25 (specific company name)",
        },
        {
            "query": "What makes a good long-term investment?",
            "expected_strength": "Vector (semantic understanding)",
        },
        {
            "query": "Berkshire Hathaway float 2020",
            "expected_strength": "Hybrid (terms + meaning)",
        },
    ]
    
    print("\n" + "=" * 60)
    print("Comparing Retrieval Methods")
    print("=" * 60)
    
    for test in test_queries:
        query = test["query"]
        expected = test["expected_strength"]
        
        results = retrieve_and_compare(query, k=3)
        
        print(f"\n📝 Query: {query}")
        print(f"   Expected strength: {expected}")
        
        print(f"\n   BM25 Results:")
        for src in results["bm25"]["sources"]:
            print(f"      - {src}")
        
        print(f"\n   Vector Results:")
        for src in results["vector"]["sources"]:
            print(f"      - {src}")
        
        print(f"\n   Hybrid Results:")
        for src in results["hybrid"]["sources"]:
            print(f"      - {src}")
        
        # Show overlap
        bm25_set = set(results["bm25"]["sources"])
        vector_set = set(results["vector"]["sources"])
        
        overlap = bm25_set & vector_set
        unique_bm25 = bm25_set - vector_set
        unique_vector = vector_set - bm25_set
        
        print(f"\n   Overlap (both methods): {len(overlap)}")
        print(f"   Unique to BM25: {len(unique_bm25)}")
        print(f"   Unique to Vector: {len(unique_vector)}")


if __name__ == "__main__":
    main()
