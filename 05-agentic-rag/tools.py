"""
Agentic RAG - Retrieval Tools
Tools the agent can use for retrieval.

Uses existing MongoDB vector collections - no new ingestion needed.
"""

import os
import certifi
from dotenv import load_dotenv
from typing import Optional

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # sandbox
    from llm_shim import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
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

# MongoDB configuration - uses metadata_filtered collection by default
DB_NAME = "rag_playbook"
COLLECTION_NAME = "metadata_filtered_rag"
INDEX_NAME = "metadata_filtered_index"


def get_mongo_client():
    """Get MongoDB client."""
    return MongoClient(MONGO_DB_URL, **MONGO_KWARGS)


def get_vector_store():
    """Connect to MongoDB vector store."""
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


# =============================================================================
# TOOL: No Retrieval
# =============================================================================

def no_retrieval(query: str) -> dict:
    """
    Skip retrieval - agent decides to answer from model knowledge.
    
    Returns:
        Empty result indicating no retrieval was done
    """
    return {
        "tool": "no_retrieval",
        "query": query,
        "documents": [],
        "message": "Agent decided retrieval is not needed for this query."
    }


# =============================================================================
# TOOL: Vector Search
# =============================================================================

def vector_search(query: str, k: int = 5) -> dict:
    """
    Standard vector similarity search.
    
    Args:
        query: Search query
        k: Number of documents to retrieve
    
    Returns:
        Retrieved documents and metadata
    """
    vector_store, client = get_vector_store()
    
    try:
        results = vector_store.similarity_search(query=query, k=k)
        
        return {
            "tool": "vector_search",
            "query": query,
            "documents": results,
            "sources": [doc.metadata.get("source_file") for doc in results]
        }
    finally:
        client.close()


# =============================================================================
# TOOL: Filtered Search
# =============================================================================

def build_pre_filter(
    year: Optional[int] = None,
    year_range: Optional[tuple[int, int]] = None,
    decade: Optional[int] = None,
    topic_buckets: Optional[list[str]] = None,
    companies_mentioned: Optional[list[str]] = None,
    has_financials: Optional[bool] = None,
) -> dict:
    """Build MongoDB pre-filter."""
    conditions = []
    
    if year is not None:
        conditions.append({"year": {"$eq": year}})
    
    if year_range is not None:
        start_year, end_year = year_range
        conditions.append({
            "$and": [
                {"year": {"$gte": start_year}},
                {"year": {"$lte": end_year}}
            ]
        })
    
    if decade is not None:
        conditions.append({"decade": {"$eq": decade}})
    
    if topic_buckets is not None:
        conditions.append({"topic_buckets": {"$in": topic_buckets}})
    
    if companies_mentioned is not None:
        conditions.append({"companies_mentioned": {"$in": companies_mentioned}})
    
    if has_financials is not None:
        conditions.append({"has_financials": {"$eq": has_financials}})
    
    if not conditions:
        return {}
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


def filtered_search(
    query: str,
    k: int = 5,
    year: Optional[int] = None,
    year_range: Optional[tuple[int, int]] = None,
    decade: Optional[int] = None,
    topic_buckets: Optional[list[str]] = None,
    companies_mentioned: Optional[list[str]] = None,
    has_financials: Optional[bool] = None,
) -> dict:
    """
    Vector search with metadata pre-filtering.
    
    Args:
        query: Search query
        k: Number of documents
        year: Filter by specific year
        year_range: Filter by (start, end) year range
        decade: Filter by decade (2000, 2010, 2020)
        topic_buckets: Filter by topics
        companies_mentioned: Filter by company mentions
        has_financials: Filter for financial content
    
    Returns:
        Retrieved documents and metadata
    """
    vector_store, client = get_vector_store()
    
    try:
        pre_filter = build_pre_filter(
            year=year,
            year_range=year_range,
            decade=decade,
            topic_buckets=topic_buckets,
            companies_mentioned=companies_mentioned,
            has_financials=has_financials
        )
        
        if pre_filter:
            results = vector_store.similarity_search(
                query=query,
                k=k,
                pre_filter=pre_filter
            )
        else:
            results = vector_store.similarity_search(query=query, k=k)
        
        # Build filter description
        filter_desc = []
        if year:
            filter_desc.append(f"year={year}")
        if year_range:
            filter_desc.append(f"years={year_range[0]}-{year_range[1]}")
        if decade:
            filter_desc.append(f"decade={decade}s")
        if topic_buckets:
            filter_desc.append(f"topics={topic_buckets}")
        if companies_mentioned:
            filter_desc.append(f"companies={companies_mentioned}")
        if has_financials:
            filter_desc.append("has_financials=True")
        
        return {
            "tool": "filtered_search",
            "query": query,
            "filters": ", ".join(filter_desc) if filter_desc else "none",
            "documents": results,
            "sources": [doc.metadata.get("source_file") for doc in results]
        }
    finally:
        client.close()


# =============================================================================
# Tool Registry
# =============================================================================

AVAILABLE_TOOLS = {
    "no_retrieval": {
        "function": no_retrieval,
        "description": "Skip retrieval and answer from model knowledge. Use for simple factual questions or when retrieval isn't needed."
    },
    "vector_search": {
        "function": vector_search,
        "description": "Semantic similarity search across all documents. Use for general questions about content."
    },
    "filtered_search": {
        "function": filtered_search,
        "description": "Vector search with metadata filters (year, topic, company). Use for specific time periods, topics, or companies."
    }
}


def format_documents_as_context(documents: list[Document]) -> str:
    """Format documents into context string for LLM."""
    if not documents:
        return "No documents retrieved."
    
    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_file", "Unknown")
        year = doc.metadata.get("year", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        
        parts.append(
            f"[Document {i}]\n"
            f"Source: {source} (Year: {year}, Page: {page})\n"
            f"Content:\n{doc.page_content}\n"
        )
    
    return "\n---\n".join(parts)
