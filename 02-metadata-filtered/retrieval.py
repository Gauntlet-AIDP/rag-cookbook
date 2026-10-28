"""
Metadata-Filtered RAG - Retrieval Module
Performs metadata filtering BEFORE vector search for more targeted results.

Supported filters:
- year: Filter by specific year(s)
- year_range: Filter by year range (start_year, end_year)
- decade: Filter by decade (2000, 2010, 2020)
- source_file: Filter by specific source file(s)
- topic_buckets: Filter by topic(s) - insurance, acquisitions, investments, etc.
- companies_mentioned: Filter by portfolio company mentions
- has_financials: Filter for chunks with financial figures
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
COLLECTION_NAME = "metadata_filtered_rag"
INDEX_NAME = "metadata_filtered_index"

# Retrieval configuration
DEFAULT_TOP_K = 5

# Available filter options (for reference)
AVAILABLE_TOPICS = [
    "insurance", "acquisitions", "investments", "management",
    "berkshire_operations", "market_commentary", "capital_allocation", "accounting"
]

AVAILABLE_COMPANIES = [
    "apple", "american_express", "bank_of_america", "coca_cola", "chevron",
    "occidental_petroleum", "moodys", "kraft_heinz", "chubb", "visa",
    "mastercard", "unitedhealth", "capital_one", "aon", "ally_financial",
    "sirius_xm", "verisign", "constellation_brands", "kroger", "dominos",
    "pool_corp", "louisiana_pacific", "dr_horton", "nucor",
    # Subsidiaries
    "geico", "bnsf", "berkshire hathaway energy", "precision castparts",
    "see's candies", "dairy queen", "duracell", "netjets"
]


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


def build_pre_filter(
    year: Optional[int | list[int]] = None,
    year_range: Optional[tuple[int, int]] = None,
    decade: Optional[int | list[int]] = None,
    source_file: Optional[str | list[str]] = None,
    topic_buckets: Optional[str | list[str]] = None,
    companies_mentioned: Optional[str | list[str]] = None,
    has_financials: Optional[bool] = None,
) -> dict:
    """
    Build a MongoDB pre-filter for vector search.
    
    Args:
        year: Single year or list of years to include
        year_range: Tuple of (start_year, end_year) inclusive
        decade: Single decade or list (2000, 2010, 2020)
        source_file: Single filename or list of filenames
        topic_buckets: Single topic or list of topics (OR matching)
        companies_mentioned: Single company or list of companies (OR matching)
        has_financials: Filter for chunks with financial figures
    
    Returns:
        MongoDB filter dictionary
    """
    conditions = []
    
    # Year filter
    if year is not None:
        if isinstance(year, list):
            conditions.append({"year": {"$in": year}})
        else:
            conditions.append({"year": {"$eq": year}})
    
    # Year range filter
    if year_range is not None:
        start_year, end_year = year_range
        conditions.append({
            "$and": [
                {"year": {"$gte": start_year}},
                {"year": {"$lte": end_year}}
            ]
        })
    
    # Decade filter
    if decade is not None:
        if isinstance(decade, list):
            conditions.append({"decade": {"$in": decade}})
        else:
            conditions.append({"decade": {"$eq": decade}})
    
    # Source file filter
    if source_file is not None:
        if isinstance(source_file, list):
            conditions.append({"source_file": {"$in": source_file}})
        else:
            conditions.append({"source_file": {"$eq": source_file}})
    
    # Topic buckets filter (OR matching - document has at least one topic)
    if topic_buckets is not None:
        if isinstance(topic_buckets, str):
            topic_buckets = [topic_buckets]
        conditions.append({"topic_buckets": {"$in": topic_buckets}})
    
    # Companies mentioned filter (OR matching)
    if companies_mentioned is not None:
        if isinstance(companies_mentioned, str):
            companies_mentioned = [companies_mentioned]
        conditions.append({"companies_mentioned": {"$in": companies_mentioned}})
    
    # Has financials filter
    if has_financials is not None:
        conditions.append({"has_financials": {"$eq": has_financials}})
    
    # Combine all conditions with AND
    if not conditions:
        return {}
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


def retrieve_with_filter(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    year: Optional[int | list[int]] = None,
    year_range: Optional[tuple[int, int]] = None,
    decade: Optional[int | list[int]] = None,
    source_file: Optional[str | list[str]] = None,
    topic_buckets: Optional[str | list[str]] = None,
    companies_mentioned: Optional[str | list[str]] = None,
    has_financials: Optional[bool] = None,
    verbose: bool = False,
) -> list:
    """
    Retrieve documents with metadata pre-filtering.
    
    Args:
        query: The search query string
        top_k: Number of documents to retrieve
        year: Filter by year(s)
        year_range: Filter by year range (start, end)
        decade: Filter by decade (2000, 2010, 2020)
        source_file: Filter by source file(s)
        topic_buckets: Filter by topic(s)
        companies_mentioned: Filter by company mentions
        has_financials: Filter for financial content
        verbose: Print filter details
    
    Returns:
        List of relevant document chunks
    """
    vector_store, client = get_vector_store()
    
    try:
        # Build pre-filter
        pre_filter = build_pre_filter(
            year=year,
            year_range=year_range,
            decade=decade,
            source_file=source_file,
            topic_buckets=topic_buckets,
            companies_mentioned=companies_mentioned,
            has_financials=has_financials
        )
        
        if verbose and pre_filter:
            print(f"   Pre-filter: {pre_filter}")
        
        # Perform filtered similarity search
        if pre_filter:
            results = vector_store.similarity_search(
                query=query,
                k=top_k,
                pre_filter=pre_filter
            )
        else:
            results = vector_store.similarity_search(
                query=query,
                k=top_k
            )
        
        return results
    finally:
        client.close()


def retrieve_with_filter_and_scores(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    year: Optional[int | list[int]] = None,
    year_range: Optional[tuple[int, int]] = None,
    decade: Optional[int | list[int]] = None,
    source_file: Optional[str | list[str]] = None,
    topic_buckets: Optional[str | list[str]] = None,
    companies_mentioned: Optional[str | list[str]] = None,
    has_financials: Optional[bool] = None,
) -> list:
    """
    Retrieve documents with metadata pre-filtering and similarity scores.
    
    Returns:
        List of tuples (document, score)
    """
    vector_store, client = get_vector_store()
    
    try:
        pre_filter = build_pre_filter(
            year=year,
            year_range=year_range,
            decade=decade,
            source_file=source_file,
            topic_buckets=topic_buckets,
            companies_mentioned=companies_mentioned,
            has_financials=has_financials
        )
        
        if pre_filter:
            results = vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                pre_filter=pre_filter
            )
        else:
            results = vector_store.similarity_search_with_score(
                query=query,
                k=top_k
            )
        
        return results
    finally:
        client.close()


def format_retrieved_context(documents: list) -> str:
    """Format retrieved documents into a context string for the LLM."""
    context_parts = []
    
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_file", "Unknown")
        year = doc.metadata.get("year", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        topics = doc.metadata.get("topic_buckets", [])
        companies = doc.metadata.get("companies_mentioned", [])
        
        context_parts.append(
            f"[Document {i}]\n"
            f"Source: {source} (Year: {year}, Page: {page})\n"
            f"Topics: {', '.join(topics) if topics else 'N/A'}\n"
            f"Companies: {', '.join(companies) if companies else 'N/A'}\n"
            f"Content:\n{doc.page_content}\n"
        )
    
    return "\n---\n".join(context_parts)


def get_available_years():
    """Get list of years available in the collection."""
    client = MongoClient(MONGO_DB_URL, **MONGO_KWARGS)
    collection = client[DB_NAME][COLLECTION_NAME]
    
    try:
        years = collection.distinct("year")
        return sorted([y for y in years if isinstance(y, int) and y > 0])
    finally:
        client.close()


def get_topic_counts():
    """Get counts of each topic bucket in the collection."""
    client = MongoClient(MONGO_DB_URL, **MONGO_KWARGS)
    collection = client[DB_NAME][COLLECTION_NAME]
    
    try:
        pipeline = [
            {"$unwind": "$topic_buckets"},
            {"$group": {"_id": "$topic_buckets", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        results = list(collection.aggregate(pipeline))
        return [(r["_id"], r["count"]) for r in results]
    finally:
        client.close()


def get_company_counts():
    """Get counts of company mentions in the collection."""
    client = MongoClient(MONGO_DB_URL, **MONGO_KWARGS)
    collection = client[DB_NAME][COLLECTION_NAME]
    
    try:
        pipeline = [
            {"$unwind": "$companies_mentioned"},
            {"$group": {"_id": "$companies_mentioned", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        results = list(collection.aggregate(pipeline))
        return [(r["_id"], r["count"]) for r in results]
    finally:
        client.close()


def debug_collection():
    """Debug function to check MongoDB collection status."""
    client = MongoClient(MONGO_DB_URL, **MONGO_KWARGS)
    collection = client[DB_NAME][COLLECTION_NAME]
    
    print("=" * 50)
    print("DEBUG: MongoDB Collection Status")
    print("=" * 50)
    
    doc_count = collection.count_documents({})
    print(f"\nTotal documents in collection: {doc_count}")
    
    if doc_count > 0:
        sample = collection.find_one()
        print(f"\nSample document fields: {list(sample.keys())}")
        print(f"\nSample metadata:")
        print(f"  source_file: {sample.get('source_file', 'N/A')}")
        print(f"  year: {sample.get('year', 'N/A')}")
        print(f"  decade: {sample.get('decade', 'N/A')}")
        print(f"  page: {sample.get('page', 'N/A')}")
        print(f"  has_financials: {sample.get('has_financials', 'N/A')}")
        print(f"  topic_buckets: {sample.get('topic_buckets', [])}")
        print(f"  companies_mentioned: {sample.get('companies_mentioned', [])}")
        
        # Show available years
        years = get_available_years()
        print(f"\nAvailable years: {years}")
        
        # Show topic counts
        print("\nTopic bucket counts:")
        for topic, count in get_topic_counts():
            print(f"  {topic}: {count}")
        
        # Show company counts
        print("\nTop 10 company mentions:")
        for company, count in get_company_counts()[:10]:
            print(f"  {company}: {count}")
    
    client.close()
    return doc_count


def main():
    """Test retrieval with various filters."""
    print("=" * 60)
    print("Metadata-Filtered RAG - Retrieval Test")
    print("=" * 60)
    
    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Debug collection first
    doc_count = debug_collection()
    
    if doc_count == 0:
        print("\n❌ No documents found! Run ingestion.py first.")
        return
    
    # Test queries with different filters
    print("\n" + "=" * 60)
    print("Running Filtered Retrieval Tests")
    print("=" * 60)
    
    # Test 1: No filter (baseline)
    test_query = "What are Buffett's views on acquisitions?"
    print("\n📝 Test 1: No filter (baseline)")
    print(f"   Query: {test_query}")
    results = retrieve_with_filter(test_query, top_k=3, verbose=True)
    print(f"   Retrieved: {len(results)} documents")
    for doc in results:
        print(f"   - {doc.metadata.get('source_file')} | Topics: {doc.metadata.get('topic_buckets', [])[:2]}")
    
    # Test 2: Filter by topic bucket
    print("\n📝 Test 2: Filter by topic_buckets=['insurance']")
    test_query = "How does Berkshire make money from insurance?"
    results = retrieve_with_filter(
        test_query, 
        top_k=3, 
        topic_buckets=["insurance"],
        verbose=True
    )
    print(f"   Retrieved: {len(results)} documents")
    for doc in results:
        print(f"   - {doc.metadata.get('source_file')} | Topics: {doc.metadata.get('topic_buckets', [])}")
    
    # Test 3: Filter by company
    print("\n📝 Test 3: Filter by companies_mentioned=['coca_cola', 'apple']")
    test_query = "What does Buffett think about consumer brands?"
    results = retrieve_with_filter(
        test_query,
        top_k=3,
        companies_mentioned=["coca_cola", "apple"],
        verbose=True
    )
    print(f"   Retrieved: {len(results)} documents")
    for doc in results:
        print(f"   - {doc.metadata.get('source_file')} | Companies: {doc.metadata.get('companies_mentioned', [])}")
    
    # Test 4: Filter by decade + financials
    print("\n📝 Test 4: Filter by decade=2010 + has_financials=True")
    test_query = "What were Berkshire's earnings?"
    results = retrieve_with_filter(
        test_query,
        top_k=3,
        decade=2010,
        has_financials=True,
        verbose=True
    )
    print(f"   Retrieved: {len(results)} documents")
    for doc in results:
        print(f"   - {doc.metadata.get('source_file')} | Year: {doc.metadata.get('year')}")
    
    # Test 5: Combined filters
    print("\n📝 Test 5: Combined filters (2020s + acquisitions topic)")
    test_query = "What companies did Berkshire acquire recently?"
    results = retrieve_with_filter(
        test_query,
        top_k=3,
        year_range=(2020, 2023),
        topic_buckets=["acquisitions"],
        verbose=True
    )
    print(f"   Retrieved: {len(results)} documents")
    for doc in results:
        print(f"   - {doc.metadata.get('source_file')} | Topics: {doc.metadata.get('topic_buckets', [])}")


if __name__ == "__main__":
    main()
