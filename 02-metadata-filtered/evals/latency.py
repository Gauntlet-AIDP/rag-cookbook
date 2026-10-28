"""
Metadata-Filtered RAG - Latency Evaluation
Compares retrieval and generation latency with different filter configurations.

Measures:
- Retrieval latency (vector search time)
- Generation latency (LLM response time)  
- Total end-to-end latency
- Comparison: filtered vs unfiltered
"""

import os
import sys
import json
import time
import statistics
import certifi
from dotenv import load_dotenv
from typing import Optional

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:  # sandbox
    from llm_shim import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
COLLECTION_NAME = "metadata_filtered_rag"
INDEX_NAME = "metadata_filtered_index"

# Evaluation configuration
DEFAULT_K = 5
DEFAULT_RUNS = 3  # Number of runs per test for averaging
GENERATION_MODEL = "gpt-4o-mini"

# RAG Prompt Template
RAG_PROMPT = """Answer based on the context below.

Context:
{context}

Question: {question}

Answer:"""


# Test cases with different filter configurations
LATENCY_TEST_CASES = [
    {
        "id": "no_filter",
        "question": "What is Buffett's investment philosophy?",
        "filters": {},
        "description": "No filters (baseline)"
    },
    {
        "id": "year_filter",
        "question": "What were Berkshire's results?",
        "filters": {"year": 2020},
        "description": "Single year filter"
    },
    {
        "id": "year_range_filter",
        "question": "How did Berkshire perform?",
        "filters": {"year_range": (2018, 2023)},
        "description": "Year range filter"
    },
    {
        "id": "decade_filter",
        "question": "What acquisitions were made?",
        "filters": {"decade": 2010},
        "description": "Decade filter"
    },
    {
        "id": "topic_filter",
        "question": "How does insurance float work?",
        "filters": {"topic_buckets": ["insurance"]},
        "description": "Topic bucket filter"
    },
    {
        "id": "company_filter",
        "question": "What does Buffett think about this investment?",
        "filters": {"companies_mentioned": ["coca_cola"]},
        "description": "Company mention filter"
    },
    {
        "id": "financials_filter",
        "question": "What were the earnings?",
        "filters": {"has_financials": True},
        "description": "Financial content filter"
    },
    {
        "id": "combined_filter",
        "question": "What recent acquisitions were made?",
        "filters": {
            "year_range": (2020, 2023),
            "topic_buckets": ["acquisitions"]
        },
        "description": "Combined filters"
    },
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
    year: Optional[int] = None,
    year_range: Optional[tuple[int, int]] = None,
    decade: Optional[int] = None,
    topic_buckets: Optional[list[str]] = None,
    companies_mentioned: Optional[list[str]] = None,
    has_financials: Optional[bool] = None,
) -> dict:
    """Build a MongoDB pre-filter for vector search."""
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


def measure_retrieval_latency(
    query: str,
    k: int = DEFAULT_K,
    filters: dict = None
) -> tuple[list, float]:
    """
    Measure retrieval latency and return documents.
    
    Returns:
        Tuple of (documents, latency_ms)
    """
    vector_store, client = get_vector_store()
    
    try:
        pre_filter = build_pre_filter(**filters) if filters else {}
        
        start_time = time.perf_counter()
        
        if pre_filter:
            results = vector_store.similarity_search(
                query=query,
                k=k,
                pre_filter=pre_filter
            )
        else:
            results = vector_store.similarity_search(
                query=query,
                k=k
            )
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        return results, latency_ms
    finally:
        client.close()


def measure_generation_latency(question: str, context: str) -> tuple[str, float]:
    """
    Measure generation latency and return answer.
    
    Returns:
        Tuple of (answer, latency_ms)
    """
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    llm = ChatOpenAI(
        model=GENERATION_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )
    chain = prompt | llm | StrOutputParser()
    
    start_time = time.perf_counter()
    
    answer = chain.invoke({
        "context": context,
        "question": question
    })
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    
    return answer, latency_ms


def format_context(documents: list) -> str:
    """Format documents into context string."""
    parts = []
    for i, doc in enumerate(documents, 1):
        parts.append(f"[{i}] {doc.page_content[:500]}")
    return "\n\n".join(parts)


def run_single_test(
    question: str,
    filters: dict,
    k: int = DEFAULT_K,
    include_generation: bool = True
) -> dict:
    """
    Run a single latency test.
    
    Returns:
        Dictionary with latency measurements
    """
    # Measure retrieval
    documents, retrieval_latency = measure_retrieval_latency(
        query=question,
        k=k,
        filters=filters if filters else None
    )
    
    result = {
        "retrieval_latency_ms": retrieval_latency,
        "documents_retrieved": len(documents),
    }
    
    # Measure generation if requested
    if include_generation and documents:
        context = format_context(documents)
        answer, generation_latency = measure_generation_latency(question, context)
        
        result["generation_latency_ms"] = generation_latency
        result["total_latency_ms"] = retrieval_latency + generation_latency
    else:
        result["generation_latency_ms"] = 0
        result["total_latency_ms"] = retrieval_latency
    
    return result


def run_latency_evaluation(
    test_cases: list = None,
    k: int = DEFAULT_K,
    num_runs: int = DEFAULT_RUNS,
    include_generation: bool = True
) -> dict:
    """
    Run latency evaluation across all test cases.
    
    Args:
        test_cases: List of test cases to run
        k: Number of documents to retrieve
        num_runs: Number of runs per test for averaging
        include_generation: Whether to measure generation latency
    
    Returns:
        Dictionary with evaluation results
    """
    if test_cases is None:
        test_cases = LATENCY_TEST_CASES
    cap = os.getenv("EVAL_MAX_CASES", "").strip()  # empty or unset runs every case
    if cap:
        if not cap.isdigit() or int(cap) <= 0:
            raise SystemExit(f"EVAL_MAX_CASES must be a positive integer, or empty for all cases (got {cap!r})")
        test_cases = test_cases[: int(cap)]
    
    print("=" * 60)
    print("Metadata-Filtered RAG - Latency Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - k (documents): {k}")
    print(f"  - Runs per test: {num_runs}")
    print(f"  - Include generation: {include_generation}")
    print(f"  - Test cases: {len(test_cases)}")
    print("\n" + "-" * 60)
    
    results = []
    
    for test_case in test_cases:
        test_id = test_case["id"]
        question = test_case["question"]
        filters = test_case["filters"]
        description = test_case["description"]
        
        print(f"\n📝 {test_id}: {description}")
        
        # Run multiple times and collect measurements
        retrieval_times = []
        generation_times = []
        total_times = []
        docs_retrieved = []
        
        for run in range(num_runs):
            run_result = run_single_test(
                question=question,
                filters=filters,
                k=k,
                include_generation=include_generation
            )
            
            retrieval_times.append(run_result["retrieval_latency_ms"])
            generation_times.append(run_result["generation_latency_ms"])
            total_times.append(run_result["total_latency_ms"])
            docs_retrieved.append(run_result["documents_retrieved"])
        
        # Calculate statistics
        result = {
            "test_id": test_id,
            "description": description,
            "filters": filters,
            "num_runs": num_runs,
            "documents_retrieved": docs_retrieved[0],
            "retrieval": {
                "mean_ms": statistics.mean(retrieval_times),
                "median_ms": statistics.median(retrieval_times),
                "min_ms": min(retrieval_times),
                "max_ms": max(retrieval_times),
                "stdev_ms": statistics.stdev(retrieval_times) if num_runs > 1 else 0
            },
            "generation": {
                "mean_ms": statistics.mean(generation_times),
                "median_ms": statistics.median(generation_times),
                "min_ms": min(generation_times),
                "max_ms": max(generation_times),
            },
            "total": {
                "mean_ms": statistics.mean(total_times),
                "median_ms": statistics.median(total_times),
            }
        }
        
        results.append(result)
        
        # Print summary for this test
        print(f"   Retrieval: {result['retrieval']['mean_ms']:.1f}ms (±{result['retrieval']['stdev_ms']:.1f}ms)")
        if include_generation:
            print(f"   Generation: {result['generation']['mean_ms']:.1f}ms")
            print(f"   Total: {result['total']['mean_ms']:.1f}ms")
        print(f"   Docs retrieved: {result['documents_retrieved']}")
    
    # Calculate summary statistics
    baseline = next((r for r in results if r["test_id"] == "no_filter"), results[0])
    baseline_retrieval = baseline["retrieval"]["mean_ms"]
    
    summary = {
        "k": k,
        "num_runs": num_runs,
        "baseline_retrieval_ms": baseline_retrieval,
        "results": results
    }
    
    # Print comparison
    print("\n" + "=" * 60)
    print("LATENCY COMPARISON")
    print("=" * 60)
    print(f"\n{'Test Case':<25} {'Retrieval':<15} {'vs Baseline':<15} {'Docs':<10}")
    print("-" * 65)
    
    for r in results:
        retrieval = r["retrieval"]["mean_ms"]
        diff = retrieval - baseline_retrieval
        diff_pct = (diff / baseline_retrieval * 100) if baseline_retrieval > 0 else 0
        
        if diff < 0:
            diff_str = f"🟢 {diff:+.1f}ms ({diff_pct:+.0f}%)"
        elif diff > 0:
            diff_str = f"🔴 {diff:+.1f}ms ({diff_pct:+.0f}%)"
        else:
            diff_str = "baseline"
        
        print(f"{r['test_id']:<25} {retrieval:<15.1f} {diff_str:<15} {r['documents_retrieved']:<10}")
    
    if include_generation:
        print(f"\n📊 End-to-End Latency Summary:")
        for r in results:
            print(f"   {r['test_id']}: {r['total']['mean_ms']:.0f}ms total")
    
    return summary


def main():
    """Run the latency evaluation."""
    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    import argparse
    parser = argparse.ArgumentParser(description="Run latency evaluation on Metadata-Filtered RAG")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"Number of documents to retrieve (default: {DEFAULT_K})")
    parser.add_argument("-n", "--num-runs", type=int, default=DEFAULT_RUNS, help=f"Number of runs per test (default: {DEFAULT_RUNS})")
    parser.add_argument("--no-generation", action="store_true", help="Skip generation latency measurement")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    summary = run_latency_evaluation(
        k=args.k,
        num_runs=args.num_runs,
        include_generation=not args.no_generation
    )
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
