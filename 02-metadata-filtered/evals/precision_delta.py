"""
Metadata-Filtered RAG - Precision Delta Evaluation
Compares retrieval precision with and without metadata filters.

Measures the improvement in precision when using targeted filters
for topic-specific or company-specific questions.

Precision Delta = Filtered Precision - Unfiltered Precision
"""

import os
import sys
import json
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
JUDGE_MODEL = "gpt-4o-mini"

# LLM-as-Judge prompt for relevance assessment
RELEVANCE_JUDGE_PROMPT = """You are a relevance judge. Determine if the retrieved document is relevant to answering the given question.

A document is RELEVANT if it contains information that would help answer the question.
A document is NOT RELEVANT if it contains no useful information for the question.

Question: {question}

Retrieved Document:
{document}

Is this document relevant? Respond with ONLY "RELEVANT" or "NOT_RELEVANT"."""


# Test cases with questions that should benefit from filtering
# Each has a question and the filter that should improve precision
PRECISION_DELTA_TEST_CASES = [
    {
        "id": "insurance_topic",
        "question": "How does Berkshire's insurance float work and why is it valuable?",
        "filter_type": "topic",
        "filters": {"topic_buckets": ["insurance"]},
        "description": "Insurance-specific question with topic filter"
    },
    {
        "id": "acquisitions_topic",
        "question": "What criteria does Berkshire use when acquiring companies?",
        "filter_type": "topic",
        "filters": {"topic_buckets": ["acquisitions"]},
        "description": "Acquisitions question with topic filter"
    },
    {
        "id": "investments_topic",
        "question": "How does Buffett decide which stocks to buy?",
        "filter_type": "topic",
        "filters": {"topic_buckets": ["investments"]},
        "description": "Investment philosophy with topic filter"
    },
    {
        "id": "coca_cola_company",
        "question": "What does Buffett think about Coca-Cola as an investment?",
        "filter_type": "company",
        "filters": {"companies_mentioned": ["coca_cola"]},
        "description": "Coca-Cola specific question"
    },
    {
        "id": "apple_company",
        "question": "Why did Berkshire invest so heavily in Apple?",
        "filter_type": "company",
        "filters": {"companies_mentioned": ["apple"]},
        "description": "Apple specific question"
    },
    {
        "id": "geico_subsidiary",
        "question": "How has GEICO performed and what makes it special?",
        "filter_type": "company",
        "filters": {"companies_mentioned": ["geico"]},
        "description": "GEICO subsidiary question"
    },
    {
        "id": "financial_content",
        "question": "What were Berkshire's earnings and operating results?",
        "filter_type": "content",
        "filters": {"has_financials": True},
        "description": "Earnings question with financials filter"
    },
    {
        "id": "recent_years",
        "question": "What has Berkshire done in recent years?",
        "filter_type": "temporal",
        "filters": {"year_range": (2020, 2023)},
        "description": "Recent activity with year filter"
    },
    {
        "id": "combined_insurance_financial",
        "question": "What are the financial results of Berkshire's insurance operations?",
        "filter_type": "combined",
        "filters": {"topic_buckets": ["insurance"], "has_financials": True},
        "description": "Insurance + financials combined filter"
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


def retrieve_documents(query: str, k: int, filters: dict = None) -> list:
    """Retrieve documents with optional filters."""
    vector_store, client = get_vector_store()
    
    try:
        pre_filter = build_pre_filter(**filters) if filters else {}
        
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
        
        return results
    finally:
        client.close()


def create_relevance_judge():
    """Create the LLM judge for assessing relevance."""
    prompt = ChatPromptTemplate.from_template(RELEVANCE_JUDGE_PROMPT)
    
    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain


def judge_relevance(judge_chain, question: str, document_content: str) -> bool:
    """Use LLM to judge if a document is relevant to the question."""
    response = judge_chain.invoke({
        "question": question,
        "document": document_content
    })
    
    return "RELEVANT" in response.upper() and "NOT_RELEVANT" not in response.upper()


def calculate_precision(question: str, documents: list, judge) -> tuple[float, list[bool]]:
    """
    Calculate precision for a set of retrieved documents.
    
    Returns:
        Tuple of (precision_score, list_of_relevance_judgments)
    """
    if not documents:
        return 0.0, []
    
    judgments = []
    for doc in documents:
        is_relevant = judge_relevance(judge, question, doc.page_content)
        judgments.append(is_relevant)
    
    precision = sum(judgments) / len(judgments)
    return precision, judgments


def evaluate_precision_delta(
    question: str,
    filters: dict,
    k: int = DEFAULT_K,
    verbose: bool = False
) -> dict:
    """
    Compare precision with and without filters for a single question.
    
    Returns:
        Dictionary with precision comparison results
    """
    judge = create_relevance_judge()
    
    # Retrieve WITHOUT filters (baseline)
    unfiltered_docs = retrieve_documents(question, k=k, filters=None)
    unfiltered_precision, unfiltered_judgments = calculate_precision(
        question, unfiltered_docs, judge
    )
    
    # Retrieve WITH filters
    filtered_docs = retrieve_documents(question, k=k, filters=filters)
    filtered_precision, filtered_judgments = calculate_precision(
        question, filtered_docs, judge
    )
    
    # Calculate delta
    precision_delta = filtered_precision - unfiltered_precision
    
    result = {
        "question": question,
        "filters": filters,
        "k": k,
        "unfiltered": {
            "precision": unfiltered_precision,
            "relevant_count": sum(unfiltered_judgments),
            "total_docs": len(unfiltered_docs),
            "judgments": unfiltered_judgments
        },
        "filtered": {
            "precision": filtered_precision,
            "relevant_count": sum(filtered_judgments),
            "total_docs": len(filtered_docs),
            "judgments": filtered_judgments
        },
        "precision_delta": precision_delta,
        "improvement": precision_delta > 0
    }
    
    if verbose:
        result["unfiltered_sources"] = [
            doc.metadata.get("source_file", "Unknown") 
            for doc in unfiltered_docs
        ]
        result["filtered_sources"] = [
            doc.metadata.get("source_file", "Unknown") 
            for doc in filtered_docs
        ]
    
    return result


def run_evaluation(
    test_cases: list = None,
    k: int = DEFAULT_K,
    verbose: bool = False
) -> dict:
    """
    Run precision delta evaluation on all test cases.
    
    Returns:
        Dictionary with aggregate results
    """
    if test_cases is None:
        test_cases = PRECISION_DELTA_TEST_CASES
    cap = os.getenv("EVAL_MAX_CASES", "").strip()  # empty or unset runs every case
    if cap:
        if not cap.isdigit() or int(cap) <= 0:
            raise SystemExit(f"EVAL_MAX_CASES must be a positive integer, or empty for all cases (got {cap!r})")
        test_cases = test_cases[: int(cap)]
    
    print("=" * 60)
    print("Metadata-Filtered RAG - Precision Delta Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - k (documents): {k}")
    print(f"  - Judge model: {JUDGE_MODEL}")
    print(f"  - Test cases: {len(test_cases)}")
    print("\n" + "-" * 60)
    
    results = []
    improvements = 0
    no_change = 0
    regressions = 0
    
    for i, test_case in enumerate(test_cases, 1):
        test_id = test_case["id"]
        question = test_case["question"]
        filters = test_case["filters"]
        description = test_case["description"]
        filter_type = test_case["filter_type"]
        
        print(f"\n[{i}/{len(test_cases)}] {test_id}")
        print(f"    Type: {filter_type}")
        print(f"    Q: {question[:55]}...")
        
        result = evaluate_precision_delta(
            question=question,
            filters=filters,
            k=k,
            verbose=verbose
        )
        result["test_id"] = test_id
        result["description"] = description
        result["filter_type"] = filter_type
        
        results.append(result)
        
        # Track outcomes
        delta = result["precision_delta"]
        if delta > 0:
            improvements += 1
            status = f"🟢 +{delta:.0%}"
        elif delta < 0:
            regressions += 1
            status = f"🔴 {delta:.0%}"
        else:
            no_change += 1
            status = "⚪ 0%"
        
        print(f"    Unfiltered: {result['unfiltered']['precision']:.0%} ({result['unfiltered']['relevant_count']}/{result['unfiltered']['total_docs']})")
        print(f"    Filtered:   {result['filtered']['precision']:.0%} ({result['filtered']['relevant_count']}/{result['filtered']['total_docs']})")
        print(f"    Delta:      {status}")
    
    # Calculate aggregate metrics
    avg_unfiltered = sum(r["unfiltered"]["precision"] for r in results) / len(results)
    avg_filtered = sum(r["filtered"]["precision"] for r in results) / len(results)
    avg_delta = sum(r["precision_delta"] for r in results) / len(results)
    
    # Group by filter type
    by_filter_type = {}
    for r in results:
        ft = r["filter_type"]
        if ft not in by_filter_type:
            by_filter_type[ft] = []
        by_filter_type[ft].append(r["precision_delta"])
    
    summary = {
        "k": k,
        "num_test_cases": len(test_cases),
        "avg_unfiltered_precision": avg_unfiltered,
        "avg_filtered_precision": avg_filtered,
        "avg_precision_delta": avg_delta,
        "improvements": improvements,
        "no_change": no_change,
        "regressions": regressions,
        "by_filter_type": {
            ft: sum(deltas) / len(deltas) 
            for ft, deltas in by_filter_type.items()
        },
        "results": results
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Aggregate Metrics:")
    print(f"   Average Unfiltered Precision: {avg_unfiltered:.2%}")
    print(f"   Average Filtered Precision:   {avg_filtered:.2%}")
    print(f"   Average Precision Delta:      {avg_delta:+.2%}")
    
    print(f"\n📈 Outcomes:")
    print(f"   🟢 Improvements: {improvements}/{len(results)} ({100*improvements/len(results):.0f}%)")
    print(f"   ⚪ No Change:    {no_change}/{len(results)} ({100*no_change/len(results):.0f}%)")
    print(f"   🔴 Regressions:  {regressions}/{len(results)} ({100*regressions/len(results):.0f}%)")
    
    print(f"\n📋 By Filter Type:")
    for ft, avg in summary["by_filter_type"].items():
        indicator = "🟢" if avg > 0 else "🔴" if avg < 0 else "⚪"
        print(f"   {indicator} {ft}: {avg:+.2%} avg delta")
    
    print(f"\n📋 Per-Query Breakdown:")
    for r in results:
        delta = r["precision_delta"]
        if delta > 0:
            indicator = "🟢"
        elif delta < 0:
            indicator = "🔴"
        else:
            indicator = "⚪"
        print(f"   {indicator} {r['test_id']}: {r['unfiltered']['precision']:.0%} → {r['filtered']['precision']:.0%} ({delta:+.0%})")
    
    return summary


def main():
    """Run the precision delta evaluation."""
    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    import argparse
    parser = argparse.ArgumentParser(description="Run precision delta evaluation")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"Number of documents to retrieve (default: {DEFAULT_K})")
    parser.add_argument("-v", "--verbose", action="store_true", help="Include document sources in output")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    summary = run_evaluation(k=args.k, verbose=args.verbose)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
