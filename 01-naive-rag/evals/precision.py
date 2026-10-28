"""
Naive RAG - Precision Evaluation
Measures retrieval precision using LLM-as-judge to assess document relevance.

Precision = (Number of relevant documents retrieved) / (Total documents retrieved, k)
"""

import os
import sys
import json
import certifi
from dotenv import load_dotenv

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

# MongoDB configuration (must match ingestion.py)
DB_NAME = "rag_playbook"
COLLECTION_NAME = "naive_rag"
INDEX_NAME = "naive"

# Evaluation configuration
DEFAULT_K = 5
JUDGE_MODEL = "gpt-4o-mini"

# LLM-as-Judge prompt for relevance assessment
RELEVANCE_JUDGE_PROMPT = """You are a relevance judge. Your task is to determine if the retrieved document is relevant to answering the given question.

A document is RELEVANT if it contains information that would help answer the question, even if it doesn't fully answer it.
A document is NOT RELEVANT if it contains no useful information for answering the question.

Question: {question}

Retrieved Document:
{document}

Is this document relevant to answering the question? 
Respond with ONLY "RELEVANT" or "NOT_RELEVANT" - nothing else."""


# Test cases for evaluation
# Each test case has a question and optionally expected topics that should appear
TEST_CASES = [
    {
        "id": "investment_philosophy",
        "question": "What is Warren Buffett's investment philosophy?",
        "description": "Core investment principles and value investing approach"
    },
    {
        "id": "market_volatility",
        "question": "How does Buffett view market volatility and stock price fluctuations?",
        "description": "Views on market timing and price swings"
    },
    {
        "id": "management_quality",
        "question": "What does Buffett look for in company management?",
        "description": "Criteria for evaluating business leaders"
    },
    {
        "id": "insurance_business",
        "question": "How does Berkshire's insurance business generate float?",
        "description": "Insurance operations and float concept"
    },
    {
        "id": "acquisition_criteria",
        "question": "What criteria does Berkshire use when acquiring companies?",
        "description": "M&A decision making process"
    },
    {
        "id": "derivatives",
        "question": "What are Buffett's views on derivatives and financial instruments?",
        "description": "Perspective on complex financial products"
    },
    {
        "id": "stock_buybacks",
        "question": "When does Buffett think stock buybacks make sense?",
        "description": "Share repurchase philosophy"
    },
    {
        "id": "long_term_investing",
        "question": "Why does Buffett prefer holding stocks for the long term?",
        "description": "Long-term investment horizon benefits"
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


def retrieve_documents(query: str, k: int = DEFAULT_K) -> list:
    """Retrieve top-k documents for a query."""
    vector_store, client = get_vector_store()
    
    try:
        results = vector_store.similarity_search(query=query, k=k)
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


def calculate_precision(question: str, k: int = DEFAULT_K, verbose: bool = False) -> dict:
    """
    Calculate precision for a single query.
    
    Precision = relevant_documents / k
    
    Returns dict with precision score and details.
    """
    # Retrieve documents
    documents = retrieve_documents(question, k=k)
    
    if not documents:
        return {
            "question": question,
            "k": k,
            "retrieved": 0,
            "relevant": 0,
            "precision": 0.0,
            "judgments": []
        }
    
    # Judge each document
    judge = create_relevance_judge()
    judgments = []
    relevant_count = 0
    
    for i, doc in enumerate(documents):
        is_relevant = judge_relevance(judge, question, doc.page_content)
        
        if is_relevant:
            relevant_count += 1
        
        judgment = {
            "doc_index": i + 1,
            "relevant": is_relevant,
            "source": doc.metadata.get("source_file", "Unknown"),
            "year": doc.metadata.get("year", "Unknown"),
        }
        
        if verbose:
            judgment["content_preview"] = doc.page_content[:200] + "..."
        
        judgments.append(judgment)
    
    precision = relevant_count / len(documents)
    
    return {
        "question": question,
        "k": k,
        "retrieved": len(documents),
        "relevant": relevant_count,
        "precision": precision,
        "judgments": judgments
    }


def run_evaluation(test_cases: list = None, k: int = DEFAULT_K, verbose: bool = False) -> dict:
    """
    Run precision evaluation on all test cases.
    
    Returns aggregate results and per-question breakdown.
    """
    if test_cases is None:
        test_cases = TEST_CASES
    cap = os.getenv("EVAL_MAX_CASES", "").strip()  # empty or unset runs every case
    if cap:
        if not cap.isdigit() or int(cap) <= 0:
            raise SystemExit(f"EVAL_MAX_CASES must be a positive integer, or empty for all cases (got {cap!r})")
        test_cases = test_cases[: int(cap)]
    
    print("=" * 60)
    print("Naive RAG - Precision Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - k (documents retrieved): {k}")
    print(f"  - Judge model: {JUDGE_MODEL}")
    print(f"  - Test cases: {len(test_cases)}")
    print("\n" + "-" * 60)
    
    results = []
    total_relevant = 0
    total_retrieved = 0
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        print(f"\n[{i}/{len(test_cases)}] {test_case['id']}")
        print(f"    Q: {question[:60]}...")
        
        result = calculate_precision(question, k=k, verbose=verbose)
        result["test_id"] = test_case["id"]
        result["description"] = test_case.get("description", "")
        
        results.append(result)
        total_relevant += result["relevant"]
        total_retrieved += result["retrieved"]
        
        print(f"    Precision: {result['precision']:.2%} ({result['relevant']}/{result['retrieved']} relevant)")
    
    # Calculate aggregate metrics
    avg_precision = sum(r["precision"] for r in results) / len(results) if results else 0
    overall_precision = total_relevant / total_retrieved if total_retrieved > 0 else 0
    
    summary = {
        "k": k,
        "num_test_cases": len(test_cases),
        "total_documents_retrieved": total_retrieved,
        "total_relevant_documents": total_relevant,
        "average_precision": avg_precision,
        "overall_precision": overall_precision,
        "results": results
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\n📊 Aggregate Metrics:")
    print(f"   Average Precision (per query):  {avg_precision:.2%}")
    print(f"   Overall Precision (all docs):   {overall_precision:.2%}")
    print(f"   Total Relevant / Total Retrieved: {total_relevant} / {total_retrieved}")
    
    print(f"\n📋 Per-Query Breakdown:")
    for r in results:
        status = "✅" if r["precision"] >= 0.6 else "⚠️" if r["precision"] >= 0.4 else "❌"
        print(f"   {status} {r['test_id']}: {r['precision']:.2%}")
    
    return summary


def main():
    """Run the precision evaluation."""
    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    import argparse
    parser = argparse.ArgumentParser(description="Run precision evaluation on Naive RAG")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"Number of documents to retrieve (default: {DEFAULT_K})")
    parser.add_argument("-v", "--verbose", action="store_true", help="Include document previews in output")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--question", type=str, help="Evaluate a single custom question")
    
    args = parser.parse_args()
    
    if args.question:
        # Single question evaluation
        print(f"\nEvaluating single question with k={args.k}")
        result = calculate_precision(args.question, k=args.k, verbose=True)
        
        print(f"\nQuestion: {result['question']}")
        print(f"Precision: {result['precision']:.2%} ({result['relevant']}/{result['retrieved']} relevant)")
        print("\nJudgments:")
        for j in result["judgments"]:
            status = "✅" if j["relevant"] else "❌"
            print(f"  {status} Doc {j['doc_index']}: {j['source']} (Year: {j['year']})")
            if args.verbose and "content_preview" in j:
                print(f"      {j['content_preview']}")
    else:
        # Full evaluation
        summary = run_evaluation(k=args.k, verbose=args.verbose)
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
