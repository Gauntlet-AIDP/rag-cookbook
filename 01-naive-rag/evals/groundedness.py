"""
Naive RAG - Groundedness Evaluation
Measures whether LLM-generated answers are supported by the retrieved context.

Groundedness assesses if the answer is faithful to the source documents,
detecting hallucinations or unsupported claims.
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
GENERATION_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"

# RAG Generation Prompt (same as generation.py)
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context from Warren Buffett's annual shareholder letters.

Use ONLY the information from the context below to answer the question. If the context doesn't contain enough information to fully answer the question, acknowledge what you can answer and what information is missing.

Context:
{context}

Question: {question}

Answer:"""

# LLM-as-Judge prompt for groundedness assessment
GROUNDEDNESS_JUDGE_PROMPT = """You are a groundedness evaluator. Your task is to determine if the given answer is fully supported by the provided context.

An answer is GROUNDED if:
- Every claim and statement in the answer can be traced back to information in the context
- The answer does not include information that isn't present in the context
- The answer does not make assumptions or inferences beyond what the context supports

An answer is NOT GROUNDED if:
- It contains claims not supported by the context (hallucinations)
- It adds information or details not present in the context
- It makes unsupported generalizations or conclusions

Context:
{context}

Question: {question}

Answer to evaluate:
{answer}

Evaluate the groundedness of this answer. First, briefly explain your reasoning (2-3 sentences), then provide your verdict.

Respond in this exact format:
REASONING: <your brief explanation>
VERDICT: <GROUNDED or NOT_GROUNDED>"""


# Test cases for groundedness evaluation
TEST_CASES = [
    {
        "id": "investment_philosophy",
        "question": "What is Warren Buffett's investment philosophy?",
        "description": "Core investment principles - should be well-grounded"
    },
    {
        "id": "market_timing",
        "question": "Does Buffett believe in timing the market?",
        "description": "Views on market timing"
    },
    {
        "id": "good_management",
        "question": "What qualities does Buffett look for in managers of companies he invests in?",
        "description": "Management evaluation criteria"
    },
    {
        "id": "insurance_float",
        "question": "How does Berkshire Hathaway use insurance float to generate returns?",
        "description": "Insurance business model explanation"
    },
    {
        "id": "mistakes",
        "question": "What investment mistakes has Buffett admitted to making?",
        "description": "Self-admitted errors - tests honesty about limitations"
    },
    {
        "id": "tech_investments",
        "question": "What is Buffett's approach to investing in technology companies?",
        "description": "Tech investment philosophy"
    },
    {
        "id": "succession",
        "question": "What are Buffett's plans for Berkshire's succession?",
        "description": "Leadership transition plans"
    },
    {
        "id": "dividends",
        "question": "Why doesn't Berkshire Hathaway pay dividends?",
        "description": "Capital allocation philosophy"
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


def format_context(documents: list) -> str:
    """Format retrieved documents into context string."""
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


def generate_answer(question: str, context: str) -> str:
    """Generate an answer using the RAG pipeline."""
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    llm = ChatOpenAI(
        model=GENERATION_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )
    
    chain = prompt | llm | StrOutputParser()
    
    answer = chain.invoke({
        "context": context,
        "question": question
    })
    
    return answer


def create_groundedness_judge():
    """Create the LLM judge for assessing groundedness."""
    prompt = ChatPromptTemplate.from_template(GROUNDEDNESS_JUDGE_PROMPT)
    
    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain


def judge_groundedness(judge_chain, question: str, context: str, answer: str) -> dict:
    """Use LLM to judge if an answer is grounded in the context."""
    response = judge_chain.invoke({
        "question": question,
        "context": context,
        "answer": answer
    })
    
    # Parse the response
    reasoning = ""
    is_grounded = False
    
    if "REASONING:" in response:
        reasoning_start = response.find("REASONING:") + len("REASONING:")
        reasoning_end = response.find("VERDICT:")
        if reasoning_end > reasoning_start:
            reasoning = response[reasoning_start:reasoning_end].strip()
    
    is_grounded = "VERDICT: GROUNDED" in response.upper() and "NOT_GROUNDED" not in response.upper()
    
    return {
        "is_grounded": is_grounded,
        "reasoning": reasoning,
        "raw_response": response
    }


def evaluate_groundedness(question: str, k: int = DEFAULT_K, verbose: bool = False) -> dict:
    """
    Evaluate groundedness for a single question.
    
    Returns dict with groundedness assessment and details.
    """
    # Step 1: Retrieve documents
    documents = retrieve_documents(question, k=k)
    
    if not documents:
        return {
            "question": question,
            "k": k,
            "retrieved": 0,
            "is_grounded": False,
            "reasoning": "No documents retrieved",
            "answer": "",
            "context": ""
        }
    
    # Step 2: Format context
    context = format_context(documents)
    
    # Step 3: Generate answer
    answer = generate_answer(question, context)
    
    # Step 4: Judge groundedness
    judge = create_groundedness_judge()
    judgment = judge_groundedness(judge, question, context, answer)
    
    result = {
        "question": question,
        "k": k,
        "retrieved": len(documents),
        "is_grounded": judgment["is_grounded"],
        "reasoning": judgment["reasoning"],
        "answer": answer,
        "sources": [
            {
                "file": doc.metadata.get("source_file", "Unknown"),
                "year": doc.metadata.get("year", "Unknown"),
            }
            for doc in documents
        ]
    }
    
    if verbose:
        result["context"] = context
        result["raw_judgment"] = judgment["raw_response"]
    
    return result


def run_evaluation(test_cases: list = None, k: int = DEFAULT_K, verbose: bool = False) -> dict:
    """
    Run groundedness evaluation on all test cases.
    
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
    print("Naive RAG - Groundedness Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - k (documents retrieved): {k}")
    print(f"  - Generation model: {GENERATION_MODEL}")
    print(f"  - Judge model: {JUDGE_MODEL}")
    print(f"  - Test cases: {len(test_cases)}")
    print("\n" + "-" * 60)
    
    results = []
    grounded_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        print(f"\n[{i}/{len(test_cases)}] {test_case['id']}")
        print(f"    Q: {question[:60]}...")
        
        result = evaluate_groundedness(question, k=k, verbose=verbose)
        result["test_id"] = test_case["id"]
        result["description"] = test_case.get("description", "")
        
        results.append(result)
        
        if result["is_grounded"]:
            grounded_count += 1
            print(f"    ✅ GROUNDED")
        else:
            print(f"    ❌ NOT GROUNDED")
        
        if result["reasoning"]:
            print(f"    Reason: {result['reasoning'][:80]}...")
    
    # Calculate aggregate metrics
    groundedness_rate = grounded_count / len(results) if results else 0
    
    summary = {
        "k": k,
        "num_test_cases": len(test_cases),
        "grounded_count": grounded_count,
        "not_grounded_count": len(results) - grounded_count,
        "groundedness_rate": groundedness_rate,
        "results": results
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\n📊 Aggregate Metrics:")
    print(f"   Groundedness Rate: {groundedness_rate:.2%}")
    print(f"   Grounded: {grounded_count} / {len(results)}")
    print(f"   Not Grounded: {len(results) - grounded_count} / {len(results)}")
    
    print(f"\n📋 Per-Query Breakdown:")
    for r in results:
        status = "✅" if r["is_grounded"] else "❌"
        print(f"   {status} {r['test_id']}")
    
    # Show any not grounded cases with reasoning
    not_grounded = [r for r in results if not r["is_grounded"]]
    if not_grounded:
        print(f"\n⚠️  Not Grounded Cases (potential hallucinations):")
        for r in not_grounded:
            print(f"\n   {r['test_id']}:")
            print(f"   Q: {r['question'][:60]}...")
            print(f"   Reason: {r['reasoning']}")
    
    return summary


def main():
    """Run the groundedness evaluation."""
    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    import argparse
    parser = argparse.ArgumentParser(description="Run groundedness evaluation on Naive RAG")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"Number of documents to retrieve (default: {DEFAULT_K})")
    parser.add_argument("-v", "--verbose", action="store_true", help="Include full context and judgments in output")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--question", type=str, help="Evaluate a single custom question")
    
    args = parser.parse_args()
    
    if args.question:
        # Single question evaluation
        print(f"\nEvaluating groundedness for single question with k={args.k}")
        print("-" * 60)
        
        result = evaluate_groundedness(args.question, k=args.k, verbose=True)
        
        print(f"\nQuestion: {result['question']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\n{'✅ GROUNDED' if result['is_grounded'] else '❌ NOT GROUNDED'}")
        print(f"Reasoning: {result['reasoning']}")
        
        print(f"\nSources used:")
        for s in result["sources"]:
            print(f"  • {s['file']} (Year: {s['year']})")
    else:
        # Full evaluation
        summary = run_evaluation(k=args.k, verbose=args.verbose)
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
