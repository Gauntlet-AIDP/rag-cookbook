"""
Agentic RAG - Tool Selection Evaluation
Tests whether the agent correctly chooses the appropriate retrieval tool.

The agent can choose from:
- no_retrieval: Skip retrieval for off-topic or trivial questions
- vector_search: Standard semantic search for general questions
- filtered_search: Metadata-filtered search for specific years/topics/companies

This eval measures how accurately the agent selects the right tool for different query types.
"""

import os
import sys
import json
from dotenv import load_dotenv

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # sandbox
    from llm_shim import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_MODEL = "gpt-4o-mini"

# Import the query analyzer prompt from agent.py
QUERY_ANALYZER_PROMPT = """Analyze this question and determine the best retrieval strategy.

Question: {question}

Analyze:
1. Is this a simple question that can be answered from general knowledge, or does it require specific information from Buffett's shareholder letters?
2. Is this a single question or a complex question that should be broken into sub-questions?
3. Does the question mention specific years, companies, or topics that could be used as filters?

Return a JSON object with this structure:
{{
    "needs_retrieval": true/false,
    "reasoning": "brief explanation of your decision",
    "is_complex": true/false,
    "sub_questions": ["sub-question 1", "sub-question 2"] or null if not complex,
    "suggested_tool": "no_retrieval" | "vector_search" | "filtered_search",
    "suggested_filters": {{
        "year": null or integer,
        "year_range": null or [start, end],
        "topic_buckets": null or ["topic1", "topic2"],
        "companies_mentioned": null or ["company1"],
        "has_financials": null or true/false
    }}
}}

Available topics: insurance, acquisitions, investments, management, berkshire_operations, market_commentary, capital_allocation, accounting
Available companies: apple, coca_cola, geico, american_express, bank_of_america, chevron, etc.

Return ONLY the JSON object, no other text."""


# =============================================================================
# Test Cases - Each has an expected tool and optional expected filters
# =============================================================================

TEST_CASES = [
    # NO_RETRIEVAL - Questions that don't need Buffett's letters
    {
        "id": "math_question",
        "question": "What is 2 + 2?",
        "expected_tool": "no_retrieval",
        "expected_needs_retrieval": False,
        "description": "Simple math - no retrieval needed"
    },
    {
        "id": "general_knowledge",
        "question": "What is the capital of France?",
        "expected_tool": "no_retrieval",
        "expected_needs_retrieval": False,
        "description": "General knowledge - no retrieval needed"
    },
    {
        "id": "greeting",
        "question": "Hello, how are you?",
        "expected_tool": "no_retrieval",
        "expected_needs_retrieval": False,
        "description": "Greeting - no retrieval needed"
    },
    
    # VECTOR_SEARCH - General questions about Buffett's philosophy
    {
        "id": "investment_philosophy",
        "question": "What is Warren Buffett's investment philosophy?",
        "expected_tool": "vector_search",
        "expected_needs_retrieval": True,
        "description": "General philosophy question - vector search"
    },
    {
        "id": "management_views",
        "question": "What does Buffett look for in company management?",
        "expected_tool": "vector_search",
        "expected_needs_retrieval": True,
        "description": "General views on management - vector search"
    },
    {
        "id": "value_investing",
        "question": "How does Buffett approach value investing?",
        "expected_tool": "vector_search",
        "expected_needs_retrieval": True,
        "description": "General investing approach - vector search"
    },
    
    # FILTERED_SEARCH - Questions with specific filters needed
    {
        "id": "specific_year",
        "question": "How did Berkshire perform in 2020?",
        "expected_tool": "filtered_search",
        "expected_needs_retrieval": True,
        "expected_filters": {"year": 2020},
        "description": "Year-specific question - filtered search"
    },
    {
        "id": "year_range",
        "question": "What acquisitions did Berkshire make between 2015 and 2020?",
        "expected_tool": "filtered_search",
        "expected_needs_retrieval": True,
        "expected_filters": {"year_range": [2015, 2020], "topic_buckets": ["acquisitions"]},
        "description": "Year range + topic - filtered search"
    },
    {
        "id": "company_specific",
        "question": "Why did Buffett invest in Apple?",
        "expected_tool": "filtered_search",
        "expected_needs_retrieval": True,
        "expected_filters": {"companies_mentioned": ["apple"]},
        "description": "Company-specific question - filtered search"
    },
    {
        "id": "topic_specific",
        "question": "How does Berkshire's insurance float work?",
        "expected_tool": "filtered_search",
        "expected_needs_retrieval": True,
        "expected_filters": {"topic_buckets": ["insurance"]},
        "description": "Topic-specific question - filtered search"
    },
    {
        "id": "company_and_year",
        "question": "How much did Apple contribute to Berkshire's portfolio in 2023?",
        "expected_tool": "filtered_search",
        "expected_needs_retrieval": True,
        "expected_filters": {"year": 2023, "companies_mentioned": ["apple"]},
        "description": "Company + year - filtered search"
    },
    {
        "id": "multiple_companies",
        "question": "Compare Buffett's views on Coca-Cola and American Express.",
        "expected_tool": "filtered_search",
        "expected_needs_retrieval": True,
        "expected_filters": {"companies_mentioned": ["coca_cola", "american_express"]},
        "description": "Multiple companies - filtered search"
    },
]


def analyze_query(question: str) -> dict:
    """Run the agent's query analyzer on a question."""
    prompt = ChatPromptTemplate.from_template(QUERY_ANALYZER_PROMPT)
    
    llm = ChatOpenAI(
        model=AGENT_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question})
    
    # Parse JSON response
    try:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        analysis = json.loads(response)
    except json.JSONDecodeError:
        analysis = {
            "needs_retrieval": True,
            "suggested_tool": "vector_search",
            "suggested_filters": {},
            "parse_error": True
        }
    
    return analysis


def evaluate_tool_selection(test_case: dict, verbose: bool = False) -> dict:
    """Evaluate tool selection for a single test case."""
    question = test_case["question"]
    expected_tool = test_case["expected_tool"]
    expected_needs_retrieval = test_case.get("expected_needs_retrieval", True)
    expected_filters = test_case.get("expected_filters", {})
    
    # Run the analyzer
    analysis = analyze_query(question)
    
    # Check tool selection
    actual_tool = analysis.get("suggested_tool", "unknown")
    tool_correct = actual_tool == expected_tool
    
    # Check needs_retrieval
    actual_needs_retrieval = analysis.get("needs_retrieval", True)
    needs_retrieval_correct = actual_needs_retrieval == expected_needs_retrieval
    
    # Check filters (partial matching - we check if expected filters are present)
    actual_filters = analysis.get("suggested_filters", {})
    filter_results = {}
    
    for filter_key, expected_value in expected_filters.items():
        actual_value = actual_filters.get(filter_key)
        
        if filter_key in ["topic_buckets", "companies_mentioned"]:
            # For lists, check if expected items are present (case-insensitive)
            if actual_value is None:
                filter_results[filter_key] = False
            else:
                expected_set = set(v.lower() for v in expected_value)
                actual_set = set(v.lower() for v in actual_value) if actual_value else set()
                filter_results[filter_key] = len(expected_set & actual_set) > 0
        elif filter_key == "year_range":
            # For year_range, check if it overlaps
            if actual_value is None:
                filter_results[filter_key] = False
            else:
                filter_results[filter_key] = (
                    actual_value[0] <= expected_value[1] and 
                    actual_value[1] >= expected_value[0]
                )
        else:
            filter_results[filter_key] = actual_value == expected_value
    
    filters_correct = all(filter_results.values()) if filter_results else True
    
    result = {
        "test_id": test_case["id"],
        "question": question,
        "description": test_case.get("description", ""),
        "expected_tool": expected_tool,
        "actual_tool": actual_tool,
        "tool_correct": tool_correct,
        "expected_needs_retrieval": expected_needs_retrieval,
        "actual_needs_retrieval": actual_needs_retrieval,
        "needs_retrieval_correct": needs_retrieval_correct,
        "expected_filters": expected_filters,
        "actual_filters": actual_filters,
        "filter_results": filter_results,
        "filters_correct": filters_correct,
        "overall_correct": tool_correct and needs_retrieval_correct and filters_correct,
    }
    
    if verbose:
        result["full_analysis"] = analysis
    
    return result


def run_evaluation(test_cases: list = None, verbose: bool = False) -> dict:
    """Run tool selection evaluation on all test cases."""
    if test_cases is None:
        test_cases = TEST_CASES
    cap = os.getenv("EVAL_MAX_CASES", "").strip()  # empty or unset runs every case
    if cap:
        if not cap.isdigit() or int(cap) <= 0:
            raise SystemExit(f"EVAL_MAX_CASES must be a positive integer, or empty for all cases (got {cap!r})")
        test_cases = test_cases[: int(cap)]
    
    print("=" * 60)
    print("Agentic RAG - Tool Selection Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Agent model: {AGENT_MODEL}")
    print(f"  - Test cases: {len(test_cases)}")
    print("\n" + "-" * 60)
    
    results = []
    tool_correct_count = 0
    retrieval_correct_count = 0
    filter_correct_count = 0
    overall_correct_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {test_case['id']}")
        print(f"    Q: {test_case['question'][:55]}...")
        print(f"    Expected: {test_case['expected_tool']}")
        
        result = evaluate_tool_selection(test_case, verbose=verbose)
        results.append(result)
        
        # Update counts
        if result["tool_correct"]:
            tool_correct_count += 1
        if result["needs_retrieval_correct"]:
            retrieval_correct_count += 1
        if result["filters_correct"]:
            filter_correct_count += 1
        if result["overall_correct"]:
            overall_correct_count += 1
        
        # Print result
        actual = result["actual_tool"]
        if result["overall_correct"]:
            print(f"    ✅ Correct: {actual}")
        elif result["tool_correct"]:
            print(f"    ⚠️  Tool correct ({actual}), but filter issues")
            if result["filter_results"]:
                for k, v in result["filter_results"].items():
                    status = "✓" if v else "✗"
                    print(f"        {status} {k}")
        else:
            print(f"    ❌ Wrong: got {actual}, expected {test_case['expected_tool']}")
    
    # Calculate metrics
    n = len(results)
    summary = {
        "num_test_cases": n,
        "tool_selection_accuracy": tool_correct_count / n if n > 0 else 0,
        "retrieval_decision_accuracy": retrieval_correct_count / n if n > 0 else 0,
        "filter_accuracy": filter_correct_count / n if n > 0 else 0,
        "overall_accuracy": overall_correct_count / n if n > 0 else 0,
        "tool_correct_count": tool_correct_count,
        "retrieval_correct_count": retrieval_correct_count,
        "filter_correct_count": filter_correct_count,
        "overall_correct_count": overall_correct_count,
        "results": results
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Aggregate Metrics:")
    print(f"   Tool Selection Accuracy:     {summary['tool_selection_accuracy']:.2%} ({tool_correct_count}/{n})")
    print(f"   Retrieval Decision Accuracy: {summary['retrieval_decision_accuracy']:.2%} ({retrieval_correct_count}/{n})")
    print(f"   Filter Extraction Accuracy:  {summary['filter_accuracy']:.2%} ({filter_correct_count}/{n})")
    print(f"   Overall Accuracy:            {summary['overall_accuracy']:.2%} ({overall_correct_count}/{n})")
    
    # Breakdown by expected tool type
    print(f"\n📋 By Expected Tool Type:")
    for tool_type in ["no_retrieval", "vector_search", "filtered_search"]:
        tool_results = [r for r in results if r["expected_tool"] == tool_type]
        if tool_results:
            correct = sum(1 for r in tool_results if r["tool_correct"])
            print(f"   {tool_type}: {correct}/{len(tool_results)} correct")
    
    # Show errors
    errors = [r for r in results if not r["overall_correct"]]
    if errors:
        print(f"\n⚠️  Errors ({len(errors)}):")
        for r in errors:
            print(f"   • {r['test_id']}: expected {r['expected_tool']}, got {r['actual_tool']}")
    
    return summary


def main():
    """Run the tool selection evaluation."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate agent's tool selection accuracy")
    parser.add_argument("-v", "--verbose", action="store_true", help="Include full analysis in output")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--question", type=str, help="Test a single custom question")
    
    args = parser.parse_args()
    
    if args.question:
        print(f"\nAnalyzing single question...")
        print(f"Q: {args.question}")
        print("-" * 50)
        
        analysis = analyze_query(args.question)
        
        print(f"\nNeeds retrieval: {analysis.get('needs_retrieval')}")
        print(f"Suggested tool: {analysis.get('suggested_tool')}")
        print(f"Is complex: {analysis.get('is_complex')}")
        print(f"Suggested filters: {json.dumps(analysis.get('suggested_filters', {}), indent=2)}")
        
        if analysis.get("sub_questions"):
            print(f"\nSub-questions:")
            for sq in analysis["sub_questions"]:
                print(f"  • {sq}")
    else:
        summary = run_evaluation(verbose=args.verbose)
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
