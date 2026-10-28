"""
Agentic RAG - Query Decomposition Evaluation
Tests quality of breaking down complex questions into sub-questions.

For complex questions, the agent should:
1. Recognize the question is complex (is_complex = true)
2. Generate appropriate sub-questions that cover all aspects
3. Each sub-question should be specific and answerable independently

This eval measures:
- Decomposition detection (did it recognize complexity?)
- Coverage (do sub-questions address all parts of the original?)
- Quality (are sub-questions well-formed?)
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
JUDGE_MODEL = "gpt-4o-mini"

# Query analyzer prompt (same as agent.py)
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


# LLM-as-Judge prompt for evaluating sub-question quality
DECOMPOSITION_JUDGE_PROMPT = """You are evaluating the quality of query decomposition for a RAG system.

Original Complex Question:
{original_question}

Expected Aspects to Cover:
{expected_aspects}

Generated Sub-questions:
{sub_questions}

Evaluate the decomposition on these criteria:

1. COVERAGE: Do the sub-questions collectively address all aspects of the original question?
   - All expected aspects should be covered
   - No major parts of the original question should be missing

2. SPECIFICITY: Are the sub-questions specific enough to retrieve relevant information?
   - Each sub-question should be focused and clear
   - Vague or overly broad sub-questions are not good

3. INDEPENDENCE: Can each sub-question be answered independently?
   - Sub-questions shouldn't require answers from other sub-questions
   - Each should make sense on its own

4. RELEVANCE: Are the sub-questions relevant to the original question?
   - No tangential or off-topic sub-questions

Rate each criterion from 1-5 (1=poor, 5=excellent) and provide brief reasoning.

Return a JSON object:
{{
    "coverage_score": 1-5,
    "coverage_reasoning": "brief explanation",
    "specificity_score": 1-5,
    "specificity_reasoning": "brief explanation",
    "independence_score": 1-5,
    "independence_reasoning": "brief explanation",
    "relevance_score": 1-5,
    "relevance_reasoning": "brief explanation",
    "overall_quality": "GOOD" | "ACCEPTABLE" | "POOR",
    "missing_aspects": ["aspect 1", "aspect 2"] or []
}}

Return ONLY the JSON object."""


# =============================================================================
# Test Cases - Complex questions that should be decomposed
# =============================================================================

TEST_CASES = [
    # Temporal comparison questions
    {
        "id": "year_comparison",
        "question": "How did Berkshire's insurance business compare between 2008 and 2020?",
        "should_decompose": True,
        "expected_aspects": [
            "Insurance performance in 2008",
            "Insurance performance in 2020",
            "Comparison or changes between the periods"
        ],
        "min_sub_questions": 2,
        "description": "Temporal comparison - should split by year"
    },
    {
        "id": "decade_comparison",
        "question": "How did Buffett's investment strategy evolve from the 2000s to the 2020s?",
        "should_decompose": True,
        "expected_aspects": [
            "Investment strategy in 2000s",
            "Investment strategy in 2020s",
            "Evolution or changes over time"
        ],
        "min_sub_questions": 2,
        "description": "Decade comparison - should split by era"
    },
    
    # Multi-entity questions
    {
        "id": "multi_company",
        "question": "What are Buffett's views on Apple, Coca-Cola, and American Express as investments?",
        "should_decompose": True,
        "expected_aspects": [
            "Views on Apple",
            "Views on Coca-Cola",
            "Views on American Express"
        ],
        "min_sub_questions": 2,
        "description": "Multiple companies - should split by company"
    },
    {
        "id": "two_topics",
        "question": "How does Berkshire approach both insurance underwriting and stock investments?",
        "should_decompose": True,
        "expected_aspects": [
            "Insurance underwriting approach",
            "Stock investment approach"
        ],
        "min_sub_questions": 2,
        "description": "Two topics - should split by topic"
    },
    
    # Multi-faceted questions
    {
        "id": "acquisition_analysis",
        "question": "What companies has Berkshire acquired, why were they acquired, and how have they performed?",
        "should_decompose": True,
        "expected_aspects": [
            "List of acquired companies",
            "Reasons for acquisitions",
            "Performance of acquisitions"
        ],
        "min_sub_questions": 2,
        "description": "Multi-faceted - what, why, and how"
    },
    {
        "id": "float_comprehensive",
        "question": "What is insurance float, how does Berkshire generate it, and how is it invested?",
        "should_decompose": True,
        "expected_aspects": [
            "Definition of insurance float",
            "How Berkshire generates float",
            "How float is invested"
        ],
        "min_sub_questions": 2,
        "description": "Multi-part question about float"
    },
    
    # Simple questions (should NOT decompose)
    {
        "id": "simple_definition",
        "question": "What is insurance float?",
        "should_decompose": False,
        "expected_aspects": ["Definition of insurance float"],
        "min_sub_questions": 0,
        "description": "Simple definition - no decomposition needed"
    },
    {
        "id": "simple_philosophy",
        "question": "What is Buffett's investment philosophy?",
        "should_decompose": False,
        "expected_aspects": ["Investment philosophy"],
        "min_sub_questions": 0,
        "description": "Simple question - no decomposition needed"
    },
    {
        "id": "single_year",
        "question": "How did Berkshire perform in 2020?",
        "should_decompose": False,
        "expected_aspects": ["2020 performance"],
        "min_sub_questions": 0,
        "description": "Single year - no decomposition needed"
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
    
    try:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        analysis = json.loads(response)
    except json.JSONDecodeError:
        analysis = {
            "is_complex": False,
            "sub_questions": None,
            "parse_error": True
        }
    
    return analysis


def judge_decomposition(
    original_question: str,
    expected_aspects: list,
    sub_questions: list
) -> dict:
    """Use LLM to judge the quality of decomposition."""
    prompt = ChatPromptTemplate.from_template(DECOMPOSITION_JUDGE_PROMPT)
    
    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )
    
    chain = prompt | llm | StrOutputParser()
    
    sub_questions_str = "\n".join(f"- {sq}" for sq in sub_questions) if sub_questions else "None generated"
    expected_aspects_str = "\n".join(f"- {asp}" for asp in expected_aspects)
    
    response = chain.invoke({
        "original_question": original_question,
        "expected_aspects": expected_aspects_str,
        "sub_questions": sub_questions_str
    })
    
    try:
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        judgment = json.loads(response)
    except json.JSONDecodeError:
        judgment = {
            "coverage_score": 1,
            "specificity_score": 1,
            "independence_score": 1,
            "relevance_score": 1,
            "overall_quality": "POOR",
            "parse_error": True
        }
    
    return judgment


def evaluate_decomposition(test_case: dict, verbose: bool = False) -> dict:
    """Evaluate query decomposition for a single test case."""
    question = test_case["question"]
    should_decompose = test_case["should_decompose"]
    expected_aspects = test_case["expected_aspects"]
    min_sub_questions = test_case.get("min_sub_questions", 2)
    
    # Run the analyzer
    analysis = analyze_query(question)
    
    is_complex = analysis.get("is_complex", False)
    sub_questions = analysis.get("sub_questions") or []
    
    # Check decomposition detection
    detection_correct = is_complex == should_decompose
    
    result = {
        "test_id": test_case["id"],
        "question": question,
        "description": test_case.get("description", ""),
        "should_decompose": should_decompose,
        "detected_complex": is_complex,
        "detection_correct": detection_correct,
        "sub_questions": sub_questions,
        "num_sub_questions": len(sub_questions),
        "expected_aspects": expected_aspects,
    }
    
    # If it should decompose and did decompose, judge quality
    if should_decompose and is_complex and sub_questions:
        # Check minimum sub-questions
        has_enough_sub_qs = len(sub_questions) >= min_sub_questions
        result["has_enough_sub_questions"] = has_enough_sub_qs
        
        # Judge quality
        judgment = judge_decomposition(question, expected_aspects, sub_questions)
        result["judgment"] = judgment
        
        # Calculate overall quality
        avg_score = (
            judgment.get("coverage_score", 1) +
            judgment.get("specificity_score", 1) +
            judgment.get("independence_score", 1) +
            judgment.get("relevance_score", 1)
        ) / 4
        
        result["avg_quality_score"] = avg_score
        result["quality_pass"] = avg_score >= 3.0 and has_enough_sub_qs
        result["overall_pass"] = detection_correct and result["quality_pass"]
        
    elif should_decompose and not is_complex:
        # Should have decomposed but didn't
        result["quality_pass"] = False
        result["overall_pass"] = False
        result["error"] = "Failed to detect complexity"
        
    elif not should_decompose and is_complex:
        # Shouldn't have decomposed but did - might be okay
        result["quality_pass"] = True  # Over-decomposition is less bad
        result["overall_pass"] = False
        result["warning"] = "Decomposed simple question (unnecessary but not wrong)"
        
    else:
        # Correctly did not decompose
        result["quality_pass"] = True
        result["overall_pass"] = True
    
    if verbose:
        result["full_analysis"] = analysis
    
    return result


def run_evaluation(test_cases: list = None, verbose: bool = False) -> dict:
    """Run query decomposition evaluation on all test cases."""
    if test_cases is None:
        test_cases = TEST_CASES
    cap = os.getenv("EVAL_MAX_CASES", "").strip()  # empty or unset runs every case
    if cap:
        if not cap.isdigit() or int(cap) <= 0:
            raise SystemExit(f"EVAL_MAX_CASES must be a positive integer, or empty for all cases (got {cap!r})")
        test_cases = test_cases[: int(cap)]
    
    print("=" * 60)
    print("Agentic RAG - Query Decomposition Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Agent model: {AGENT_MODEL}")
    print(f"  - Judge model: {JUDGE_MODEL}")
    print(f"  - Test cases: {len(test_cases)}")
    print("\n" + "-" * 60)
    
    results = []
    detection_correct_count = 0
    quality_pass_count = 0
    overall_pass_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {test_case['id']}")
        print(f"    Q: {test_case['question'][:55]}...")
        print(f"    Should decompose: {test_case['should_decompose']}")
        
        result = evaluate_decomposition(test_case, verbose=verbose)
        results.append(result)
        
        # Update counts
        if result["detection_correct"]:
            detection_correct_count += 1
        if result.get("quality_pass", False):
            quality_pass_count += 1
        if result.get("overall_pass", False):
            overall_pass_count += 1
        
        # Print result
        if result.get("overall_pass"):
            print(f"    ✅ Pass")
            if result["sub_questions"]:
                print(f"       Generated {len(result['sub_questions'])} sub-questions")
                if "avg_quality_score" in result:
                    print(f"       Quality score: {result['avg_quality_score']:.1f}/5")
        elif result.get("warning"):
            print(f"    ⚠️  {result['warning']}")
            if result["sub_questions"]:
                for sq in result["sub_questions"][:3]:
                    print(f"       • {sq[:50]}...")
        else:
            print(f"    ❌ Fail")
            if result.get("error"):
                print(f"       {result['error']}")
            elif result["sub_questions"]:
                print(f"       Sub-questions generated but quality insufficient")
                if "judgment" in result:
                    j = result["judgment"]
                    print(f"       Coverage: {j.get('coverage_score', '?')}/5")
                    print(f"       Specificity: {j.get('specificity_score', '?')}/5")
    
    # Calculate metrics
    n = len(results)
    complex_cases = [r for r in results if r["should_decompose"]]
    simple_cases = [r for r in results if not r["should_decompose"]]
    
    summary = {
        "num_test_cases": n,
        "detection_accuracy": detection_correct_count / n if n > 0 else 0,
        "quality_pass_rate": quality_pass_count / n if n > 0 else 0,
        "overall_pass_rate": overall_pass_count / n if n > 0 else 0,
        "detection_correct_count": detection_correct_count,
        "quality_pass_count": quality_pass_count,
        "overall_pass_count": overall_pass_count,
        "results": results
    }
    
    # Calculate quality scores for complex cases that decomposed correctly
    quality_scores = [
        r["avg_quality_score"] 
        for r in results 
        if r.get("avg_quality_score") is not None
    ]
    if quality_scores:
        summary["avg_quality_score"] = sum(quality_scores) / len(quality_scores)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Aggregate Metrics:")
    print(f"   Detection Accuracy:  {summary['detection_accuracy']:.2%} ({detection_correct_count}/{n})")
    print(f"   Quality Pass Rate:   {summary['quality_pass_rate']:.2%} ({quality_pass_count}/{n})")
    print(f"   Overall Pass Rate:   {summary['overall_pass_rate']:.2%} ({overall_pass_count}/{n})")
    
    if "avg_quality_score" in summary:
        print(f"\n   Avg Quality Score:   {summary['avg_quality_score']:.2f}/5 (for decomposed queries)")
    
    print(f"\n📋 By Question Type:")
    if complex_cases:
        complex_pass = sum(1 for r in complex_cases if r.get("overall_pass", False))
        print(f"   Complex questions: {complex_pass}/{len(complex_cases)} passed")
    if simple_cases:
        simple_pass = sum(1 for r in simple_cases if r.get("overall_pass", False))
        print(f"   Simple questions:  {simple_pass}/{len(simple_cases)} passed")
    
    # Show failures
    failures = [r for r in results if not r.get("overall_pass", False)]
    if failures:
        print(f"\n⚠️  Failures ({len(failures)}):")
        for r in failures:
            reason = r.get("error") or r.get("warning") or "Quality insufficient"
            print(f"   • {r['test_id']}: {reason}")
    
    return summary


def main():
    """Run the query decomposition evaluation."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate agent's query decomposition")
    parser.add_argument("-v", "--verbose", action="store_true", help="Include full analysis in output")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--question", type=str, help="Test a single custom question")
    
    args = parser.parse_args()
    
    if args.question:
        print(f"\nAnalyzing question for decomposition...")
        print(f"Q: {args.question}")
        print("-" * 50)
        
        analysis = analyze_query(args.question)
        
        print(f"\nIs complex: {analysis.get('is_complex')}")
        
        sub_qs = analysis.get("sub_questions") or []
        if sub_qs:
            print(f"\nSub-questions ({len(sub_qs)}):")
            for sq in sub_qs:
                print(f"  • {sq}")
        else:
            print("\nNo sub-questions generated (simple query)")
    else:
        summary = run_evaluation(verbose=args.verbose)
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
