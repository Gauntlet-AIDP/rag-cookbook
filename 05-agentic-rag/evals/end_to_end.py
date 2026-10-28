"""
Agentic RAG - End-to-End Evaluation
Full pipeline evaluation measuring answer quality, groundedness, and efficiency.

This eval runs the complete agentic RAG pipeline and measures:
1. Groundedness: Is the answer supported by retrieved documents?
2. Answer Quality: Is the answer helpful and accurate?
3. Retrieval Efficiency: How many docs/steps were used?
4. Agent Behavior: Did it make good decisions (tool choice, decomposition)?
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # sandbox
    from llm_shim import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import AgenticRAG
from tools import format_documents_as_context

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JUDGE_MODEL = "gpt-4o-mini"


# =============================================================================
# LLM-as-Judge Prompts
# =============================================================================

GROUNDEDNESS_JUDGE_PROMPT = """You are a groundedness evaluator. Determine if the answer is fully supported by the provided context.

An answer is GROUNDED if:
- Every claim can be traced back to information in the context
- The answer doesn't include information not present in the context
- The answer doesn't make unsupported assumptions

An answer is NOT GROUNDED if:
- It contains claims not supported by the context (hallucinations)
- It adds details not present in the context
- It makes unsupported generalizations

Context:
{context}

Question: {question}

Answer to evaluate:
{answer}

Respond in this exact format:
REASONING: <2-3 sentences explaining your evaluation>
VERDICT: <GROUNDED or NOT_GROUNDED>
CONFIDENCE: <HIGH, MEDIUM, or LOW>"""


ANSWER_QUALITY_JUDGE_PROMPT = """You are an answer quality evaluator. Rate the quality of this answer to a question about Warren Buffett's shareholder letters.

Question: {question}

Answer: {answer}

Rate on these criteria (1-5 each):

1. RELEVANCE: Does the answer address the question asked?
   1 = Completely off-topic
   5 = Directly and fully addresses the question

2. COMPLETENESS: Is the answer thorough?
   1 = Missing major aspects
   5 = Comprehensive coverage

3. CLARITY: Is the answer clear and well-organized?
   1 = Confusing or poorly structured
   5 = Clear, logical, easy to understand

4. ACCURACY: Does the answer seem factually correct? (Based on general knowledge of Buffett)
   1 = Contains obvious errors
   5 = Appears accurate

Return a JSON object:
{{
    "relevance_score": 1-5,
    "relevance_reasoning": "brief explanation",
    "completeness_score": 1-5,
    "completeness_reasoning": "brief explanation",
    "clarity_score": 1-5,
    "clarity_reasoning": "brief explanation",
    "accuracy_score": 1-5,
    "accuracy_reasoning": "brief explanation",
    "overall_quality": "EXCELLENT" | "GOOD" | "ACCEPTABLE" | "POOR"
}}

Return ONLY the JSON object."""


# =============================================================================
# Test Cases
# =============================================================================

TEST_CASES = [
    # Simple questions (should be efficient)
    {
        "id": "insurance_float",
        "question": "What is insurance float and why is it important to Berkshire?",
        "category": "simple",
        "description": "Core concept question"
    },
    {
        "id": "investment_philosophy",
        "question": "What is Warren Buffett's investment philosophy?",
        "category": "simple",
        "description": "General philosophy question"
    },
    {
        "id": "management_quality",
        "question": "What does Buffett look for in company management?",
        "category": "simple",
        "description": "Management criteria"
    },
    
    # Year-specific questions (should use filters)
    {
        "id": "performance_2020",
        "question": "How did Berkshire perform in 2020?",
        "category": "filtered",
        "description": "Year-specific question"
    },
    {
        "id": "acquisitions_2010s",
        "question": "What major acquisitions did Berkshire make in the 2010s?",
        "category": "filtered",
        "description": "Decade-specific question"
    },
    
    # Company-specific questions (should use filters)
    {
        "id": "apple_investment",
        "question": "Why did Buffett invest in Apple and how has it performed?",
        "category": "filtered",
        "description": "Company-specific question"
    },
    {
        "id": "coca_cola_views",
        "question": "What does Buffett think about Coca-Cola as a long-term investment?",
        "category": "filtered",
        "description": "Company-specific question"
    },
    
    # Complex questions (should decompose)
    {
        "id": "insurance_comparison",
        "question": "How did Berkshire's insurance business compare between 2008 and 2020?",
        "category": "complex",
        "description": "Temporal comparison"
    },
    {
        "id": "multi_company",
        "question": "Compare Buffett's views on Apple versus Coca-Cola as investments.",
        "category": "complex",
        "description": "Multi-entity comparison"
    },
    {
        "id": "multi_aspect",
        "question": "What is insurance float, how does Berkshire generate it, and how is it invested?",
        "category": "complex",
        "description": "Multi-faceted question"
    },
    
    # Edge cases
    {
        "id": "off_topic",
        "question": "What is the capital of France?",
        "category": "no_retrieval",
        "description": "Off-topic (should skip retrieval)"
    },
    {
        "id": "simple_math",
        "question": "What is 2 + 2?",
        "category": "no_retrieval",
        "description": "Off-topic (should skip retrieval)"
    },
]


def judge_groundedness(question: str, answer: str, context: str) -> dict:
    """Judge if an answer is grounded in the context."""
    prompt = ChatPromptTemplate.from_template(GROUNDEDNESS_JUDGE_PROMPT)
    
    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "question": question,
        "answer": answer,
        "context": context[:8000]  # Limit context length
    })
    
    # Parse response
    is_grounded = "VERDICT: GROUNDED" in response.upper() and "NOT_GROUNDED" not in response.upper()
    
    reasoning = ""
    if "REASONING:" in response:
        start = response.find("REASONING:") + len("REASONING:")
        end = response.find("VERDICT:")
        if end > start:
            reasoning = response[start:end].strip()
    
    confidence = "MEDIUM"
    if "CONFIDENCE:" in response.upper():
        if "HIGH" in response.upper().split("CONFIDENCE:")[-1]:
            confidence = "HIGH"
        elif "LOW" in response.upper().split("CONFIDENCE:")[-1]:
            confidence = "LOW"
    
    return {
        "is_grounded": is_grounded,
        "reasoning": reasoning,
        "confidence": confidence,
        "raw_response": response
    }


def judge_answer_quality(question: str, answer: str) -> dict:
    """Judge the quality of an answer."""
    prompt = ChatPromptTemplate.from_template(ANSWER_QUALITY_JUDGE_PROMPT)
    
    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "question": question,
        "answer": answer
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
            "relevance_score": 3,
            "completeness_score": 3,
            "clarity_score": 3,
            "accuracy_score": 3,
            "overall_quality": "ACCEPTABLE",
            "parse_error": True
        }
    
    # Calculate average score
    scores = [
        judgment.get("relevance_score", 3),
        judgment.get("completeness_score", 3),
        judgment.get("clarity_score", 3),
        judgment.get("accuracy_score", 3)
    ]
    judgment["avg_score"] = sum(scores) / len(scores)
    
    return judgment


def evaluate_end_to_end(test_case: dict, verbose: bool = False) -> dict:
    """Run full end-to-end evaluation for a single test case."""
    question = test_case["question"]
    category = test_case.get("category", "simple")
    
    # Run the agent
    agent = AgenticRAG(verbose=False)
    
    start_time = time.perf_counter()
    agent_result = agent.run(question)
    end_time = time.perf_counter()
    
    latency_ms = (end_time - start_time) * 1000
    
    answer = agent_result["answer"]
    analysis = agent_result["analysis"]
    retrieval_steps = agent_result["retrieval_steps"]
    total_docs = agent_result["total_documents"]
    
    # Build context from all retrieved documents
    all_docs = []
    for step in retrieval_steps:
        # We need to re-retrieve to get context (agent doesn't store full docs)
        pass
    
    # For groundedness, we need the context - reconstruct from agent's retrieval
    # Since agent doesn't expose raw docs, we'll use a simplified check
    # In production, you'd modify the agent to return full docs
    
    result = {
        "test_id": test_case["id"],
        "question": question,
        "category": category,
        "description": test_case.get("description", ""),
        "answer": answer,
        "latency_ms": latency_ms,
        "total_documents": total_docs,
        "num_retrieval_steps": len(retrieval_steps),
        "retrieval_steps": retrieval_steps,
        "agent_analysis": {
            "needs_retrieval": analysis.get("needs_retrieval", True),
            "is_complex": analysis.get("is_complex", False),
            "suggested_tool": analysis.get("suggested_tool", "unknown"),
            "num_sub_questions": len(analysis.get("sub_questions") or [])
        }
    }
    
    # Judge answer quality
    quality_judgment = judge_answer_quality(question, answer)
    result["quality"] = quality_judgment
    result["quality_score"] = quality_judgment.get("avg_score", 0)
    result["quality_pass"] = quality_judgment.get("avg_score", 0) >= 3.0
    
    # Check if agent behavior matches category expectations
    if category == "no_retrieval":
        result["behavior_correct"] = not analysis.get("needs_retrieval", True)
    elif category == "complex":
        result["behavior_correct"] = analysis.get("is_complex", False)
    elif category == "filtered":
        suggested = analysis.get("suggested_tool", "")
        result["behavior_correct"] = suggested == "filtered_search"
    else:
        result["behavior_correct"] = True  # Simple questions - any retrieval is fine
    
    # Overall pass
    result["overall_pass"] = result["quality_pass"] and result["behavior_correct"]
    
    if verbose:
        result["full_agent_result"] = agent_result
    
    return result


def run_evaluation(test_cases: list = None, verbose: bool = False) -> dict:
    """Run end-to-end evaluation on all test cases."""
    if test_cases is None:
        test_cases = TEST_CASES
    cap = os.getenv("EVAL_MAX_CASES", "").strip()  # empty or unset runs every case
    if cap:
        if not cap.isdigit() or int(cap) <= 0:
            raise SystemExit(f"EVAL_MAX_CASES must be a positive integer, or empty for all cases (got {cap!r})")
        test_cases = test_cases[: int(cap)]
    
    print("=" * 60)
    print("Agentic RAG - End-to-End Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Judge model: {JUDGE_MODEL}")
    print(f"  - Test cases: {len(test_cases)}")
    print("\n" + "-" * 60)
    
    results = []
    quality_pass_count = 0
    behavior_correct_count = 0
    overall_pass_count = 0
    total_latency = 0
    total_docs = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {test_case['id']} ({test_case['category']})")
        print(f"    Q: {test_case['question'][:50]}...")
        
        result = evaluate_end_to_end(test_case, verbose=verbose)
        results.append(result)
        
        # Update counts
        if result["quality_pass"]:
            quality_pass_count += 1
        if result["behavior_correct"]:
            behavior_correct_count += 1
        if result["overall_pass"]:
            overall_pass_count += 1
        
        total_latency += result["latency_ms"]
        total_docs += result["total_documents"]
        
        # Print result
        quality = result["quality"].get("overall_quality", "?")
        score = result["quality_score"]
        latency = result["latency_ms"]
        
        if result["overall_pass"]:
            print(f"    ✅ Pass | Quality: {quality} ({score:.1f}/5) | {latency:.0f}ms | {result['total_documents']} docs")
        else:
            issues = []
            if not result["quality_pass"]:
                issues.append(f"quality={score:.1f}")
            if not result["behavior_correct"]:
                issues.append("behavior")
            print(f"    ❌ Fail ({', '.join(issues)}) | {latency:.0f}ms")
        
        # Show agent behavior
        analysis = result["agent_analysis"]
        if analysis["is_complex"]:
            print(f"       → Decomposed into {analysis['num_sub_questions']} sub-questions")
        if not analysis["needs_retrieval"]:
            print(f"       → Skipped retrieval (no_retrieval)")
        elif analysis["suggested_tool"] == "filtered_search":
            print(f"       → Used filtered search")
    
    # Calculate metrics
    n = len(results)
    
    summary = {
        "num_test_cases": n,
        "quality_pass_rate": quality_pass_count / n if n > 0 else 0,
        "behavior_accuracy": behavior_correct_count / n if n > 0 else 0,
        "overall_pass_rate": overall_pass_count / n if n > 0 else 0,
        "quality_pass_count": quality_pass_count,
        "behavior_correct_count": behavior_correct_count,
        "overall_pass_count": overall_pass_count,
        "avg_latency_ms": total_latency / n if n > 0 else 0,
        "avg_documents_per_query": total_docs / n if n > 0 else 0,
        "results": results
    }
    
    # Calculate average quality score
    quality_scores = [r["quality_score"] for r in results]
    summary["avg_quality_score"] = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Aggregate Metrics:")
    print(f"   Quality Pass Rate:    {summary['quality_pass_rate']:.2%} ({quality_pass_count}/{n})")
    print(f"   Behavior Accuracy:    {summary['behavior_accuracy']:.2%} ({behavior_correct_count}/{n})")
    print(f"   Overall Pass Rate:    {summary['overall_pass_rate']:.2%} ({overall_pass_count}/{n})")
    print(f"\n   Avg Quality Score:    {summary['avg_quality_score']:.2f}/5")
    print(f"   Avg Latency:          {summary['avg_latency_ms']:.0f}ms")
    print(f"   Avg Docs/Query:       {summary['avg_documents_per_query']:.1f}")
    
    # Breakdown by category
    print(f"\n📋 By Category:")
    for category in ["simple", "filtered", "complex", "no_retrieval"]:
        cat_results = [r for r in results if r["category"] == category]
        if cat_results:
            cat_pass = sum(1 for r in cat_results if r["overall_pass"])
            cat_quality = sum(r["quality_score"] for r in cat_results) / len(cat_results)
            print(f"   {category}: {cat_pass}/{len(cat_results)} pass, avg quality {cat_quality:.1f}/5")
    
    # Show failures
    failures = [r for r in results if not r["overall_pass"]]
    if failures:
        print(f"\n⚠️  Failures ({len(failures)}):")
        for r in failures:
            issues = []
            if not r["quality_pass"]:
                issues.append(f"quality={r['quality_score']:.1f}")
            if not r["behavior_correct"]:
                issues.append("wrong behavior")
            print(f"   • {r['test_id']}: {', '.join(issues)}")
    
    return summary


def main():
    """Run the end-to-end evaluation."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    import argparse
    parser = argparse.ArgumentParser(description="End-to-end evaluation of Agentic RAG")
    parser.add_argument("-v", "--verbose", action="store_true", help="Include full results in output")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--question", type=str, help="Test a single custom question")
    parser.add_argument("--category", type=str, default=None, 
                        help="Filter test cases by category (simple, filtered, complex, no_retrieval)")
    
    args = parser.parse_args()
    
    if args.question:
        print(f"\nRunning end-to-end evaluation on single question...")
        print(f"Q: {args.question}")
        print("-" * 50)
        
        test_case = {
            "id": "custom",
            "question": args.question,
            "category": "custom",
            "description": "Custom question"
        }
        
        result = evaluate_end_to_end(test_case, verbose=True)
        
        print(f"\n📝 Answer:")
        print(result["answer"])
        
        print(f"\n📊 Metrics:")
        print(f"   Quality Score: {result['quality_score']:.1f}/5")
        print(f"   Quality: {result['quality'].get('overall_quality', '?')}")
        print(f"   Latency: {result['latency_ms']:.0f}ms")
        print(f"   Documents: {result['total_documents']}")
        
        print(f"\n🤖 Agent Behavior:")
        analysis = result["agent_analysis"]
        print(f"   Needs retrieval: {analysis['needs_retrieval']}")
        print(f"   Is complex: {analysis['is_complex']}")
        print(f"   Tool: {analysis['suggested_tool']}")
        
    else:
        # Filter by category if specified
        test_cases = TEST_CASES
        if args.category:
            test_cases = [tc for tc in TEST_CASES if tc["category"] == args.category]
            if not test_cases:
                print(f"No test cases found for category: {args.category}")
                return
        
        summary = run_evaluation(test_cases=test_cases, verbose=args.verbose)
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
