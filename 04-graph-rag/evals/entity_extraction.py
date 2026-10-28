"""
GraphRAG - Entity Extraction Evaluation
Tests the accuracy of entity extraction from user queries.

Entity extraction is the first step in GraphRAG - it identifies companies,
people, concepts, and terms to use as graph traversal starting points.

Metrics:
- Precision: Extracted entities that are correct / Total extracted
- Recall: Correct entities extracted / Expected entities
- F1 Score: Harmonic mean of precision and recall
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

# Query entity extraction prompt (same as retrieval.py)
QUERY_ENTITY_PROMPT = """Extract the key entities from this question about Warren Buffett's shareholder letters.

Question: {question}

Return a comma-separated list of entity names (lowercase), focusing on:
- Company names (e.g., "apple", "geico", "berkshire")
- People (e.g., "buffett", "munger")
- Concepts (e.g., "insurance", "float", "acquisitions")
- Financial terms (e.g., "dividends", "earnings")

Return ONLY the comma-separated list, nothing else. If no clear entities, return "general"."""


# =============================================================================
# Test Cases
# =============================================================================

TEST_CASES = [
    # Company-focused questions
    {
        "id": "apple_investment",
        "question": "Why did Buffett invest in Apple?",
        "expected_entities": ["buffett", "apple"],
        "entity_types": {"buffett": "person", "apple": "company"},
        "description": "Person + company extraction"
    },
    {
        "id": "geico_performance",
        "question": "How has GEICO performed over the years?",
        "expected_entities": ["geico"],
        "entity_types": {"geico": "company"},
        "description": "Single company extraction"
    },
    {
        "id": "coca_cola_amex",
        "question": "Compare Coca-Cola and American Express as investments.",
        "expected_entities": ["coca-cola", "american express"],
        "entity_types": {"coca-cola": "company", "american express": "company"},
        "description": "Multiple companies"
    },
    {
        "id": "berkshire_subsidiaries",
        "question": "What are Berkshire Hathaway's main subsidiaries?",
        "expected_entities": ["berkshire", "berkshire hathaway"],
        "entity_types": {"berkshire": "company", "berkshire hathaway": "company"},
        "description": "Company with variant names"
    },
    
    # Concept-focused questions
    {
        "id": "insurance_float",
        "question": "How does insurance float work?",
        "expected_entities": ["insurance", "float"],
        "entity_types": {"insurance": "concept", "float": "concept"},
        "description": "Financial concepts"
    },
    {
        "id": "acquisitions_strategy",
        "question": "What is Berkshire's acquisition strategy?",
        "expected_entities": ["berkshire", "acquisition"],
        "entity_types": {"berkshire": "company", "acquisition": "concept"},
        "description": "Company + concept"
    },
    {
        "id": "dividends_policy",
        "question": "Why doesn't Berkshire pay dividends?",
        "expected_entities": ["berkshire", "dividends"],
        "entity_types": {"berkshire": "company", "dividends": "concept"},
        "description": "Company + financial term"
    },
    
    # People-focused questions
    {
        "id": "buffett_munger",
        "question": "How do Buffett and Munger make investment decisions?",
        "expected_entities": ["buffett", "munger"],
        "entity_types": {"buffett": "person", "munger": "person"},
        "description": "Multiple people"
    },
    {
        "id": "charlie_munger",
        "question": "What is Charlie Munger's role at Berkshire?",
        "expected_entities": ["charlie munger", "munger", "berkshire"],
        "entity_types": {"charlie munger": "person", "munger": "person", "berkshire": "company"},
        "description": "Person with full name + company"
    },
    
    # Mixed questions
    {
        "id": "geico_float",
        "question": "How does GEICO generate insurance float for Berkshire?",
        "expected_entities": ["geico", "insurance", "float", "berkshire"],
        "entity_types": {"geico": "company", "insurance": "concept", "float": "concept", "berkshire": "company"},
        "description": "Companies + concepts mixed"
    },
    {
        "id": "apple_tech_sector",
        "question": "What does Buffett think about Apple and the technology sector?",
        "expected_entities": ["buffett", "apple", "technology"],
        "entity_types": {"buffett": "person", "apple": "company", "technology": "concept"},
        "description": "Person + company + sector"
    },
    
    # Edge cases
    {
        "id": "general_philosophy",
        "question": "What is the investment philosophy?",
        "expected_entities": ["investment"],
        "entity_types": {"investment": "concept"},
        "description": "General question - minimal entities"
    },
    {
        "id": "year_specific",
        "question": "How did Berkshire perform in 2020?",
        "expected_entities": ["berkshire"],
        "entity_types": {"berkshire": "company"},
        "description": "Year should NOT be an entity (it's metadata)"
    },
    {
        "id": "no_entities",
        "question": "What happened last year?",
        "expected_entities": [],
        "entity_types": {},
        "description": "Vague question - may return 'general' or empty"
    },
]


def extract_query_entities(question: str) -> list[str]:
    """Extract entities from a question using the LLM."""
    prompt = ChatPromptTemplate.from_template(QUERY_ENTITY_PROMPT)
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({"question": question})
    entities = [e.strip().lower() for e in response.split(",")]
    return [e for e in entities if e and e != "general"]


def normalize_entity(entity: str) -> str:
    """Normalize entity name for comparison."""
    # Remove common variations
    entity = entity.lower().strip()
    entity = entity.replace("'s", "").replace("'", "")
    entity = entity.replace("-", " ").replace("_", " ")
    return entity


def entities_match(extracted: str, expected: str) -> bool:
    """Check if two entities match (allowing for variations)."""
    ext = normalize_entity(extracted)
    exp = normalize_entity(expected)
    
    # Exact match
    if ext == exp:
        return True
    
    # One contains the other (e.g., "berkshire" matches "berkshire hathaway")
    if ext in exp or exp in ext:
        return True
    
    # Common aliases
    aliases = {
        "buffett": ["warren buffett", "warren", "mr buffett"],
        "munger": ["charlie munger", "charlie", "mr munger"],
        "berkshire": ["berkshire hathaway", "brk", "bh"],
        "coca cola": ["coke", "ko"],
        "american express": ["amex", "axp"],
        "apple": ["aapl"],
        "geico": ["government employees insurance"],
    }
    
    for canonical, variants in aliases.items():
        all_forms = [canonical] + variants
        if ext in all_forms and exp in all_forms:
            return True
    
    return False


def calculate_metrics(extracted: list[str], expected: list[str]) -> dict:
    """Calculate precision, recall, and F1 for entity extraction."""
    if not extracted and not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "true_positives": 0}
    
    if not extracted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "true_positives": 0}
    
    if not expected:
        # If we extracted entities but none were expected
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "true_positives": 0}
    
    # Count true positives (extracted entities that match expected)
    true_positives = 0
    matched_expected = set()
    
    for ext in extracted:
        for exp in expected:
            if exp not in matched_expected and entities_match(ext, exp):
                true_positives += 1
                matched_expected.add(exp)
                break
    
    precision = true_positives / len(extracted) if extracted else 0
    recall = true_positives / len(expected) if expected else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": len(extracted) - true_positives,
        "false_negatives": len(expected) - true_positives
    }


def evaluate_entity_extraction(test_case: dict, verbose: bool = False) -> dict:
    """Evaluate entity extraction for a single test case."""
    question = test_case["question"]
    expected = test_case["expected_entities"]
    
    # Extract entities
    extracted = extract_query_entities(question)
    
    # Calculate metrics
    metrics = calculate_metrics(extracted, expected)
    
    result = {
        "test_id": test_case["id"],
        "question": question,
        "description": test_case.get("description", ""),
        "expected_entities": expected,
        "extracted_entities": extracted,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "true_positives": metrics["true_positives"],
        "false_positives": metrics.get("false_positives", 0),
        "false_negatives": metrics.get("false_negatives", 0),
    }
    
    # Determine pass/fail
    result["pass"] = metrics["f1"] >= 0.5  # At least 50% F1 score
    
    if verbose:
        # Find which entities were missed/extra
        result["missed_entities"] = [
            exp for exp in expected 
            if not any(entities_match(ext, exp) for ext in extracted)
        ]
        result["extra_entities"] = [
            ext for ext in extracted
            if not any(entities_match(ext, exp) for exp in expected)
        ]
    
    return result


def run_evaluation(test_cases: list = None, verbose: bool = False) -> dict:
    """Run entity extraction evaluation on all test cases."""
    if test_cases is None:
        test_cases = TEST_CASES
    cap = os.getenv("EVAL_MAX_CASES", "").strip()  # empty or unset runs every case
    if cap:
        if not cap.isdigit() or int(cap) <= 0:
            raise SystemExit(f"EVAL_MAX_CASES must be a positive integer, or empty for all cases (got {cap!r})")
        test_cases = test_cases[: int(cap)]
    
    print("=" * 60)
    print("GraphRAG - Entity Extraction Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Test cases: {len(test_cases)}")
    print("\n" + "-" * 60)
    
    results = []
    pass_count = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {test_case['id']}")
        print(f"    Q: {test_case['question'][:50]}...")
        print(f"    Expected: {test_case['expected_entities']}")
        
        result = evaluate_entity_extraction(test_case, verbose=verbose)
        results.append(result)
        
        # Update totals
        total_precision += result["precision"]
        total_recall += result["recall"]
        total_f1 += result["f1"]
        
        if result["pass"]:
            pass_count += 1
        
        # Print result
        extracted = result["extracted_entities"]
        print(f"    Extracted: {extracted}")
        
        if result["pass"]:
            print(f"    ✅ P={result['precision']:.0%} R={result['recall']:.0%} F1={result['f1']:.0%}")
        else:
            print(f"    ❌ P={result['precision']:.0%} R={result['recall']:.0%} F1={result['f1']:.0%}")
            if verbose and result.get("missed_entities"):
                print(f"       Missed: {result['missed_entities']}")
            if verbose and result.get("extra_entities"):
                print(f"       Extra: {result['extra_entities']}")
    
    # Calculate summary metrics
    n = len(results)
    summary = {
        "num_test_cases": n,
        "pass_count": pass_count,
        "pass_rate": pass_count / n if n > 0 else 0,
        "avg_precision": total_precision / n if n > 0 else 0,
        "avg_recall": total_recall / n if n > 0 else 0,
        "avg_f1": total_f1 / n if n > 0 else 0,
        "results": results
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Aggregate Metrics:")
    print(f"   Pass Rate:       {summary['pass_rate']:.2%} ({pass_count}/{n})")
    print(f"   Avg Precision:   {summary['avg_precision']:.2%}")
    print(f"   Avg Recall:      {summary['avg_recall']:.2%}")
    print(f"   Avg F1 Score:    {summary['avg_f1']:.2%}")
    
    # Breakdown by entity type (from test case metadata)
    print(f"\n📋 Per-Query Breakdown:")
    for r in results:
        status = "✅" if r["pass"] else "❌"
        print(f"   {status} {r['test_id']}: F1={r['f1']:.0%} (P={r['precision']:.0%}, R={r['recall']:.0%})")
    
    # Show common issues
    all_missed = []
    all_extra = []
    for r in results:
        if "missed_entities" in r:
            all_missed.extend(r["missed_entities"])
        if "extra_entities" in r:
            all_extra.extend(r["extra_entities"])
    
    if all_missed:
        from collections import Counter
        missed_counts = Counter(all_missed)
        print(f"\n⚠️  Commonly Missed Entities:")
        for entity, count in missed_counts.most_common(5):
            print(f"   • {entity} (missed {count}x)")
    
    return summary


def main():
    """Run the entity extraction evaluation."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate entity extraction accuracy")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show missed/extra entities")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--question", type=str, help="Test a single custom question")
    
    args = parser.parse_args()
    
    if args.question:
        print(f"\nExtracting entities from question...")
        print(f"Q: {args.question}")
        print("-" * 50)
        
        entities = extract_query_entities(args.question)
        
        print(f"\nExtracted entities ({len(entities)}):")
        for e in entities:
            print(f"  • {e}")
    else:
        summary = run_evaluation(verbose=args.verbose)
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
