"""
GraphRAG - Multi-Hop Reasoning Evaluation
Tests whether graph traversal discovers relevant entity connections.

Multi-hop reasoning is the core value of GraphRAG - finding connections
between entities that don't appear together in any single document.

This eval tests:
1. Can the graph find expected relationship paths?
2. Are intermediate entities relevant to the question?
3. Does multi-hop context improve answer quality?
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
from neo4j import GraphDatabase

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

JUDGE_MODEL = "gpt-4o-mini"

# Query entity extraction prompt
QUERY_ENTITY_PROMPT = """Extract the key entities from this question about Warren Buffett's shareholder letters.

Question: {question}

Return a comma-separated list of entity names (lowercase), focusing on:
- Company names (e.g., "apple", "geico", "berkshire")
- People (e.g., "buffett", "munger")
- Concepts (e.g., "insurance", "float", "acquisitions")
- Financial terms (e.g., "dividends", "earnings")

Return ONLY the comma-separated list, nothing else. If no clear entities, return "general"."""


# LLM-as-Judge prompt for path relevance
PATH_RELEVANCE_PROMPT = """You are evaluating whether a graph traversal path is relevant to answering a question.

Question: {question}

Starting Entities (from question): {start_entities}

Graph Traversal Path:
{path_description}

Related Entities Found: {related_entities}

Evaluate:
1. Are the related entities relevant to answering the question?
2. Does the path make logical sense for this question?
3. Would these entities help find useful documents?

Return a JSON object:
{{
    "path_relevant": true/false,
    "relevance_score": 1-5 (1=irrelevant, 5=highly relevant),
    "reasoning": "brief explanation",
    "useful_entities": ["entity1", "entity2"] // subset of related entities that are useful
}}

Return ONLY the JSON object."""


# =============================================================================
# Test Cases
# =============================================================================

TEST_CASES = [
    # Single-hop connections
    {
        "id": "geico_insurance",
        "question": "How does GEICO contribute to Berkshire's insurance operations?",
        "start_entities": ["geico"],
        "expected_connections": ["insurance", "berkshire", "float"],
        "expected_hops": 1,
        "description": "GEICO → Insurance concepts (1 hop)"
    },
    {
        "id": "apple_berkshire",
        "question": "Why is Apple important to Berkshire's portfolio?",
        "start_entities": ["apple"],
        "expected_connections": ["berkshire", "investment", "portfolio"],
        "expected_hops": 1,
        "description": "Apple → Berkshire connection (1 hop)"
    },
    
    # Two-hop connections
    {
        "id": "geico_float_investments",
        "question": "How does GEICO's float contribute to Berkshire's investment returns?",
        "start_entities": ["geico", "float"],
        "expected_connections": ["insurance", "berkshire", "investment", "returns"],
        "expected_hops": 2,
        "description": "GEICO → Float → Investments (2 hops)"
    },
    {
        "id": "buffett_apple_tech",
        "question": "What does Buffett think about Apple and technology investments?",
        "start_entities": ["buffett", "apple"],
        "expected_connections": ["berkshire", "technology", "investment"],
        "expected_hops": 2,
        "description": "Buffett → Apple → Technology (2 hops)"
    },
    {
        "id": "insurance_acquisitions",
        "question": "How does Berkshire's insurance business fund its acquisitions?",
        "start_entities": ["insurance", "acquisitions"],
        "expected_connections": ["float", "berkshire", "capital"],
        "expected_hops": 2,
        "description": "Insurance → Float → Capital → Acquisitions (2 hops)"
    },
    
    # Cross-domain connections
    {
        "id": "sees_candies_float",
        "question": "What's the connection between See's Candies and insurance float?",
        "start_entities": ["see's candies", "float"],
        "expected_connections": ["berkshire", "consumer", "insurance", "capital"],
        "expected_hops": 2,
        "description": "Consumer brand → Berkshire → Insurance (cross-domain)"
    },
    {
        "id": "munger_investments",
        "question": "How has Charlie Munger influenced Berkshire's investment decisions?",
        "start_entities": ["munger", "charlie munger"],
        "expected_connections": ["berkshire", "buffett", "investment", "management"],
        "expected_hops": 2,
        "description": "Munger → Berkshire → Investment philosophy"
    },
    
    # Topic-bridging connections
    {
        "id": "management_performance",
        "question": "How do Berkshire's management principles affect company performance?",
        "start_entities": ["management"],
        "expected_connections": ["berkshire", "culture", "performance", "subsidiaries"],
        "expected_hops": 2,
        "description": "Management → Culture → Performance"
    },
    {
        "id": "dividends_capital",
        "question": "Why does Berkshire prefer buybacks over dividends for capital allocation?",
        "start_entities": ["dividends", "buybacks"],
        "expected_connections": ["berkshire", "capital", "allocation", "shareholders"],
        "expected_hops": 2,
        "description": "Dividends → Capital allocation → Buybacks"
    },
    
    # Complex multi-entity
    {
        "id": "coca_cola_moat",
        "question": "How does Coca-Cola's brand moat align with Buffett's investment philosophy?",
        "start_entities": ["coca-cola", "buffett"],
        "expected_connections": ["brand", "moat", "investment", "berkshire"],
        "expected_hops": 2,
        "description": "Coca-Cola → Brand/Moat → Investment philosophy"
    },
]


class Neo4jConnection:
    """Neo4j database connection manager."""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def execute_query(self, query: str, parameters: dict = None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]


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


def find_related_entities(neo4j: Neo4jConnection, entities: list[str], max_hops: int = 2) -> dict:
    """Find entities related via graph traversal."""
    if not entities:
        return {}
    
    related = {}
    
    for entity in entities:
        try:
            results = neo4j.execute_query("""
                MATCH (start:Entity)
                WHERE start.name CONTAINS $entity_name
                MATCH path = (start)-[*1..""" + str(max_hops) + """]-(related:Entity)
                RETURN DISTINCT related.name as name, length(path) as distance
                ORDER BY distance
                LIMIT 30
            """, {"entity_name": entity})
            
            for row in results:
                name = row["name"]
                distance = row["distance"]
                if name not in related or distance < related[name]:
                    related[name] = distance
        except Exception as e:
            print(f"   Warning: Graph query failed for '{entity}': {e}")
    
    return related


def find_paths_between_entities(neo4j: Neo4jConnection, start: str, end: str, max_hops: int = 3) -> list:
    """Find paths between two entities in the graph."""
    try:
        results = neo4j.execute_query("""
            MATCH (s:Entity), (e:Entity)
            WHERE s.name CONTAINS $start AND e.name CONTAINS $end
            MATCH path = shortestPath((s)-[*1..""" + str(max_hops) + """]-(e))
            RETURN [n IN nodes(path) | n.name] as path_nodes, length(path) as length
            LIMIT 5
        """, {"start": start, "end": end})
        
        return [{"nodes": r["path_nodes"], "length": r["length"]} for r in results]
    except Exception:
        return []


def judge_path_relevance(question: str, start_entities: list, related_entities: dict) -> dict:
    """Use LLM to judge if the traversal path is relevant."""
    prompt = ChatPromptTemplate.from_template(PATH_RELEVANCE_PROMPT)
    
    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )
    
    chain = prompt | llm | StrOutputParser()
    
    # Build path description
    path_parts = []
    for entity, distance in sorted(related_entities.items(), key=lambda x: x[1]):
        path_parts.append(f"  • {entity} ({distance} hop{'s' if distance > 1 else ''})")
    path_description = "\n".join(path_parts[:15]) if path_parts else "No paths found"
    
    response = chain.invoke({
        "question": question,
        "start_entities": ", ".join(start_entities),
        "path_description": path_description,
        "related_entities": ", ".join(list(related_entities.keys())[:15])
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
            "path_relevant": False,
            "relevance_score": 1,
            "reasoning": "Failed to parse judgment",
            "useful_entities": []
        }
    
    return judgment


def evaluate_multi_hop(test_case: dict, neo4j: Neo4jConnection, verbose: bool = False) -> dict:
    """Evaluate multi-hop reasoning for a single test case."""
    question = test_case["question"]
    start_entities = test_case["start_entities"]
    expected_connections = test_case["expected_connections"]
    expected_hops = test_case.get("expected_hops", 2)
    
    # Find related entities via graph traversal
    related_entities = find_related_entities(neo4j, start_entities, max_hops=expected_hops)
    
    # Check if expected connections were found
    found_connections = []
    missed_connections = []
    
    for expected in expected_connections:
        found = False
        for related in related_entities.keys():
            if expected.lower() in related.lower() or related.lower() in expected.lower():
                found = True
                found_connections.append(expected)
                break
        if not found:
            missed_connections.append(expected)
    
    connection_recall = len(found_connections) / len(expected_connections) if expected_connections else 0
    
    # Judge path relevance
    judgment = judge_path_relevance(question, start_entities, related_entities)
    
    result = {
        "test_id": test_case["id"],
        "question": question,
        "description": test_case.get("description", ""),
        "start_entities": start_entities,
        "expected_connections": expected_connections,
        "expected_hops": expected_hops,
        "related_entities_found": len(related_entities),
        "related_entities": dict(list(related_entities.items())[:10]),
        "found_connections": found_connections,
        "missed_connections": missed_connections,
        "connection_recall": connection_recall,
        "path_relevant": judgment.get("path_relevant", False),
        "relevance_score": judgment.get("relevance_score", 1),
        "reasoning": judgment.get("reasoning", ""),
        "useful_entities": judgment.get("useful_entities", []),
    }
    
    # Determine pass/fail
    # Pass if: found at least half expected connections AND path is relevant
    result["pass"] = connection_recall >= 0.5 and judgment.get("path_relevant", False)
    
    if verbose:
        result["all_related_entities"] = related_entities
    
    return result


def run_evaluation(test_cases: list = None, verbose: bool = False) -> dict:
    """Run multi-hop reasoning evaluation on all test cases."""
    if test_cases is None:
        test_cases = TEST_CASES
    cap = os.getenv("EVAL_MAX_CASES", "").strip()  # empty or unset runs every case
    if cap:
        if not cap.isdigit() or int(cap) <= 0:
            raise SystemExit(f"EVAL_MAX_CASES must be a positive integer, or empty for all cases (got {cap!r})")
        test_cases = test_cases[: int(cap)]
    
    print("=" * 60)
    print("GraphRAG - Multi-Hop Reasoning Evaluation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Neo4j: {NEO4J_URI}")
    print(f"  - Test cases: {len(test_cases)}")
    
    # Connect to Neo4j
    print("\n🔌 Connecting to Neo4j...")
    try:
        neo4j = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        neo4j.execute_query("RETURN 1")
        print("   Connected!")
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        print("\n   Make sure Neo4j is running:")
        print("   cd 04-graph-rag && docker-compose up -d")
        print("   Then run graph_builder.py to populate the graph")
        return {"error": str(e), "results": []}
    
    print("\n" + "-" * 60)
    
    results = []
    pass_count = 0
    total_recall = 0
    total_relevance = 0
    
    try:
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] {test_case['id']}")
            print(f"    Q: {test_case['question'][:50]}...")
            print(f"    Start: {test_case['start_entities']}")
            print(f"    Expected: {test_case['expected_connections'][:3]}...")
            
            result = evaluate_multi_hop(test_case, neo4j, verbose=verbose)
            results.append(result)
            
            # Update totals
            total_recall += result["connection_recall"]
            total_relevance += result["relevance_score"]
            
            if result["pass"]:
                pass_count += 1
            
            # Print result
            if result["pass"]:
                print(f"    ✅ Found {len(result['found_connections'])}/{len(result['expected_connections'])} connections")
                print(f"       Related entities: {result['related_entities_found']}")
                print(f"       Relevance: {result['relevance_score']}/5")
            else:
                print(f"    ❌ Recall={result['connection_recall']:.0%}, Relevant={result['path_relevant']}")
                if result["missed_connections"]:
                    print(f"       Missed: {result['missed_connections']}")
                if result["reasoning"]:
                    print(f"       {result['reasoning'][:60]}...")
    finally:
        neo4j.close()
    
    # Calculate summary metrics
    n = len(results)
    summary = {
        "num_test_cases": n,
        "pass_count": pass_count,
        "pass_rate": pass_count / n if n > 0 else 0,
        "avg_connection_recall": total_recall / n if n > 0 else 0,
        "avg_relevance_score": total_relevance / n if n > 0 else 0,
        "results": results
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Aggregate Metrics:")
    print(f"   Pass Rate:            {summary['pass_rate']:.2%} ({pass_count}/{n})")
    print(f"   Avg Connection Recall: {summary['avg_connection_recall']:.2%}")
    print(f"   Avg Relevance Score:   {summary['avg_relevance_score']:.1f}/5")
    
    # Breakdown by hop count
    print(f"\n📋 By Expected Hop Count:")
    for hops in [1, 2]:
        hop_results = [r for r in results if r["expected_hops"] == hops]
        if hop_results:
            hop_pass = sum(1 for r in hop_results if r["pass"])
            print(f"   {hops}-hop questions: {hop_pass}/{len(hop_results)} passed")
    
    # Show common missed connections
    all_missed = []
    for r in results:
        all_missed.extend(r.get("missed_connections", []))
    
    if all_missed:
        from collections import Counter
        missed_counts = Counter(all_missed)
        print(f"\n⚠️  Commonly Missed Connections:")
        for conn, count in missed_counts.most_common(5):
            print(f"   • {conn} (missed {count}x)")
    
    return summary


def main():
    """Run the multi-hop reasoning evaluation."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate multi-hop reasoning in GraphRAG")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show all related entities")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--question", type=str, help="Test a single custom question")
    
    args = parser.parse_args()
    
    if args.question:
        print(f"\nTesting multi-hop reasoning for question...")
        print(f"Q: {args.question}")
        print("-" * 50)
        
        # Connect to Neo4j
        try:
            neo4j = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            return
        
        try:
            # Extract entities
            entities = extract_query_entities(args.question)
            print(f"\nExtracted entities: {entities}")
            
            # Find related
            related = find_related_entities(neo4j, entities, max_hops=2)
            
            print(f"\nRelated entities found ({len(related)}):")
            for entity, distance in sorted(related.items(), key=lambda x: x[1])[:15]:
                print(f"  {distance} hop{'s' if distance > 1 else ''}: {entity}")
            
            # Judge relevance
            judgment = judge_path_relevance(args.question, entities, related)
            
            print(f"\n📊 Path Relevance:")
            print(f"   Relevant: {judgment.get('path_relevant', False)}")
            print(f"   Score: {judgment.get('relevance_score', 0)}/5")
            print(f"   Reasoning: {judgment.get('reasoning', 'N/A')}")
            print(f"   Useful entities: {judgment.get('useful_entities', [])}")
            
        finally:
            neo4j.close()
    else:
        summary = run_evaluation(verbose=args.verbose)
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
