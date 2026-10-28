"""
GraphRAG - Generation Module
Uses graph-enhanced retrieval to generate answers with richer context.
"""

import os
from dotenv import load_dotenv

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # sandbox
    from llm_shim import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieval import graph_enhanced_retrieval, format_retrieved_context

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM Configuration
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
TOP_K = 5

# RAG Prompt Template with graph context
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on Warren Buffett's annual shareholder letters.

Search Method: Graph-Enhanced RAG (combining knowledge graph traversal with semantic search)

The context below was retrieved by:
1. Identifying key entities in your question
2. Traversing a knowledge graph to find related concepts
3. Combining graph results with semantic search

{entity_context}

Use ONLY the information from the context below to answer the question. If the context doesn't contain enough information, acknowledge what you can answer and what's missing.

Context:
{context}

Question: {question}

Answer:"""


def create_rag_chain():
    """Create the RAG chain."""
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )
    
    return prompt | llm | StrOutputParser()


def build_entity_context(query_entities: list, related_entities: dict) -> str:
    """Build a description of the entities found."""
    if not query_entities and not related_entities:
        return ""
    
    parts = []
    
    if query_entities:
        parts.append(f"Key entities in your question: {', '.join(query_entities)}")
    
    if related_entities:
        # Group by distance
        by_distance = {}
        for entity, distance in related_entities.items():
            if distance not in by_distance:
                by_distance[distance] = []
            by_distance[distance].append(entity)
        
        for distance in sorted(by_distance.keys()):
            entities = by_distance[distance][:5]  # Limit per level
            parts.append(f"Related entities ({distance} hop{'s' if distance > 1 else ''} away): {', '.join(entities)}")
    
    return "\n".join(parts)


def generate_answer(
    question: str,
    top_k: int = TOP_K,
    graph_weight: float = 0.4,
    vector_weight: float = 0.6,
    verbose: bool = False,
) -> dict:
    """
    Generate an answer using graph-enhanced RAG.
    
    Args:
        question: User's question
        top_k: Number of documents to retrieve
        graph_weight: Weight for graph-based results
        vector_weight: Weight for vector search results
        verbose: Include retrieval details in response
    
    Returns:
        Dictionary with answer and metadata
    """
    # Perform graph-enhanced retrieval
    retrieval_result = graph_enhanced_retrieval(
        query=question,
        k=top_k,
        graph_weight=graph_weight,
        vector_weight=vector_weight,
        verbose=verbose
    )
    
    documents = retrieval_result["combined_documents"]
    
    if not documents:
        return {
            "answer": "I couldn't find any relevant information.",
            "sources": [],
            "query_entities": [],
            "related_entities": {},
            "search_method": "GraphRAG"
        }
    
    # Build context
    context = format_retrieved_context(documents)
    entity_context = build_entity_context(
        retrieval_result["query_entities"],
        retrieval_result["related_entities"]
    )
    
    # Generate answer
    chain = create_rag_chain()
    answer = chain.invoke({
        "context": context,
        "question": question,
        "entity_context": entity_context
    })
    
    response = {
        "answer": answer,
        "sources": [
            {
                "file": doc.metadata.get("source_file", "Unknown"),
                "year": doc.metadata.get("year", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "retrieval_method": doc.metadata.get("retrieval_method", "unknown")
            }
            for doc in documents
        ],
        "query_entities": retrieval_result["query_entities"],
        "related_entities": dict(list(retrieval_result["related_entities"].items())[:10]),
        "search_method": "GraphRAG"
    }
    
    if verbose:
        response["graph_doc_count"] = len(retrieval_result["graph_documents"])
        response["vector_doc_count"] = len(retrieval_result["vector_documents"])
    
    return response


def interactive_mode():
    """Run interactive Q&A with graph-enhanced retrieval."""
    print("\n" + "=" * 60)
    print("GraphRAG - Interactive Q&A")
    print("=" * 60)
    print("\n🔗 Using knowledge graph + vector search")
    print("\nCommands:")
    print("  weights:0.5,0.5    - Set graph/vector weights")
    print("  quit               - Exit")
    print("-" * 60)
    
    graph_weight = 0.4
    vector_weight = 0.6
    
    while True:
        print()
        print(f"⚖️  Weights: Graph={graph_weight}, Vector={vector_weight}")
        user_input = input(">> ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if user_input.startswith('weights:'):
            try:
                weights = user_input.split(':')[1].split(',')
                graph_weight = float(weights[0])
                vector_weight = float(weights[1])
                print(f"✅ Weights updated: Graph={graph_weight}, Vector={vector_weight}")
            except Exception:
                print("❌ Invalid format. Use 'weights:0.5,0.5'")
            continue
        
        print("\n🔗 Traversing knowledge graph...")
        print("🔢 Running vector search...")
        print("🤖 Generating answer...\n")
        
        try:
            result = generate_answer(
                user_input,
                graph_weight=graph_weight,
                vector_weight=vector_weight,
                verbose=True
            )
            
            # Show entities found
            if result["query_entities"]:
                print(f"🏷️  Query entities: {', '.join(result['query_entities'])}")
            if result["related_entities"]:
                related = list(result["related_entities"].keys())[:5]
                print(f"🔗 Related entities: {', '.join(related)}")
            
            print("\n" + "-" * 50)
            print("Answer:")
            print("-" * 50)
            print(result["answer"])
            
            print(f"\n📚 Sources ({len(result['sources'])} documents):")
            for s in result["sources"]:
                method = s['retrieval_method']
                icon = "🔗" if method == "graph" else "🔢"
                print(f"  {icon} {s['file']} (Year: {s['year']}) [{method}]")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Run example queries or interactive mode."""
    print("=" * 60)
    print("GraphRAG - Graph-Enhanced Generation")
    print("=" * 60)
    print("\n🔗 Combines knowledge graph traversal with vector search")
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    examples = [
        {
            "question": "How does GEICO contribute to Berkshire's overall business?",
            "note": "Graph helps find GEICO → Insurance → Float connections"
        },
        {
            "question": "What's the relationship between Buffett and Apple?",
            "note": "Graph connects Buffett → Berkshire → Apple"
        },
        {
            "question": "How does insurance float work and why is it valuable?",
            "note": "Graph expands 'float' to related insurance concepts"
        },
    ]
    
    print("\n📋 Example Queries:")
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. {ex['question'][:45]}...")
        print(f"     💡 {ex['note']}")
    
    print("\n" + "-" * 50)
    choice = input("Enter 1-3 for examples, 'i' for interactive, or your question: ").strip()
    
    if choice.lower() == 'i':
        interactive_mode()
    elif choice in ['1', '2', '3']:
        ex = examples[int(choice) - 1]
        print(f"\n📝 Question: {ex['question']}")
        print(f"💡 Note: {ex['note']}")
        
        print("\n🔗 Traversing knowledge graph...")
        print("🔢 Running vector search...")
        print("🤖 Generating answer...\n")
        
        result = generate_answer(ex['question'], verbose=True)
        
        if result["query_entities"]:
            print(f"🏷️  Query entities: {', '.join(result['query_entities'])}")
        if result["related_entities"]:
            related = list(result["related_entities"].keys())[:5]
            print(f"🔗 Related entities: {', '.join(related)}")
        
        print("\n" + "-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])
        
        print(f"\n📚 Sources:")
        for s in result["sources"]:
            method = s['retrieval_method']
            icon = "🔗" if method == "graph" else "🔢"
            print(f"  {icon} {s['file']} (Year: {s['year']}) [{method}]")
    elif choice:
        print(f"\n📝 Question: {choice}")
        
        result = generate_answer(choice, verbose=True)
        
        print("\n" + "-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])
        
        print(f"\n📚 Sources:")
        for s in result["sources"]:
            print(f"  • {s['file']} (Year: {s['year']})")
    else:
        print("\n👋 No input. Run again to try!")


if __name__ == "__main__":
    main()
