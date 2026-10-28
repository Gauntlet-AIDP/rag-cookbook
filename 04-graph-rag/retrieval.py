"""
GraphRAG - Graph-Enhanced Retrieval
Combines knowledge graph traversal with vector search for richer context.

Retrieval Strategy:
1. Extract entities from the query
2. Find related entities via graph traversal (1-2 hops)
3. Get document chunks that mention these entities
4. Also run vector search
5. Combine and rerank results
"""

import os
import certifi
from dotenv import load_dotenv
from collections import defaultdict

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:  # sandbox
    from llm_shim import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from pymongo import MongoClient
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# Configuration
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
# Atlas (mongodb+srv, mongodb.net or tls/ssl=true) needs the certifi CA bundle; a plain local URL does not
_TLS_HINTS = ("mongodb+srv", "mongodb.net", "tls=true", "ssl=true")
MONGO_KWARGS = {"tlsCAFile": certifi.where()} if any(h in (MONGO_DB_URL or "").lower() for h in _TLS_HINTS) else {}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

# MongoDB configuration
DB_NAME = "rag_playbook"
COLLECTION_NAME = "metadata_filtered_rag"
INDEX_NAME = "metadata_filtered_index"

# Retrieval configuration
DEFAULT_TOP_K = 5
GRAPH_HOP_DEPTH = 2  # How many hops to traverse in the graph
GRAPH_WEIGHT = 0.4   # Weight for graph-based results
VECTOR_WEIGHT = 0.6  # Weight for vector search results

# Query entity extraction prompt
QUERY_ENTITY_PROMPT = """Extract the key entities from this question about Warren Buffett's shareholder letters.

Question: {question}

Return a comma-separated list of entity names (lowercase), focusing on:
- Company names (e.g., "apple", "geico", "berkshire")
- People (e.g., "buffett", "munger")
- Concepts (e.g., "insurance", "float", "acquisitions")
- Financial terms (e.g., "dividends", "earnings")

Return ONLY the comma-separated list, nothing else. If no clear entities, return "general"."""


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


def get_mongo_client():
    """Get MongoDB client."""
    return MongoClient(MONGO_DB_URL, **MONGO_KWARGS)


def get_vector_store():
    """Connect to MongoDB vector store."""
    client = get_mongo_client()
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


def extract_query_entities(question: str) -> list[str]:
    """Extract entities from the user's question using LLM."""
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


def find_related_entities(neo4j: Neo4jConnection, entities: list[str], max_hops: int = GRAPH_HOP_DEPTH) -> dict:
    """
    Find entities related to the query entities via graph traversal.
    
    Returns:
        Dictionary mapping entity names to their hop distance from query entities
    """
    if not entities:
        return {}
    
    related = {}
    
    for entity in entities:
        # Find entities within max_hops
        results = neo4j.execute_query("""
            MATCH (start:Entity)
            WHERE start.name CONTAINS $entity_name
            MATCH path = (start)-[*1..""" + str(max_hops) + """]-(related:Entity)
            RETURN DISTINCT related.name as name, length(path) as distance
            ORDER BY distance
            LIMIT 20
        """, {"entity_name": entity})
        
        for row in results:
            name = row["name"]
            distance = row["distance"]
            # Keep the shortest distance
            if name not in related or distance < related[name]:
                related[name] = distance
    
    return related


def find_documents_for_entities(neo4j: Neo4jConnection, entities: list[str]) -> list[str]:
    """Find document names that mention any of the given entities."""
    if not entities:
        return []
    
    results = neo4j.execute_query("""
        MATCH (d:Document)-[:MENTIONS]->(e:Entity)
        WHERE e.name IN $entities
        RETURN DISTINCT d.name as doc_name, count(e) as entity_count
        ORDER BY entity_count DESC
        LIMIT 20
    """, {"entities": entities})
    
    return [row["doc_name"] for row in results]


def get_chunks_by_source(source_files: list[str], limit: int = 10) -> list[Document]:
    """Get chunks from MongoDB by source file names."""
    if not source_files:
        return []
    
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    try:
        chunks = []
        for doc in collection.find({"source_file": {"$in": source_files}}).limit(limit):
            text = doc.get("text", "")
            if not text:
                continue
            
            chunks.append(Document(
                page_content=text,
                metadata={
                    "source_file": doc.get("source_file", "Unknown"),
                    "year": doc.get("year", 0),
                    "page": doc.get("page", 0),
                    "retrieval_method": "graph"
                }
            ))
        
        return chunks
    finally:
        client.close()


def vector_search(query: str, k: int = DEFAULT_TOP_K) -> list[Document]:
    """Perform vector similarity search."""
    vector_store, client = get_vector_store()
    
    try:
        results = vector_store.similarity_search(query=query, k=k)
        
        # Add retrieval method metadata
        for doc in results:
            doc.metadata["retrieval_method"] = "vector"
        
        return results
    finally:
        client.close()


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    weights: list[float],
    k: int = 60
) -> list[Document]:
    """Combine ranked lists using Reciprocal Rank Fusion."""
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    doc_scores = defaultdict(float)
    doc_map = {}
    
    for list_idx, doc_list in enumerate(result_lists):
        weight = weights[list_idx]
        for rank, doc in enumerate(doc_list):
            doc_key = hash(doc.page_content)
            rrf_score = weight / (k + rank + 1)
            doc_scores[doc_key] += rrf_score
            
            if doc_key not in doc_map:
                doc_map[doc_key] = doc
    
    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]


def graph_enhanced_retrieval(
    query: str,
    k: int = DEFAULT_TOP_K,
    graph_weight: float = GRAPH_WEIGHT,
    vector_weight: float = VECTOR_WEIGHT,
    verbose: bool = False
) -> dict:
    """
    Perform graph-enhanced retrieval.
    
    Steps:
    1. Extract entities from query
    2. Traverse graph to find related entities
    3. Get chunks mentioning these entities
    4. Run vector search
    5. Combine results with RRF
    
    Returns:
        Dictionary with documents and metadata about the retrieval process
    """
    result = {
        "query": query,
        "query_entities": [],
        "related_entities": {},
        "graph_documents": [],
        "vector_documents": [],
        "combined_documents": [],
    }
    
    # Connect to Neo4j
    try:
        neo4j = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    except Exception as e:
        if verbose:
            print(f"⚠️  Neo4j not available: {e}")
            print("   Falling back to vector-only search")
        # Fallback to vector-only
        result["vector_documents"] = vector_search(query, k=k)
        result["combined_documents"] = result["vector_documents"]
        return result
    
    try:
        # Step 1: Extract entities from query
        if verbose:
            print("🔍 Extracting query entities...")
        query_entities = extract_query_entities(query)
        result["query_entities"] = query_entities
        
        if verbose:
            print(f"   Found entities: {query_entities}")
        
        # Step 2: Graph traversal to find related entities
        if verbose:
            print("🔗 Traversing graph for related entities...")
        related_entities = find_related_entities(neo4j, query_entities)
        result["related_entities"] = related_entities
        
        if verbose:
            print(f"   Found {len(related_entities)} related entities")
        
        # Step 3: Find documents mentioning these entities
        all_entities = query_entities + list(related_entities.keys())
        
        if verbose:
            print("📄 Finding documents via graph...")
        doc_names = find_documents_for_entities(neo4j, all_entities)
        graph_docs = get_chunks_by_source(doc_names, limit=k * 2)
        result["graph_documents"] = graph_docs
        
        if verbose:
            print(f"   Found {len(graph_docs)} chunks via graph")
        
        # Step 4: Vector search
        if verbose:
            print("🔢 Running vector search...")
        vector_docs = vector_search(query, k=k * 2)
        result["vector_documents"] = vector_docs
        
        if verbose:
            print(f"   Found {len(vector_docs)} chunks via vector search")
        
        # Step 5: Combine with RRF
        if verbose:
            print("🔀 Combining results...")
        
        if graph_docs and vector_docs:
            combined = reciprocal_rank_fusion(
                [graph_docs, vector_docs],
                [graph_weight, vector_weight]
            )
        elif graph_docs:
            combined = graph_docs
        else:
            combined = vector_docs
        
        result["combined_documents"] = combined[:k]
        
        if verbose:
            print(f"   Final: {len(result['combined_documents'])} documents")
        
        return result
        
    finally:
        neo4j.close()


def format_retrieved_context(documents: list) -> str:
    """Format retrieved documents into context string."""
    context_parts = []
    
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_file", "Unknown")
        year = doc.metadata.get("year", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        method = doc.metadata.get("retrieval_method", "unknown")
        
        context_parts.append(
            f"[Document {i}] (via {method})\n"
            f"Source: {source} (Year: {year}, Page: {page})\n"
            f"Content:\n{doc.page_content}\n"
        )
    
    return "\n---\n".join(context_parts)


def main():
    """Test graph-enhanced retrieval."""
    print("=" * 60)
    print("GraphRAG - Graph-Enhanced Retrieval Test")
    print("=" * 60)
    
    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Test queries
    test_queries = [
        "How does GEICO contribute to Berkshire's insurance operations?",
        "What is Warren Buffett's view on Apple as an investment?",
        "How does insurance float work?",
    ]
    
    print("\n" + "=" * 60)
    print("Running Graph-Enhanced Retrieval Tests")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 50)
        
        result = graph_enhanced_retrieval(query, k=3, verbose=True)
        
        print(f"\n📊 Results:")
        print(f"   Query entities: {result['query_entities']}")
        print(f"   Related entities: {list(result['related_entities'].keys())[:5]}...")
        print(f"   Graph docs: {len(result['graph_documents'])}")
        print(f"   Vector docs: {len(result['vector_documents'])}")
        print(f"   Combined docs: {len(result['combined_documents'])}")
        
        print(f"\n📚 Top documents:")
        for doc in result["combined_documents"][:3]:
            method = doc.metadata.get("retrieval_method", "?")
            source = doc.metadata.get("source_file", "?")
            print(f"   [{method}] {source}")


if __name__ == "__main__":
    main()
