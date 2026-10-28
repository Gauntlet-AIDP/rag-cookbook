"""
GraphRAG - Knowledge Graph Builder
Extracts entities and relationships from MongoDB chunks and builds a Neo4j graph.

Reuses existing vectorized documents from MongoDB.
"""

import os
import json
import certifi
from dotenv import load_dotenv
from typing import Optional

try:
    from langchain_openai import ChatOpenAI
    SANDBOX = False
except ImportError:  # sandbox
    from llm_shim import ChatOpenAI
    SANDBOX = True
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

# MongoDB configuration (reuse existing collection)
DB_NAME = "rag_playbook"
COLLECTION_NAME = "metadata_filtered_rag"  # or "naive_rag"

# LLM for entity extraction
EXTRACTION_MODEL = "gpt-4o-mini"

# Entity extraction prompt
ENTITY_EXTRACTION_PROMPT = """Extract entities and relationships from this text about Warren Buffett's shareholder letters.

Text:
{text}

Source: {source_file} (Year: {year})

Extract:
1. ENTITIES: Important named entities (companies, people, concepts, financial terms)
2. RELATIONSHIPS: How entities relate to each other

Return a JSON object with this exact structure:
{{
  "entities": [
    {{"name": "entity name (lowercase)", "type": "COMPANY|PERSON|CONCEPT|FINANCIAL_TERM"}},
    ...
  ],
  "relationships": [
    {{"source": "entity1", "target": "entity2", "type": "RELATIONSHIP_TYPE"}},
    ...
  ]
}}

Relationship types to use:
- INVESTS_IN (for investments)
- OWNS (for ownership/subsidiaries)
- MANAGES (for management)
- RELATED_TO (general relationship)
- COMPETES_WITH (competitors)
- PARTNERS_WITH (partnerships)

Focus on the most important entities and relationships. Return ONLY the JSON, no other text."""


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
    
    def execute_write(self, query: str, parameters: dict = None):
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run(query, parameters or {}))


def get_mongo_client():
    """Get MongoDB client."""
    return MongoClient(MONGO_DB_URL, **MONGO_KWARGS)


def load_chunks_from_mongo(limit: Optional[int] = None) -> list[dict]:
    """Load document chunks from MongoDB."""
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    try:
        cursor = collection.find()
        if limit:
            cursor = cursor.limit(limit)
        
        chunks = []
        for doc in cursor:
            text = doc.get("text", "")
            if not text or len(text) < 100:  # Skip very short chunks
                continue
            
            chunks.append({
                "id": str(doc.get("_id")),
                "text": text,
                "source_file": doc.get("source_file", "Unknown"),
                "year": doc.get("year", 0),
                "page": doc.get("page", 0),
            })
        
        return chunks
    finally:
        client.close()


def create_extraction_chain():
    """Create LLM chain for entity/relationship extraction."""
    prompt = ChatPromptTemplate.from_template(ENTITY_EXTRACTION_PROMPT)
    
    llm = ChatOpenAI(
        model=EXTRACTION_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )
    
    return prompt | llm | StrOutputParser()


def extract_entities_and_relationships(chain, chunk: dict) -> dict:
    """Extract entities and relationships from a chunk."""
    try:
        response = chain.invoke({
            "text": chunk["text"][:3000],  # Limit text length
            "source_file": chunk["source_file"],
            "year": chunk["year"]
        })
        
        # Parse JSON response
        # Clean up response if needed
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        result = json.loads(response)
        return {
            "entities": result.get("entities", []),
            "relationships": result.get("relationships", []),
            "chunk_id": chunk["id"],
            "source_file": chunk["source_file"],
            "year": chunk["year"]
        }
    except Exception as e:
        print(f"   Warning: Extraction failed: {e}")
        return {
            "entities": [],
            "relationships": [],
            "chunk_id": chunk["id"],
            "source_file": chunk["source_file"],
            "year": chunk["year"]
        }


def setup_neo4j_schema(neo4j: Neo4jConnection):
    """Create indexes and constraints in Neo4j."""
    print("📊 Setting up Neo4j schema...")
    
    # Create constraints for unique entity names
    constraints = [
        "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        "CREATE CONSTRAINT document_name IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT year_value IF NOT EXISTS FOR (y:Year) REQUIRE y.value IS UNIQUE",
    ]
    
    for constraint in constraints:
        try:
            neo4j.execute_write(constraint)
        except Exception as e:
            # Constraint might already exist
            pass
    
    print("   Schema ready")


def clear_graph(neo4j: Neo4jConnection):
    """Clear all nodes and relationships from the graph."""
    print("🗑️  Clearing existing graph...")
    neo4j.execute_write("MATCH (n) DETACH DELETE n")
    print("   Graph cleared")


def build_graph(neo4j: Neo4jConnection, extractions: list[dict]):
    """Build the knowledge graph from extracted entities and relationships."""
    print("\n🔨 Building knowledge graph...")
    
    entity_count = 0
    relationship_count = 0
    document_count = 0
    
    for extraction in extractions:
        source_file = extraction["source_file"]
        year = extraction["year"]
        chunk_id = extraction["chunk_id"]
        
        # Create Document node
        neo4j.execute_write("""
            MERGE (d:Document {name: $name})
            SET d.chunk_id = $chunk_id
        """, {"name": source_file, "chunk_id": chunk_id})
        document_count += 1
        
        # Create Year node and link to document
        if year and year > 0:
            neo4j.execute_write("""
                MERGE (y:Year {value: $year})
                WITH y
                MATCH (d:Document {name: $doc_name})
                MERGE (d)-[:FROM_YEAR]->(y)
            """, {"year": year, "doc_name": source_file})
        
        # Create Entity nodes and link to document
        for entity in extraction["entities"]:
            entity_name = entity.get("name", "").lower().strip()
            entity_type = entity.get("type", "CONCEPT")
            
            if not entity_name or len(entity_name) < 2:
                continue
            
            neo4j.execute_write("""
                MERGE (e:Entity {name: $name})
                SET e.type = $type
                WITH e
                MATCH (d:Document {name: $doc_name})
                MERGE (d)-[:MENTIONS]->(e)
            """, {
                "name": entity_name,
                "type": entity_type,
                "doc_name": source_file
            })
            entity_count += 1
        
        # Create relationships between entities
        for rel in extraction["relationships"]:
            source = rel.get("source", "").lower().strip()
            target = rel.get("target", "").lower().strip()
            rel_type = rel.get("type", "RELATED_TO").upper().replace(" ", "_")
            
            if not source or not target or source == target:
                continue
            
            # Use RELATED_TO as a safe fallback, store original type as property
            neo4j.execute_write("""
                MATCH (s:Entity {name: $source})
                MATCH (t:Entity {name: $target})
                MERGE (s)-[r:RELATED_TO]->(t)
                SET r.relationship_type = $rel_type
            """, {
                "source": source,
                "target": target,
                "rel_type": rel_type
            })
            relationship_count += 1
    
    print(f"   Created {entity_count} entity mentions")
    print(f"   Created {relationship_count} relationships")
    print(f"   Linked to {document_count} document chunks")


def add_topic_nodes(neo4j: Neo4jConnection):
    """Add high-level topic nodes based on entity patterns."""
    print("\n📌 Adding topic nodes...")
    
    topics = {
        "Insurance": ["insurance", "geico", "float", "underwriting", "reinsurance", "policyholder"],
        "Investments": ["investment", "stock", "portfolio", "equity", "dividend", "shares"],
        "Acquisitions": ["acquisition", "acquire", "merger", "buyout", "purchase"],
        "Management": ["management", "manager", "ceo", "leadership", "culture"],
        "Banking": ["bank", "banking", "financial", "credit", "lending"],
        "Technology": ["technology", "tech", "software", "apple", "ibm"],
        "Consumer": ["consumer", "retail", "coca-cola", "see's candies", "dairy queen"],
        "Energy": ["energy", "utilities", "oil", "gas", "chevron", "occidental"],
    }
    
    for topic_name, keywords in topics.items():
        # Create topic node
        neo4j.execute_write("""
            MERGE (t:Topic {name: $name})
        """, {"name": topic_name})
        
        # Link entities to topics
        for keyword in keywords:
            neo4j.execute_write("""
                MATCH (e:Entity)
                WHERE e.name CONTAINS $keyword
                MATCH (t:Topic {name: $topic_name})
                MERGE (e)-[:BELONGS_TO]->(t)
            """, {"keyword": keyword, "topic_name": topic_name})
    
    print(f"   Created {len(topics)} topic nodes")


def print_graph_stats(neo4j: Neo4jConnection):
    """Print statistics about the graph."""
    print("\n📊 Graph Statistics:")
    
    # Count nodes by type
    node_counts = neo4j.execute_query("""
        MATCH (n)
        RETURN labels(n)[0] as label, count(n) as count
        ORDER BY count DESC
    """)
    
    print("   Nodes:")
    for row in node_counts:
        print(f"      {row['label']}: {row['count']}")
    
    # Count relationships
    rel_count = neo4j.execute_query("""
        MATCH ()-[r]->()
        RETURN count(r) as count
    """)
    print(f"   Relationships: {rel_count[0]['count']}")
    
    # Top entities by connections
    top_entities = neo4j.execute_query("""
        MATCH (e:Entity)-[r]-()
        RETURN e.name as entity, count(r) as connections
        ORDER BY connections DESC
        LIMIT 10
    """)
    
    print("\n   Top 10 connected entities:")
    for row in top_entities:
        print(f"      {row['entity']}: {row['connections']} connections")


def main():
    """Build the knowledge graph from MongoDB chunks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build knowledge graph from MongoDB chunks")
    parser.add_argument(
        "-n", "--num-chunks", 
        type=int, 
        default=None,
        help="Number of chunks to process (default: all). Use 3-5 for quick testing."
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Don't clear existing graph (append to it)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GraphRAG - Knowledge Graph Builder")
    print("=" * 60)
    print(f"\n📌 Source: MongoDB {DB_NAME}.{COLLECTION_NAME}")
    print(f"📌 Target: Neo4j {NEO4J_URI}")
    
    if args.num_chunks:
        print(f"📌 Limit: {args.num_chunks} chunks (quick mode)")
    
    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Connect to Neo4j
    print("\n🔌 Connecting to Neo4j...")
    try:
        neo4j = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        # Test connection
        neo4j.execute_query("RETURN 1")
        print("   Connected!")
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        print("\n   Make sure Neo4j is running:")
        print("   cd graph-rag && docker-compose up -d")
        return
    
    try:
        # Setup schema
        setup_neo4j_schema(neo4j)
        
        # Clear existing graph (unless --no-clear)
        if not args.no_clear:
            clear_graph(neo4j)
        else:
            print("📌 Keeping existing graph (append mode)")
        
        # Load chunks from MongoDB
        print("\n📚 Loading chunks from MongoDB...")
        chunks = load_chunks_from_mongo(limit=args.num_chunks)
        print(f"   Loaded {len(chunks)} chunks")
        
        if not chunks:
            print("❌ No chunks found. Run ingestion first!")
            return
        
        # Create extraction chain
        print("\n🤖 Initializing entity extractor...")
        chain = create_extraction_chain()
        
        # Extract entities and relationships from each chunk
        print("\n🔍 Extracting entities and relationships...")
        extractions = []
        
        for i, chunk in enumerate(chunks):
            print(f"   [{i + 1}/{len(chunks)}] Processing {chunk['source_file']}...")
            
            extraction = extract_entities_and_relationships(chain, chunk)
            extractions.append(extraction)
            
            # Show what was extracted
            entity_count = len(extraction.get("entities", []))
            rel_count = len(extraction.get("relationships", []))
            print(f"            Found {entity_count} entities, {rel_count} relationships")
        
        print(f"\n   Processed {len(extractions)} chunks")
        
        # Build the graph
        build_graph(neo4j, extractions)
        
        # Add topic nodes
        add_topic_nodes(neo4j)
        
        # Print statistics
        print_graph_stats(neo4j)
        
        print("\n✅ Knowledge graph built successfully!")
        if SANDBOX:
            print("\n   Sandbox: no browser port; run cypher-shell via docker compose exec neo4j")
            print("   Login: neo4j / $NEO4J_PASSWORD from .env")
        else:
            print("\n   View in Neo4j Browser: http://localhost:7474")
            print("   Login: neo4j / password123")
        print("\n   Try this Cypher query:")
        print("   MATCH (e:Entity)-[r]-(connected) RETURN e, r, connected LIMIT 50")
        
    finally:
        neo4j.close()


if __name__ == "__main__":
    main()
