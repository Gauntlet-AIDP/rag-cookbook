"""
Hybrid Search RAG - Generation Module
Uses hybrid retrieval (BM25 + Vector) to generate answers.

NO INGESTION REQUIRED - Reuses existing vectorized documents.
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
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Configuration
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
# Atlas (mongodb+srv, mongodb.net or tls/ssl=true) needs the certifi CA bundle; a plain local URL does not
_TLS_HINTS = ("mongodb+srv", "mongodb.net", "tls=true", "ssl=true")
MONGO_KWARGS = {"tlsCAFile": certifi.where()} if any(h in (MONGO_DB_URL or "").lower() for h in _TLS_HINTS) else {}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB configuration (reusing existing collection)
DB_NAME = "rag_playbook"
COLLECTION_NAME = "metadata_filtered_rag"  # or "naive_rag"
INDEX_NAME = "metadata_filtered_index"  # or "naive"

# LLM Configuration
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
TOP_K = 5

# Default hybrid weights
DEFAULT_BM25_WEIGHT = 0.5
DEFAULT_VECTOR_WEIGHT = 0.5
RRF_K = 60

# RAG Prompt Template
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on Warren Buffett's annual shareholder letters.

Search Method: Hybrid (combining keyword and semantic search)

Use ONLY the information from the context below to answer the question. If the context doesn't contain enough information, acknowledge what you can answer and what's missing.

Context:
{context}

Question: {question}

Answer:"""

# Global cache for BM25 (expensive to rebuild)
_bm25_cache = {
    "documents": None,
}


def get_mongo_client():
    """Get MongoDB client."""
    return MongoClient(MONGO_DB_URL, **MONGO_KWARGS)


def load_documents_for_bm25() -> list[Document]:
    """Load documents from MongoDB for BM25 indexing."""
    if _bm25_cache["documents"] is not None:
        return _bm25_cache["documents"]
    
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    try:
        documents = []
        for doc in collection.find():
            text = doc.get("text", "")
            if not text:
                continue
            
            metadata = {
                "source_file": doc.get("source_file", "Unknown"),
                "year": doc.get("year", 0),
                "page": doc.get("page", 0),
            }
            
            if "topic_buckets" in doc:
                metadata["topic_buckets"] = doc["topic_buckets"]
            if "companies_mentioned" in doc:
                metadata["companies_mentioned"] = doc["companies_mentioned"]
            
            documents.append(Document(
                page_content=text,
                metadata=metadata
            ))
        
        _bm25_cache["documents"] = documents
        return documents
    finally:
        client.close()


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    weights: list[float],
    k: int = RRF_K
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


def hybrid_search(
    query: str,
    k: int = TOP_K,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
) -> list[Document]:
    """Perform hybrid search combining BM25 and vector search."""
    documents = load_documents_for_bm25()
    
    if not documents:
        raise ValueError("No documents found. Run ingestion first!")
    
    # BM25 search
    bm25_retriever = BM25Retriever.from_documents(documents, k=k * 3)
    bm25_results = bm25_retriever.invoke(query)
    
    # Vector search
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
    
    try:
        vector_results = vector_store.similarity_search(query=query, k=k * 3)
    finally:
        client.close()
    
    # Combine with RRF
    combined = reciprocal_rank_fusion(
        result_lists=[bm25_results, vector_results],
        weights=[bm25_weight, vector_weight]
    )
    
    return combined[:k]


def format_context(documents: list) -> str:
    """Format documents into context string."""
    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_file", "Unknown")
        year = doc.metadata.get("year", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        
        parts.append(
            f"[Document {i}]\n"
            f"Source: {source} (Year: {year}, Page: {page})\n"
            f"Content:\n{doc.page_content}\n"
        )
    return "\n---\n".join(parts)


def create_rag_chain():
    """Create the RAG chain."""
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )
    
    return prompt | llm | StrOutputParser()


def generate_answer(
    question: str,
    top_k: int = TOP_K,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    verbose: bool = False,
) -> dict:
    """Generate an answer using hybrid RAG."""
    documents = hybrid_search(
        query=question,
        k=top_k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight
    )
    
    if not documents:
        return {
            "answer": "I couldn't find any relevant information.",
            "sources": [],
            "search_method": f"Hybrid (BM25: {bm25_weight}, Vector: {vector_weight})"
        }
    
    context = format_context(documents)
    chain = create_rag_chain()
    answer = chain.invoke({
        "context": context,
        "question": question
    })
    
    response = {
        "answer": answer,
        "sources": [
            {
                "file": doc.metadata.get("source_file", "Unknown"),
                "year": doc.metadata.get("year", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
            }
            for doc in documents
        ],
        "search_method": f"Hybrid (BM25: {bm25_weight}, Vector: {vector_weight})"
    }
    
    if verbose:
        response["retrieved_documents"] = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]
    
    return response


def interactive_mode():
    """Run interactive Q&A with hybrid search."""
    print("\n" + "=" * 60)
    print("Hybrid Search RAG - Interactive Q&A")
    print("=" * 60)
    print("\n📌 Using existing documents (no ingestion needed)")
    print(f"   Collection: {DB_NAME}.{COLLECTION_NAME}")
    print("\nCommands:")
    print("  weights:0.7,0.3    - Set BM25/Vector weights")
    print("  quit               - Exit")
    print("-" * 60)
    
    print("\n🔄 Loading documents for BM25 index...")
    docs = load_documents_for_bm25()
    print(f"   Loaded {len(docs)} documents")
    
    bm25_weight = DEFAULT_BM25_WEIGHT
    vector_weight = DEFAULT_VECTOR_WEIGHT
    
    while True:
        print()
        print(f"⚖️  Current weights: BM25={bm25_weight}, Vector={vector_weight}")
        user_input = input(">> ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if user_input.startswith('weights:'):
            try:
                weights = user_input.split(':')[1].split(',')
                bm25_weight = float(weights[0])
                vector_weight = float(weights[1])
                print(f"✅ Weights updated: BM25={bm25_weight}, Vector={vector_weight}")
            except Exception:
                print("❌ Invalid format. Use 'weights:0.7,0.3'")
            continue
        
        print("\n🔀 Searching with hybrid retrieval...")
        print("🤖 Generating answer...\n")
        
        try:
            result = generate_answer(
                user_input,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight
            )
            
            print("-" * 50)
            print("Answer:")
            print("-" * 50)
            print(result["answer"])
            
            print(f"\n📚 Sources ({len(result['sources'])} documents):")
            for s in result["sources"]:
                print(f"  • {s['file']} (Year: {s['year']}, Page: {s['page']})")
            
            print(f"\n🔀 {result['search_method']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Run example queries or interactive mode."""
    print("=" * 60)
    print("Hybrid Search RAG - Generation")
    print("=" * 60)
    print("\n📌 NO INGESTION REQUIRED")
    print(f"   Reusing collection: {DB_NAME}.{COLLECTION_NAME}")
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    examples = [
        {
            "question": "What is GEICO and how does it contribute to Berkshire?",
            "note": "BM25 helps with 'GEICO' exact match"
        },
        {
            "question": "What makes a great long-term investment?",
            "note": "Vector helps with semantic meaning"
        },
        {
            "question": "Berkshire float insurance 2015",
            "note": "Hybrid combines terms + meaning"
        },
    ]
    
    print("\n📋 Example Queries:")
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. {ex['question'][:45]}... ({ex['note']})")
    
    print("\n" + "-" * 50)
    choice = input("Enter 1-3 for examples, 'i' for interactive, or your question: ").strip()
    
    if choice.lower() == 'i':
        interactive_mode()
    elif choice in ['1', '2', '3']:
        ex = examples[int(choice) - 1]
        print(f"\n📝 Question: {ex['question']}")
        print(f"💡 Note: {ex['note']}")
        
        print("\n🔀 Retrieving with hybrid search...")
        print("🤖 Generating answer...\n")
        
        result = generate_answer(ex['question'])
        
        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])
        
        print(f"\n📚 Sources:")
        for s in result["sources"]:
            print(f"  • {s['file']} (Year: {s['year']})")
    elif choice:
        print(f"\n📝 Question: {choice}")
        print("\n🔀 Retrieving with hybrid search...")
        
        result = generate_answer(choice)
        
        print("-" * 50)
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
