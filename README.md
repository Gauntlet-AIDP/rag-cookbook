# RAG Cookbook 📚

A hands-on guide to Retrieval-Augmented Generation (RAG) patterns, progressing from simple to sophisticated implementations.

To run the modules locally with no cloud accounts or API keys, see [sandbox/README.md](sandbox/README.md).

## What You'll Build

Using **Warren Buffett's Berkshire Hathaway shareholder letters** (2004-2023) as your corpus, you'll implement 5 increasingly complex RAG patterns:

| Step | Pattern | What You'll Learn |
|------|---------|-------------------|
| 01 | **Naive RAG** | Basic chunking, embedding, vector search |
| 02 | **Metadata Filtering** | Pre-filtering by year, topic, company |
| 03 | **Hybrid Search** | Combining BM25 + vector search with RRF |
| 04 | **Graph RAG** | Knowledge graphs + vector search |
| 05 | **Agentic RAG** | Dynamic retrieval decisions with agents |

## Tech Stack

- **Vector Database**: MongoDB Atlas
- **Embeddings/LLM**: OpenAI (text-embedding-3-small, gpt-4o-mini)
- **Framework**: LangChain
- **Graph Database**: Neo4j (Step 04 only)

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install all dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the root directory:

```env
# Required for all steps
OPENAI_API_KEY=your_openai_key
MONGO_DB_URL=mongodb+srv://user:pass@cluster.mongodb.net/

# Optional (for LangSmith tracing)
LANGCHAIN_API_KEY=your_langchain_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=rag-cookbook

# For Step 04 (Graph RAG)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
```

### 3. Start with Step 01

```bash
cd 01-naive-rag
python ingestion.py   # Ingest PDFs into MongoDB
python generation.py  # Ask questions!
```

---

## Step-by-Step Guide

### 📁 01-naive-rag — The Foundation

**Concept**: The simplest RAG implementation. Chunk documents, embed them, store in a vector database, retrieve the top-k similar chunks, and generate an answer.

```
PDF → Chunk → Embed → Store → Query → Retrieve Top-K → Generate
```

**What you'll learn**:
- PDF loading and text chunking
- Creating embeddings with OpenAI
- Storing vectors in MongoDB Atlas
- Basic similarity search
- Prompt engineering for RAG

**Files**:
- `ingestion.py` - Load PDFs, chunk, embed, store
- `retrieval.py` - Vector similarity search
- `generation.py` - RAG pipeline with LLM
- `evals/precision.py` - Measure retrieval precision
- `evals/groundedness.py` - Measure answer groundedness

**Run it**:
```bash
python 01-naive-rag/ingestion.py
python 01-naive-rag/generation.py
```

---

### 📁 02-metadata-filtered — Smarter Retrieval

**Concept**: Enhance retrieval by filtering documents BEFORE vector search using metadata like year, topic, and company mentions.

```
Query → Extract Filters → Pre-Filter → Vector Search → Generate
```

**What you'll learn**:
- Extracting rich metadata (topics, companies, financial indicators)
- MongoDB pre-filtering with vector search
- Targeted retrieval for specific time periods or topics
- Comparing filtered vs. unfiltered precision

**New metadata fields**:
- `year`, `decade` - Temporal filtering
- `topic_buckets` - Insurance, acquisitions, investments, etc.
- `companies_mentioned` - Apple, Coca-Cola, GEICO, etc.
- `has_financials` - Contains dollar amounts or percentages

**Files**:
- `ingestion.py` - Extract metadata using fast string matching
- `retrieval.py` - Filtered vector search
- `generation.py` - Interactive Q&A with filters
- `evals/latency.py` - Compare retrieval speed
- `evals/precision_delta.py` - Measure precision improvement

**Run it**:
```bash
python 02-metadata-filtered/ingestion.py
python 02-metadata-filtered/generation.py
# Try: "year:2020" then "How did Berkshire perform?"
```

---

### 📁 03-hybrid-search — Best of Both Worlds

**Concept**: Combine keyword search (BM25) with semantic search (vectors) for better retrieval. BM25 catches exact terms; vectors catch meaning.

```
Query → BM25 Search ─┬─→ Reciprocal Rank Fusion → Generate
      → Vector Search ─┘
```

**What you'll learn**:
- BM25 (TF-IDF based) retrieval
- Reciprocal Rank Fusion (RRF) for combining results
- When hybrid beats pure vector search
- Tuning BM25/vector weights

**When hybrid helps**:
| Query | Vector Only | Hybrid |
|-------|-------------|--------|
| "GEICO earnings 2020" | ⚠️ May miss exact terms | ✅ Catches both |
| "What makes a good investment?" | ✅ Semantic match | ✅ Also good |
| "BRK.A stock split" | ⚠️ May miss ticker | ✅ Keyword match |

**Files**:
- `retrieval.py` - Hybrid search with RRF
- `generation.py` - Interactive Q&A with weight tuning

**No new ingestion needed** - uses existing MongoDB vectors!

**Run it**:
```bash
python 03-hybrid-search/retrieval.py
python 03-hybrid-search/generation.py
# Try: "weights:0.7,0.3" to favor BM25
```

---

### 📁 04-graph-rag — Knowledge Graph Enhanced

**Concept**: Build a knowledge graph of entities and relationships, then use graph traversal to find related context before vector search.

```
Query → Extract Entities → Graph Traversal → Find Related Docs
                                    ↓
                            Vector Search → Combine → Generate
```

**What you'll learn**:
- Entity and relationship extraction with LLMs
- Building knowledge graphs in Neo4j
- Cypher queries for graph traversal
- Combining graph context with vector search
- Multi-hop reasoning

**Graph structure**:
```
(Buffett)-[:MANAGES]->(Berkshire)
(Berkshire)-[:OWNS]->(GEICO)
(GEICO)-[:BELONGS_TO]->(Insurance Topic)
(GEICO)-[:MENTIONED_IN]->(2020ltr.pdf)
```

**Files**:
- `docker-compose.yml` - Neo4j setup
- `graph_builder.py` - Extract entities, build graph
- `retrieval.py` - Graph traversal + vector search
- `generation.py` - Interactive Q&A
- `evals/entity_extraction.py` - Measure entity extraction accuracy
- `evals/multi_hop_reasoning.py` - Measure multi-hop path discovery

**Run it**:
```bash
cd 04-graph-rag
docker-compose up -d           # Start Neo4j
python graph_builder.py -n 10  # Build graph (10 chunks for testing)
python generation.py
```

---

### 📁 05-agentic-rag — Dynamic Decision Making

**Concept**: An AI agent that dynamically decides WHETHER to retrieve, WHICH method to use, and WHETHER to retry with a different approach.

```
Query → Analyze → Decide: Retrieve? ─→ No: Use model knowledge
                      ↓ Yes
                Choose tool → Execute → Evaluate: Sufficient?
                      ↑                      ↓ No
                      └────── Retry ─────────┘
                                             ↓ Yes
                                         Synthesize Answer
```

**What you'll learn**:
- ReAct (Reason + Act) agent pattern
- Query analysis and decomposition
- Tool selection (vector vs. filtered vs. none)
- Self-evaluation and retry logic
- Multi-step retrieval for complex questions

**Agent capabilities**:
1. **Decides IF** retrieval is needed
2. **Chooses HOW** to retrieve (vector vs. filtered)
3. **Decomposes** complex queries into sub-queries
4. **Evaluates** if retrieved info is sufficient
5. **Retries** with different approach if needed

**Files**:
- `tools.py` - Retrieval tools for the agent
- `agent.py` - ReAct agent implementation
- `generation.py` - Interactive mode
- `evals/tool_selection.py` - Measure tool choice accuracy
- `evals/query_decomposition.py` - Measure query breakdown quality
- `evals/end_to_end.py` - Full pipeline evaluation

**No new ingestion needed** - uses existing MongoDB vectors!

**Run it**:
```bash
python 05-agentic-rag/generation.py
# Try: "Compare insurance performance in 2008 vs 2020"
```

---

## Evaluation

Each RAG pattern includes evaluation metrics:

| Metric | What it Measures | Location |
|--------|------------------|----------|
| **Precision** | Relevant docs / Retrieved docs | `01-naive-rag/evals/` |
| **Groundedness** | Is answer supported by context? | `01-naive-rag/evals/` |
| **Latency** | Time to retrieve and generate | `02-metadata-filtered/evals/` |
| **Precision Delta** | Improvement from filtering | `02-metadata-filtered/evals/` |
| **Entity Extraction** | Accuracy of extracting entities from queries | `04-graph-rag/evals/` |
| **Multi-Hop Reasoning** | Can graph find expected entity connections? | `04-graph-rag/evals/` |
| **Tool Selection** | Does agent pick the right tool? | `05-agentic-rag/evals/` |
| **Query Decomposition** | Quality of breaking down complex queries | `05-agentic-rag/evals/` |
| **End-to-End** | Answer quality, behavior, latency | `05-agentic-rag/evals/` |

```bash
# Run evaluations
python 01-naive-rag/evals/precision.py
python 01-naive-rag/evals/groundedness.py
python 02-metadata-filtered/evals/latency.py
python 02-metadata-filtered/evals/precision_delta.py
python 04-graph-rag/evals/entity_extraction.py
python 04-graph-rag/evals/multi_hop_reasoning.py
python 05-agentic-rag/evals/tool_selection.py
python 05-agentic-rag/evals/query_decomposition.py
python 05-agentic-rag/evals/end_to_end.py
```

---

## Corpus

The `letters/` directory contains Warren Buffett's annual shareholder letters from 2004-2023 (20 PDFs). These are ideal for RAG because they:

- Span 20 years (good for temporal filtering)
- Cover diverse topics (insurance, acquisitions, investments)
- Mention many companies (Apple, Coca-Cola, GEICO, etc.)
- Contain both narrative and financial data

---

## Project Structure

```
rag-cookbook/
├── README.md                    # You are here
├── .env                         # API keys (create this)
├── letters/                     # Source PDFs (Buffett letters)
│
├── 01-naive-rag/               # Step 1: Basic RAG
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── generation.py
│   └── evals/
│       ├── precision.py
│       └── groundedness.py
│
├── 02-metadata-filtered/       # Step 2: Filtered RAG
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── generation.py
│   └── evals/
│       ├── latency.py
│       └── precision_delta.py
│
├── 03-hybrid-search/           # Step 3: BM25 + Vector
│   ├── README.md
│   ├── retrieval.py
│   └── generation.py
│
├── 04-graph-rag/               # Step 4: Knowledge Graph
│   ├── README.md
│   ├── docker-compose.yml
│   ├── graph_builder.py
│   ├── retrieval.py
│   ├── generation.py
│   └── evals/
│       ├── entity_extraction.py
│       └── multi_hop_reasoning.py
│
└── 05-agentic-rag/             # Step 5: Agent-based
    ├── README.md
    ├── tools.py
    ├── agent.py
    ├── generation.py
    └── evals/
        ├── tool_selection.py
        ├── query_decomposition.py
        └── end_to_end.py
```

---

## License

MIT License - feel free to use this for learning and building!

---

## Next Steps

After completing this playbook, consider exploring:

- **Reranking** - Add a cross-encoder reranker after retrieval
- **Query Expansion** - Generate multiple query variations
- **Contextual Chunking** - Chunk with document structure awareness
- **Multi-modal RAG** - Add images and tables from PDFs
- **RAG Fusion** - Generate multiple queries, retrieve, and fuse results
