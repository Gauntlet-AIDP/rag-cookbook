"""
Metadata-Filtered RAG - Ingestion Pipeline
Loads PDFs, extracts metadata using fast string matching, chunks, and stores in MongoDB.

Enhanced metadata includes:
- source_file: Original PDF filename
- year: Year extracted from filename
- decade: Decade (2000, 2010, 2020)
- page: Page number within the PDF
- chunk_index: Position of chunk within the document
- word_count: Number of words in chunk
- has_financials: Whether chunk contains financial figures ($, %)
- topic_buckets: List of topics detected via keyword matching
- companies_mentioned: Berkshire portfolio companies mentioned
"""

import os
import re
import certifi
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # sandbox
    from llm_shim import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

try:
    from llm_shim import ensure_vector_index  # sandbox only: creates the index Atlas UI would
except ImportError:
    ensure_vector_index = None

# Load environment variables
load_dotenv()

# Configuration
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
# Atlas (mongodb+srv, mongodb.net or tls/ssl=true) needs the certifi CA bundle; a plain local URL does not
_TLS_HINTS = ("mongodb+srv", "mongodb.net", "tls=true", "ssl=true")
MONGO_KWARGS = {"tlsCAFile": certifi.where()} if any(h in (MONGO_DB_URL or "").lower() for h in _TLS_HINTS) else {}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB configuration
DB_NAME = "rag_playbook"
COLLECTION_NAME = "metadata_filtered_rag"
INDEX_NAME = "metadata_filtered_index"

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# =============================================================================
# TOPIC BUCKETS - Fast keyword-based topic detection
# =============================================================================
TOPIC_BUCKETS = {
    "insurance": [
        "insurance", "float", "underwriting", "policyholder", "reinsurance",
        "premium", "claims", "actuarial", "catastrophe", "geico", "insurer"
    ],
    "acquisitions": [
        "acquire", "acquisition", "purchase", "bought", "deal", "merger",
        "takeover", "buyout", "acquired", "purchasing"
    ],
    "investments": [
        "stock", "equity", "bond", "investment", "portfolio", "shares",
        "securities", "dividend", "capital gains", "unrealized"
    ],
    "management": [
        "manager", "ceo", "leadership", "culture", "compensation", "executive",
        "incentive", "governance", "board", "directors"
    ],
    "berkshire_operations": [
        "railroad", "bnsf", "energy", "utilities", "manufacturing",
        "retail", "subsidiary", "operating earnings"
    ],
    "market_commentary": [
        "market", "valuation", "bubble", "recession", "economy", "gdp",
        "inflation", "interest rate", "fed", "monetary"
    ],
    "capital_allocation": [
        "buyback", "repurchase", "dividend", "retained earnings", "reinvest",
        "capital allocation", "cash", "treasury"
    ],
    "accounting": [
        "gaap", "earnings", "depreciation", "amortization", "goodwill",
        "book value", "intrinsic value", "accounting"
    ],
}

# =============================================================================
# BERKSHIRE STOCK PORTFOLIO COMPANIES
# Includes name variations, tickers, and common references
# =============================================================================
PORTFOLIO_COMPANIES = {
    "apple": {
        "names": ["apple", "apple inc"],
        "ticker": "aapl",
        "variations": ["iphone", "tim cook"]
    },
    "american_express": {
        "names": ["american express", "amex"],
        "ticker": "axp",
        "variations": ["amex card"]
    },
    "bank_of_america": {
        "names": ["bank of america", "bofa"],
        "ticker": "bac",
        "variations": ["b of a"]
    },
    "coca_cola": {
        "names": ["coca-cola", "coca cola", "coke"],
        "ticker": "ko",
        "variations": ["soft drink"]
    },
    "chevron": {
        "names": ["chevron", "chevron corporation"],
        "ticker": "cvx",
        "variations": []
    },
    "occidental_petroleum": {
        "names": ["occidental petroleum", "occidental"],
        "ticker": "oxy",
        "variations": []
    },
    "moodys": {
        "names": ["moody's", "moodys", "moody"],
        "ticker": "mco",
        "variations": ["credit rating"]
    },
    "kraft_heinz": {
        "names": ["kraft heinz", "kraft", "heinz"],
        "ticker": "khc",
        "variations": []
    },
    "chubb": {
        "names": ["chubb", "chubb limited"],
        "ticker": "cb",
        "variations": []
    },
    "visa": {
        "names": ["visa"],
        "ticker": "v",
        "variations": ["payment network"]
    },
    "mastercard": {
        "names": ["mastercard", "master card"],
        "ticker": "ma",
        "variations": []
    },
    "unitedhealth": {
        "names": ["unitedhealth", "united health", "unitedhealth group"],
        "ticker": "unh",
        "variations": []
    },
    "capital_one": {
        "names": ["capital one", "capitalone"],
        "ticker": "cof",
        "variations": []
    },
    "aon": {
        "names": ["aon", "aon plc"],
        "ticker": "aon",
        "variations": []
    },
    "ally_financial": {
        "names": ["ally financial", "ally"],
        "ticker": "ally",
        "variations": []
    },
    "sirius_xm": {
        "names": ["sirius xm", "sirius", "siriusxm"],
        "ticker": "siri",
        "variations": ["satellite radio"]
    },
    "verisign": {
        "names": ["verisign", "veri sign"],
        "ticker": "vrsn",
        "variations": []
    },
    "constellation_brands": {
        "names": ["constellation brands", "constellation"],
        "ticker": "stz",
        "variations": ["corona", "modelo"]
    },
    "kroger": {
        "names": ["kroger", "kroger co"],
        "ticker": "kr",
        "variations": []
    },
    "dominos": {
        "names": ["domino's", "dominos", "domino's pizza"],
        "ticker": "dpz",
        "variations": []
    },
    "pool_corp": {
        "names": ["pool corporation", "pool corp"],
        "ticker": "pool",
        "variations": []
    },
    "louisiana_pacific": {
        "names": ["louisiana-pacific", "louisiana pacific"],
        "ticker": "lpx",
        "variations": []
    },
    "dr_horton": {
        "names": ["d.r. horton", "dr horton", "d r horton"],
        "ticker": "dhi",
        "variations": ["homebuilder"]
    },
    "nucor": {
        "names": ["nucor", "nucor corp", "nucor corporation"],
        "ticker": "nue",
        "variations": ["steel"]
    },
}

# =============================================================================
# BERKSHIRE SUBSIDIARIES (wholly owned)
# =============================================================================
BERKSHIRE_SUBSIDIARIES = [
    "geico", "bnsf", "berkshire hathaway energy", "precision castparts",
    "lubrizol", "marmon", "imc", "forest river", "clayton homes",
    "shaw industries", "benjamin moore", "duracell", "dairy queen",
    "see's candies", "fruit of the loom", "netjets", "business wire",
    "brooks", "oriental trading", "pampered chef"
]


def get_pdf_files(letters_dir: str) -> list[Path]:
    """Get all PDF files from the letters directory."""
    letters_path = Path(letters_dir)
    pdf_files = sorted(letters_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    return pdf_files


def detect_topic_buckets(text: str) -> list[str]:
    """Detect which topic buckets apply to this text using keyword matching."""
    text_lower = text.lower()
    detected_topics = []
    
    for topic, keywords in TOPIC_BUCKETS.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected_topics.append(topic)
                break  # Only add topic once
    
    return detected_topics


def detect_portfolio_companies(text: str) -> list[str]:
    """Detect mentions of Berkshire portfolio companies."""
    text_lower = text.lower()
    detected_companies = []
    
    for company_key, company_data in PORTFOLIO_COMPANIES.items():
        found = False
        
        # Check company names
        for name in company_data["names"]:
            if name in text_lower:
                found = True
                break
        
        # Check ticker (with word boundaries to avoid false positives)
        if not found:
            ticker = company_data["ticker"]
            # Look for ticker with word boundaries (e.g., "AAPL" not "aaplause")
            ticker_pattern = r'\b' + re.escape(ticker) + r'\b'
            if re.search(ticker_pattern, text_lower):
                found = True
        
        # Check variations
        if not found:
            for variation in company_data.get("variations", []):
                if variation in text_lower:
                    found = True
                    break
        
        if found:
            detected_companies.append(company_key)
    
    return detected_companies


def detect_subsidiaries(text: str) -> list[str]:
    """Detect mentions of Berkshire wholly-owned subsidiaries."""
    text_lower = text.lower()
    detected = []
    
    for subsidiary in BERKSHIRE_SUBSIDIARIES:
        if subsidiary in text_lower:
            detected.append(subsidiary)
    
    return detected


def has_financial_figures(text: str) -> bool:
    """Check if text contains financial figures ($, %, large numbers)."""
    # Check for dollar amounts
    if re.search(r'\$[\d,]+', text):
        return True
    # Check for percentages
    if re.search(r'\d+\.?\d*\s*%', text):
        return True
    # Check for billion/million mentions
    if re.search(r'\d+\.?\d*\s*(billion|million|B|M)\b', text, re.IGNORECASE):
        return True
    return False


def extract_metadata(text: str, pdf_path: Path, page: int, chunk_idx: int) -> dict:
    """Extract all metadata for a chunk using fast string matching."""
    
    # Basic metadata
    year_str = pdf_path.stem.replace("ltr", "")
    try:
        year = int(year_str)
    except ValueError:
        year = 0
    
    decade = (year // 10) * 10 if year > 0 else 0
    
    # Text statistics
    word_count = len(text.split())
    
    # Topic detection
    topic_buckets = detect_topic_buckets(text)
    
    # Company detection
    portfolio_companies = detect_portfolio_companies(text)
    subsidiaries = detect_subsidiaries(text)
    
    # Combine portfolio and subsidiaries into companies_mentioned
    companies_mentioned = list(set(portfolio_companies + subsidiaries))
    
    # Financial content detection
    has_financials = has_financial_figures(text)
    
    return {
        "source_file": pdf_path.name,
        "year": year,
        "decade": decade,
        "page": page,
        "chunk_index": chunk_idx,
        "word_count": word_count,
        "has_financials": has_financials,
        "topic_buckets": topic_buckets,
        "companies_mentioned": companies_mentioned,
    }


def load_and_chunk_pdfs_with_metadata(pdf_files: list[Path]) -> list:
    """Load PDFs and split into chunks with rich metadata."""
    all_documents = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            
            # Split documents into chunks
            chunks = text_splitter.split_documents(documents)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Get page from existing metadata
                page = int(chunk.metadata.get("page", 0))
                
                # Extract all metadata
                metadata = extract_metadata(
                    text=chunk.page_content,
                    pdf_path=pdf_path,
                    page=page,
                    chunk_idx=chunk_idx
                )
                
                # Update chunk metadata
                chunk.metadata.update(metadata)
                all_documents.append(chunk)
            
            # Summary for this PDF
            topics_found = set()
            companies_found = set()
            for chunk in chunks:
                topics_found.update(chunk.metadata.get("topic_buckets", []))
                companies_found.update(chunk.metadata.get("companies_mentioned", []))
            
            print(f"  -> {len(chunks)} chunks")
            print(f"     Topics: {list(topics_found)[:5]}{'...' if len(topics_found) > 5 else ''}")
            print(f"     Companies: {list(companies_found)[:5]}{'...' if len(companies_found) > 5 else ''}")
            
        except Exception as e:
            print(f"  -> Error processing {pdf_path.name}: {e}")
    
    print(f"\nTotal chunks created: {len(all_documents)}")
    return all_documents


def setup_mongodb_collection():
    """Set up MongoDB collection for vector storage."""
    client = MongoClient(MONGO_DB_URL, **MONGO_KWARGS)
    db = client[DB_NAME]
    
    # Create collection if it doesn't exist
    if COLLECTION_NAME not in db.list_collection_names():
        db.create_collection(COLLECTION_NAME)
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        # Clear existing documents for fresh ingestion
        db[COLLECTION_NAME].delete_many({})
        print(f"Cleared existing documents in: {COLLECTION_NAME}")
    
    return client, db[COLLECTION_NAME]


def create_vector_store(collection, documents: list):
    """Create embeddings and store in MongoDB Atlas Vector Search."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    print("\nCreating embeddings and storing in MongoDB...")
    
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=documents,
        embedding=embeddings,
        collection=collection,
        index_name=INDEX_NAME
    )
    
    print(f"Successfully stored {len(documents)} document chunks in MongoDB")
    return vector_store


def print_metadata_summary(collection):
    """Print summary of metadata in the collection."""
    print("\n📊 Metadata Summary:")
    
    # Count by decade
    pipeline = [
        {"$group": {"_id": "$decade", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    decades = list(collection.aggregate(pipeline))
    print(f"\n   By Decade:")
    for d in decades:
        if d["_id"]:
            print(f"   {d['_id']}s: {d['count']} chunks")
    
    # Top topics
    pipeline = [
        {"$unwind": "$topic_buckets"},
        {"$group": {"_id": "$topic_buckets", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    topics = list(collection.aggregate(pipeline))
    print(f"\n   Top Topics:")
    for t in topics:
        print(f"   {t['_id']}: {t['count']} chunks")
    
    # Top companies mentioned
    pipeline = [
        {"$unwind": "$companies_mentioned"},
        {"$group": {"_id": "$companies_mentioned", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    companies = list(collection.aggregate(pipeline))
    print(f"\n   Top Companies Mentioned:")
    for c in companies:
        print(f"   {c['_id']}: {c['count']} chunks")
    
    # Financial content
    financial_count = collection.count_documents({"has_financials": True})
    total_count = collection.count_documents({})
    print(f"\n   Financial Content: {financial_count}/{total_count} chunks ({100*financial_count/total_count:.1f}%)")


def print_vector_search_index_instructions():
    """Print instructions for creating the vector search index in MongoDB Atlas."""
    print("\n" + "=" * 70)
    print("IMPORTANT: Create Vector Search Index in MongoDB Atlas")
    print("=" * 70)
    print(f"""
Create a vector search index with these filter fields:

1. Go to MongoDB Atlas → Your Cluster → Atlas Search → Create Search Index
2. Select "Atlas Vector Search" → JSON Editor
3. Use this configuration:

{{
  "fields": [
    {{
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }},
    {{
      "type": "filter",
      "path": "year"
    }},
    {{
      "type": "filter",
      "path": "decade"
    }},
    {{
      "type": "filter",
      "path": "source_file"
    }},
    {{
      "type": "filter",
      "path": "has_financials"
    }},
    {{
      "type": "filter",
      "path": "topic_buckets"
    }},
    {{
      "type": "filter",
      "path": "companies_mentioned"
    }}
  ]
}}

4. Set the index name to: {INDEX_NAME}
5. Select database: {DB_NAME}
6. Select collection: {COLLECTION_NAME}

Wait for the index to become "Active" before running queries.
""")
    print("=" * 70)


def main():
    """Main ingestion pipeline with fast metadata extraction."""
    print("=" * 60)
    print("Metadata-Filtered RAG - Ingestion Pipeline")
    print("(Fast string-matching metadata extraction)")
    print("=" * 60)
    
    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Get path to letters directory
    script_dir = Path(__file__).parent
    letters_dir = script_dir.parent / "letters"
    
    if not letters_dir.exists():
        raise FileNotFoundError(f"Letters directory not found: {letters_dir}")
    
    # Step 1: Get PDF files
    pdf_files = get_pdf_files(letters_dir)
    
    if not pdf_files:
        raise ValueError("No PDF files found in letters directory")
    
    # Step 2: Load, chunk PDFs, and extract metadata (FAST!)
    print("\nExtracting metadata using fast string matching...")
    documents = load_and_chunk_pdfs_with_metadata(pdf_files)
    
    # Step 3: Setup MongoDB
    client, collection = setup_mongodb_collection()
    
    try:
        # Step 4: Create embeddings and store in vector database
        vector_store = create_vector_store(collection, documents)
        
        # Show metadata summary
        print_metadata_summary(collection)
        
        print("\n✅ Ingestion complete!")
        print(f"   Database: {DB_NAME}")
        print(f"   Collection: {COLLECTION_NAME}")
        print(f"   Documents stored: {len(documents)}")
        
        # Print instructions for creating the vector search index
        if ensure_vector_index:
            ensure_vector_index(
                collection,
                INDEX_NAME,
                filter_fields=["year", "decade", "source_file", "has_financials", "topic_buckets", "companies_mentioned"],
            )
        else:
            print_vector_search_index_instructions()
        
    finally:
        client.close()


if __name__ == "__main__":
    main()
