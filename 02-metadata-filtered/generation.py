"""
Metadata-Filtered RAG - Generation Module
Uses filtered retrieval to generate answers from targeted document subsets.

Supports answering questions with constraints like:
- "Based on letters from the 2010s..."
- "Looking at insurance-related content..."
- "Focusing on Apple and Coca-Cola mentions..."
- "From sections with financial figures..."
"""

import os
from dotenv import load_dotenv
from typing import Optional

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # sandbox
    from llm_shim import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieval import (
    retrieve_with_filter, 
    format_retrieved_context, 
    get_available_years,
    get_topic_counts,
    get_company_counts,
    AVAILABLE_TOPICS,
    AVAILABLE_COMPANIES
)

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM Configuration
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
TOP_K = 5

# RAG Prompt Template with filter context
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on Warren Buffett's annual shareholder letters.

{filter_context}

Use ONLY the information from the context below to answer the question. If the context doesn't contain enough information to fully answer the question, acknowledge what you can answer and what information is missing.

Context:
{context}

Question: {question}

Answer:"""


def create_rag_chain():
    """Create the RAG chain with prompt template and LLM."""
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )
    
    output_parser = StrOutputParser()
    
    chain = prompt | llm | output_parser
    return chain


def build_filter_context(
    year: Optional[int | list[int]] = None,
    year_range: Optional[tuple[int, int]] = None,
    decade: Optional[int | list[int]] = None,
    source_file: Optional[str | list[str]] = None,
    topic_buckets: Optional[str | list[str]] = None,
    companies_mentioned: Optional[str | list[str]] = None,
    has_financials: Optional[bool] = None,
) -> str:
    """Build a human-readable description of the applied filters."""
    parts = []
    
    if year is not None:
        if isinstance(year, list):
            parts.append(f"from years {', '.join(map(str, year))}")
        else:
            parts.append(f"from year {year}")
    
    if year_range is not None:
        parts.append(f"from {year_range[0]} to {year_range[1]}")
    
    if decade is not None:
        if isinstance(decade, list):
            parts.append(f"from decades {', '.join(str(d) + 's' for d in decade)}")
        else:
            parts.append(f"from the {decade}s")
    
    if source_file is not None:
        if isinstance(source_file, list):
            parts.append(f"from files: {', '.join(source_file)}")
        else:
            parts.append(f"from {source_file}")
    
    if topic_buckets is not None:
        if isinstance(topic_buckets, str):
            topic_buckets = [topic_buckets]
        parts.append(f"focusing on topics: {', '.join(topic_buckets)}")
    
    if companies_mentioned is not None:
        if isinstance(companies_mentioned, str):
            companies_mentioned = [companies_mentioned]
        parts.append(f"mentioning companies: {', '.join(companies_mentioned)}")
    
    if has_financials is True:
        parts.append("containing financial figures and data")
    
    if parts:
        return f"You are answering based on a filtered subset of documents: {'; '.join(parts)}."
    else:
        return "You are answering based on all available shareholder letters."


def generate_answer(
    question: str,
    top_k: int = TOP_K,
    year: Optional[int | list[int]] = None,
    year_range: Optional[tuple[int, int]] = None,
    decade: Optional[int | list[int]] = None,
    source_file: Optional[str | list[str]] = None,
    topic_buckets: Optional[str | list[str]] = None,
    companies_mentioned: Optional[str | list[str]] = None,
    has_financials: Optional[bool] = None,
    verbose: bool = False,
) -> dict:
    """
    Generate an answer using the metadata-filtered RAG pipeline.
    
    Args:
        question: The user's question
        top_k: Number of documents to retrieve
        year: Filter by year(s)
        year_range: Filter by year range (start, end)
        decade: Filter by decade (2000, 2010, 2020)
        source_file: Filter by source file(s)
        topic_buckets: Filter by topic(s)
        companies_mentioned: Filter by company mentions
        has_financials: Filter for financial content
        verbose: Include retrieved documents in response
    
    Returns:
        Dictionary containing the answer and metadata
    """
    # Step 1: Retrieve with filters
    documents = retrieve_with_filter(
        query=question,
        top_k=top_k,
        year=year,
        year_range=year_range,
        decade=decade,
        source_file=source_file,
        topic_buckets=topic_buckets,
        companies_mentioned=companies_mentioned,
        has_financials=has_financials,
        verbose=verbose
    )
    
    if not documents:
        return {
            "answer": "I couldn't find any relevant information matching your criteria.",
            "sources": [],
            "filters_applied": build_filter_context(
                year, year_range, decade, source_file, 
                topic_buckets, companies_mentioned, has_financials
            )
        }
    
    # Step 2: Format context
    context = format_retrieved_context(documents)
    filter_context = build_filter_context(
        year, year_range, decade, source_file,
        topic_buckets, companies_mentioned, has_financials
    )
    
    # Step 3: Generate answer
    chain = create_rag_chain()
    answer = chain.invoke({
        "context": context,
        "question": question,
        "filter_context": filter_context
    })
    
    # Prepare response
    response = {
        "answer": answer,
        "sources": [
            {
                "file": doc.metadata.get("source_file", "Unknown"),
                "year": doc.metadata.get("year", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "topics": doc.metadata.get("topic_buckets", []),
                "companies": doc.metadata.get("companies_mentioned", [])
            }
            for doc in documents
        ],
        "filters_applied": filter_context
    }
    
    if verbose:
        response["retrieved_documents"] = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]
    
    return response


def interactive_mode():
    """Run an interactive Q&A session with filter support."""
    print("\n" + "=" * 60)
    print("Metadata-Filtered RAG - Interactive Q&A")
    print("=" * 60)
    print("\nFilter commands:")
    print("  year:2020           - Filter by specific year")
    print("  years:2018-2023     - Filter by year range")
    print("  decade:2010         - Filter by decade")
    print("  topic:insurance     - Filter by topic bucket")
    print("  company:apple       - Filter by company mention")
    print("  financials:on       - Show only financial content")
    print("  clear               - Reset all filters")
    print("  filters             - Show active filters")
    print("  help                - Show available topics/companies")
    print("  quit                - Exit")
    print("-" * 60)
    
    # Active filters
    active_filters = {
        "year": None,
        "year_range": None,
        "decade": None,
        "topic_buckets": None,
        "companies_mentioned": None,
        "has_financials": None
    }
    
    while True:
        print()
        user_input = input(">> ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'clear':
            active_filters = {k: None for k in active_filters}
            print("✅ All filters cleared")
            continue
        
        if user_input.lower() == 'filters':
            active = {k: v for k, v in active_filters.items() if v is not None}
            if active:
                print("🔍 Active filters:")
                for k, v in active.items():
                    print(f"   {k}: {v}")
            else:
                print("No active filters")
            continue
        
        if user_input.lower() == 'help':
            print("\n📚 Available Topics:")
            for topic, count in get_topic_counts():
                print(f"   {topic} ({count} chunks)")
            print("\n🏢 Top Companies (by mention count):")
            for company, count in get_company_counts()[:15]:
                print(f"   {company} ({count} chunks)")
            continue
        
        # Parse filter commands
        if user_input.startswith('year:'):
            try:
                active_filters["year"] = int(user_input.split(':')[1])
                active_filters["year_range"] = None
                print(f"✅ Year filter: {active_filters['year']}")
            except ValueError:
                print("❌ Invalid format. Use 'year:2020'")
            continue
        
        if user_input.startswith('years:'):
            try:
                range_str = user_input.split(':')[1]
                start, end = map(int, range_str.split('-'))
                active_filters["year_range"] = (start, end)
                active_filters["year"] = None
                print(f"✅ Year range filter: {start}-{end}")
            except ValueError:
                print("❌ Invalid format. Use 'years:2018-2023'")
            continue
        
        if user_input.startswith('decade:'):
            try:
                active_filters["decade"] = int(user_input.split(':')[1])
                print(f"✅ Decade filter: {active_filters['decade']}s")
            except ValueError:
                print("❌ Invalid format. Use 'decade:2010'")
            continue
        
        if user_input.startswith('topic:'):
            topics = user_input.split(':')[1].split(',')
            active_filters["topic_buckets"] = [t.strip() for t in topics]
            print(f"✅ Topic filter: {active_filters['topic_buckets']}")
            continue
        
        if user_input.startswith('company:'):
            companies = user_input.split(':')[1].split(',')
            active_filters["companies_mentioned"] = [c.strip() for c in companies]
            print(f"✅ Company filter: {active_filters['companies_mentioned']}")
            continue
        
        if user_input.startswith('financials:'):
            val = user_input.split(':')[1].lower()
            active_filters["has_financials"] = val in ['on', 'true', 'yes', '1']
            print(f"✅ Financials filter: {'ON' if active_filters['has_financials'] else 'OFF'}")
            continue
        
        # Treat as question
        question = user_input
        
        # Show active filters
        active = {k: v for k, v in active_filters.items() if v is not None}
        if active:
            print(f"🔍 Searching with filters: {active}")
        
        print("🤖 Generating answer...\n")
        
        try:
            result = generate_answer(
                question,
                **active_filters,
                verbose=False
            )
            
            print("-" * 50)
            print("Answer:")
            print("-" * 50)
            print(result["answer"])
            
            print(f"\n📚 Sources ({len(result['sources'])} documents):")
            for source in result["sources"]:
                topics_str = ', '.join(source['topics'][:2]) if source['topics'] else 'N/A'
                companies_str = ', '.join(source['companies'][:2]) if source['companies'] else 'N/A'
                print(f"  • {source['file']} (Year: {source['year']}) | Topics: {topics_str}")
            
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Run example queries or interactive mode."""
    print("=" * 60)
    print("Metadata-Filtered RAG - Generation Pipeline")
    print("=" * 60)
    
    # Validate environment
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Example queries with filters
    example_queries = [
        {
            "question": "How does Berkshire's insurance float work?",
            "topic_buckets": ["insurance"],
            "description": "Insurance-focused query"
        },
        {
            "question": "What does Buffett think about Coca-Cola as an investment?",
            "companies_mentioned": ["coca_cola"],
            "description": "Company-specific query"
        },
        {
            "question": "What were Berkshire's major acquisitions?",
            "topic_buckets": ["acquisitions"],
            "decade": 2010,
            "description": "2010s acquisitions"
        },
        {
            "question": "What are Apple's contributions to Berkshire?",
            "companies_mentioned": ["apple"],
            "has_financials": True,
            "description": "Apple with financials"
        },
    ]
    
    print("\n📋 Example Queries with Filters:")
    for i, q in enumerate(example_queries, 1):
        print(f"  {i}. {q['question'][:45]}... ({q['description']})")
    
    print("\n" + "-" * 50)
    choice = input("Enter 1-4 for examples, 'i' for interactive mode, or your question: ").strip()
    
    if choice.lower() == 'i':
        interactive_mode()
    elif choice in ['1', '2', '3', '4']:
        q = example_queries[int(choice) - 1]
        print(f"\n📝 Question: {q['question']}")
        print(f"📁 Filters: {q['description']}")
        print("\n🔍 Retrieving with filters...")
        print("🤖 Generating answer...\n")
        
        # Build kwargs from example
        kwargs = {k: v for k, v in q.items() if k not in ['question', 'description']}
        
        result = generate_answer(q['question'], **kwargs, verbose=False)
        
        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])
        
        print(f"\n📚 Sources:")
        for source in result["sources"]:
            print(f"  • {source['file']} (Year: {source['year']})")
            if source['companies']:
                print(f"    Companies: {', '.join(source['companies'][:3])}")
        
        print(f"\n🔍 {result['filters_applied']}")
    elif choice:
        print(f"\n📝 Question: {choice}")
        print("\n🔍 Retrieving (no filters)...")
        print("🤖 Generating answer...\n")
        
        result = generate_answer(choice, verbose=False)
        
        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])
        
        print(f"\n📚 Sources:")
        for source in result["sources"]:
            print(f"  • {source['file']} (Year: {source['year']})")
    else:
        print("\n👋 No input provided. Run again to try!")


if __name__ == "__main__":
    main()
