"""
Naive RAG - Generation Module
Uses retrieved context to generate answers using OpenAI LLM.
"""

import os
from dotenv import load_dotenv

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # sandbox
    from llm_shim import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieval import retrieve_documents, format_retrieved_context

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM Configuration
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
TOP_K = 5

# RAG Prompt Template
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context from Warren Buffett's annual shareholder letters.

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


def generate_answer(question: str, top_k: int = TOP_K, verbose: bool = False) -> dict:
    """
    Generate an answer using the naive RAG pipeline.
    
    Args:
        question: The user's question
        top_k: Number of documents to retrieve
        verbose: If True, include retrieved documents in response
    
    Returns:
        Dictionary containing the answer and optionally the sources
    """
    # Step 1: Retrieve relevant documents
    documents = retrieve_documents(question, top_k=top_k)
    
    if not documents:
        return {
            "answer": "I couldn't find any relevant information to answer your question.",
            "sources": []
        }
    
    # Step 2: Format context
    context = format_retrieved_context(documents)
    
    # Step 3: Generate answer
    chain = create_rag_chain()
    answer = chain.invoke({
        "context": context,
        "question": question
    })
    
    # Prepare response
    response = {
        "answer": answer,
        "sources": [
            {
                "file": doc.metadata.get("source_file", "Unknown"),
                "year": doc.metadata.get("year", "Unknown"),
                "page": doc.metadata.get("page", "Unknown")
            }
            for doc in documents
        ]
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
    """Run an interactive Q&A session."""
    print("\n" + "=" * 60)
    print("Naive RAG - Interactive Q&A")
    print("Ask questions about Warren Buffett's shareholder letters")
    print("Type 'quit' or 'exit' to end the session")
    print("=" * 60)
    
    while True:
        print()
        question = input("Your question: ").strip()
        
        if not question:
            continue
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        print("\n🔍 Retrieving relevant documents...")
        print("🤖 Generating answer...\n")
        
        try:
            result = generate_answer(question, verbose=False)
            
            print("-" * 50)
            print("Answer:")
            print("-" * 50)
            print(result["answer"])
            
            print("\n📚 Sources:")
            for source in result["sources"]:
                print(f"  • {source['file']} (Year: {source['year']}, Page: {source['page']})")
            
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Run example queries or interactive mode."""
    print("=" * 50)
    print("Naive RAG - Generation Pipeline")
    print("=" * 50)
    
    # Validate environment
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Example queries
    example_questions = [
        "What does Warren Buffett say about the importance of management quality?",
        "How does Buffett view market volatility?",
        "What are Berkshire Hathaway's main business segments?",
    ]
    
    print("\n📋 Example Questions Available:")
    for i, q in enumerate(example_questions, 1):
        print(f"  {i}. {q}")
    
    print("\n" + "-" * 50)
    choice = input("Enter question number (1-3), 'i' for interactive mode, or your own question: ").strip()
    
    if choice.lower() == 'i':
        interactive_mode()
    elif choice in ['1', '2', '3']:
        question = example_questions[int(choice) - 1]
        print(f"\n📝 Question: {question}\n")
        print("🔍 Retrieving relevant documents...")
        print("🤖 Generating answer...\n")
        
        result = generate_answer(question, verbose=True)
        
        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])
        
        print("\n📚 Sources:")
        for source in result["sources"]:
            print(f"  • {source['file']} (Year: {source['year']}, Page: {source['page']})")
    elif choice:
        print(f"\n📝 Question: {choice}\n")
        print("🔍 Retrieving relevant documents...")
        print("🤖 Generating answer...\n")
        
        result = generate_answer(choice, verbose=False)
        
        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])
        
        print("\n📚 Sources:")
        for source in result["sources"]:
            print(f"  • {source['file']} (Year: {source['year']}, Page: {source['page']})")
    else:
        print("\n👋 No question provided. Run again to try!")


if __name__ == "__main__":
    main()
