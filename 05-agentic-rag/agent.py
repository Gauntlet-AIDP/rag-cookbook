"""
Agentic RAG - ReAct Agent
An agent that dynamically decides when and how to retrieve.

Uses a ReAct (Reason + Act) loop:
1. Analyze the query
2. Decide on action (retrieve or not, which method)
3. Execute the action
4. Evaluate results
5. Retry or synthesize answer
"""

import os
import json
from dotenv import load_dotenv
from typing import Optional

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # sandbox
    from llm_shim import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import (
    no_retrieval,
    vector_search,
    filtered_search,
    format_documents_as_context,
    AVAILABLE_TOOLS
)

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_MODEL = "gpt-4o-mini"
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))  # Maximum retrieval attempts per sub-question


# =============================================================================
# Agent Prompts
# =============================================================================

QUERY_ANALYZER_PROMPT = """Analyze this question and determine the best retrieval strategy.

Question: {question}

Analyze:
1. Is this a simple question that can be answered from general knowledge, or does it require specific information from Buffett's shareholder letters?
2. Is this a single question or a complex question that should be broken into sub-questions?
3. Does the question mention specific years, companies, or topics that could be used as filters?

Return a JSON object with this structure:
{{
    "needs_retrieval": true/false,
    "reasoning": "brief explanation of your decision",
    "is_complex": true/false,
    "sub_questions": ["sub-question 1", "sub-question 2"] or null if not complex,
    "suggested_tool": "no_retrieval" | "vector_search" | "filtered_search",
    "suggested_filters": {{
        "year": null or integer,
        "year_range": null or [start, end],
        "topic_buckets": null or ["topic1", "topic2"],
        "companies_mentioned": null or ["company1"],
        "has_financials": null or true/false
    }}
}}

Available topics: insurance, acquisitions, investments, management, berkshire_operations, market_commentary, capital_allocation, accounting
Available companies: apple, coca_cola, geico, american_express, bank_of_america, chevron, etc.

Return ONLY the JSON object, no other text."""


EVALUATION_PROMPT = """Evaluate whether the retrieved information is sufficient to answer the question.

Original Question: {question}

Retrieved Context:
{context}

Evaluate:
1. Does the context contain relevant information for answering the question?
2. Is there enough information to provide a complete answer?
3. If not, what additional information would be helpful?

Return a JSON object:
{{
    "is_sufficient": true/false,
    "relevance_score": 1-5 (1=irrelevant, 5=highly relevant),
    "reasoning": "brief explanation",
    "retry_suggestion": null or "suggestion for different query or filters"
}}

Return ONLY the JSON object."""


SYNTHESIS_PROMPT = """You are a helpful assistant answering questions about Warren Buffett's shareholder letters.

Question: {question}

{retrieval_summary}

Context from retrieved documents:
{context}

Based on the context provided, answer the question. If the context doesn't fully answer the question, acknowledge what you can answer and what information is missing.

Answer:"""


# =============================================================================
# Agent Core
# =============================================================================

class AgenticRAG:
    """ReAct agent for dynamic retrieval decisions."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.llm = ChatOpenAI(
            model=AGENT_MODEL,
            temperature=0.0,
            openai_api_key=OPENAI_API_KEY
        )
    
    def log(self, message: str, indent: int = 0):
        """Log message if verbose mode is on."""
        if self.verbose:
            prefix = "   " * indent
            print(f"{prefix}{message}")
    
    def analyze_query(self, question: str) -> dict:
        """Analyze the query to determine retrieval strategy."""
        self.log(f"\n🤔 ANALYZING query...")
        
        prompt = ChatPromptTemplate.from_template(QUERY_ANALYZER_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({"question": question})
        
        # Parse JSON
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            analysis = json.loads(response)
        except json.JSONDecodeError:
            # Fallback to vector search
            analysis = {
                "needs_retrieval": True,
                "reasoning": "Defaulting to vector search",
                "is_complex": False,
                "sub_questions": None,
                "suggested_tool": "vector_search",
                "suggested_filters": {}
            }
        
        self.log(f"   Needs retrieval: {analysis.get('needs_retrieval', True)}")
        self.log(f"   Is complex: {analysis.get('is_complex', False)}")
        self.log(f"   Suggested tool: {analysis.get('suggested_tool', 'vector_search')}")
        
        if analysis.get("is_complex") and analysis.get("sub_questions"):
            self.log(f"   Sub-questions:")
            for sq in analysis["sub_questions"]:
                self.log(f"   - {sq}", indent=1)
        
        return analysis
    
    def execute_retrieval(
        self,
        query: str,
        tool_name: str,
        filters: dict = None
    ) -> dict:
        """Execute a retrieval tool."""
        self.log(f"\n🔧 EXECUTING {tool_name}...")
        self.log(f"   Query: {query[:50]}...")
        
        if tool_name == "no_retrieval":
            result = no_retrieval(query)
        elif tool_name == "filtered_search" and filters:
            # Extract filter parameters
            result = filtered_search(
                query=query,
                k=5,
                year=filters.get("year"),
                year_range=tuple(filters["year_range"]) if filters.get("year_range") else None,
                topic_buckets=filters.get("topic_buckets"),
                companies_mentioned=filters.get("companies_mentioned"),
                has_financials=filters.get("has_financials")
            )
            self.log(f"   Filters: {result.get('filters', 'none')}")
        else:
            result = vector_search(query=query, k=5)
        
        doc_count = len(result.get("documents", []))
        self.log(f"   Retrieved: {doc_count} documents")
        
        return result
    
    def evaluate_results(self, question: str, documents: list) -> dict:
        """Evaluate if retrieved results are sufficient."""
        self.log(f"\n📊 EVALUATING results...")
        
        if not documents:
            return {
                "is_sufficient": False,
                "relevance_score": 1,
                "reasoning": "No documents retrieved",
                "retry_suggestion": "Try with different query or filters"
            }
        
        context = format_documents_as_context(documents)
        
        prompt = ChatPromptTemplate.from_template(EVALUATION_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "question": question,
            "context": context[:4000]  # Limit context length
        })
        
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            evaluation = json.loads(response)
        except json.JSONDecodeError:
            evaluation = {
                "is_sufficient": True,
                "relevance_score": 3,
                "reasoning": "Unable to evaluate, proceeding with results"
            }
        
        is_sufficient = evaluation.get("is_sufficient", True)
        score = evaluation.get("relevance_score", 3)
        
        self.log(f"   Sufficient: {is_sufficient}")
        self.log(f"   Relevance: {score}/5")
        
        return evaluation
    
    def synthesize_answer(
        self,
        question: str,
        all_contexts: list[dict],
    ) -> str:
        """Synthesize final answer from all retrieved contexts."""
        self.log(f"\n📝 SYNTHESIZING answer...")
        
        # Combine all documents
        all_documents = []
        retrieval_summary_parts = []
        
        for ctx in all_contexts:
            all_documents.extend(ctx.get("documents", []))
            tool = ctx.get("tool", "unknown")
            query = ctx.get("query", "")
            doc_count = len(ctx.get("documents", []))
            retrieval_summary_parts.append(
                f"- {tool}: \"{query[:40]}...\" → {doc_count} docs"
            )
        
        context = format_documents_as_context(all_documents)
        retrieval_summary = "Retrieval steps:\n" + "\n".join(retrieval_summary_parts)
        
        prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({
            "question": question,
            "context": context,
            "retrieval_summary": retrieval_summary
        })
        
        return answer
    
    def run(self, question: str) -> dict:
        """
        Run the agentic RAG pipeline.
        
        Returns:
            Dictionary with answer and metadata about the agent's decisions
        """
        self.log("=" * 60)
        self.log("🤖 AGENTIC RAG")
        self.log("=" * 60)
        self.log(f"\n❓ Question: {question}")
        
        # Step 1: Analyze the query
        analysis = self.analyze_query(question)
        
        all_contexts = []
        
        # Step 2: Check if retrieval is needed
        if not analysis.get("needs_retrieval", True):
            self.log("\n⏭️  Skipping retrieval (not needed)")
            all_contexts.append(no_retrieval(question))
        
        # Step 3: Handle complex vs simple queries
        elif analysis.get("is_complex") and analysis.get("sub_questions"):
            # Complex query: process each sub-question
            self.log(f"\n📋 Processing {len(analysis['sub_questions'])} sub-questions...")
            
            for i, sub_q in enumerate(analysis["sub_questions"], 1):
                self.log(f"\n--- Sub-question {i} ---")
                self.log(f"   \"{sub_q}\"")
                
                # Re-analyze sub-question for filters
                sub_analysis = self.analyze_query(sub_q)
                
                # Execute retrieval
                tool = sub_analysis.get("suggested_tool", "vector_search")
                filters = sub_analysis.get("suggested_filters", {})
                
                result = self.execute_retrieval(sub_q, tool, filters)
                
                # Evaluate
                evaluation = self.evaluate_results(sub_q, result.get("documents", []))
                
                # Retry if not sufficient (up to MAX_ITERATIONS)
                iteration = 1
                while not evaluation.get("is_sufficient", True) and iteration < MAX_ITERATIONS:
                    self.log(f"\n🔄 RETRYING (attempt {iteration + 1})...")
                    
                    # Try broader search
                    result = self.execute_retrieval(sub_q, "vector_search", None)
                    evaluation = self.evaluate_results(sub_q, result.get("documents", []))
                    iteration += 1
                
                all_contexts.append(result)
        
        else:
            # Simple query: single retrieval
            tool = analysis.get("suggested_tool", "vector_search")
            filters = analysis.get("suggested_filters", {})
            
            result = self.execute_retrieval(question, tool, filters)
            
            # Evaluate
            evaluation = self.evaluate_results(question, result.get("documents", []))
            
            # Retry if not sufficient
            iteration = 1
            while not evaluation.get("is_sufficient", True) and iteration < MAX_ITERATIONS:
                self.log(f"\n🔄 RETRYING (attempt {iteration + 1})...")
                
                if iteration == 1:
                    # Try without filters
                    result = self.execute_retrieval(question, "vector_search", None)
                else:
                    # Try with broader query
                    result = self.execute_retrieval(
                        f"Warren Buffett {question}",
                        "vector_search",
                        None
                    )
                
                evaluation = self.evaluate_results(question, result.get("documents", []))
                iteration += 1
            
            all_contexts.append(result)
        
        # Step 4: Synthesize answer
        answer = self.synthesize_answer(question, all_contexts)
        
        # Build response
        response = {
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "retrieval_steps": [
                {
                    "tool": ctx.get("tool"),
                    "query": ctx.get("query"),
                    "documents_retrieved": len(ctx.get("documents", []))
                }
                for ctx in all_contexts
            ],
            "total_documents": sum(len(ctx.get("documents", [])) for ctx in all_contexts)
        }
        
        self.log("\n" + "=" * 60)
        self.log("✅ COMPLETE")
        self.log("=" * 60)
        
        return response


def main():
    """Test the agentic RAG system."""
    print("=" * 60)
    print("Agentic RAG - Test")
    print("=" * 60)
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    
    agent = AgenticRAG(verbose=True)
    
    # Test questions of varying complexity
    test_questions = [
        # Simple question
        "What is insurance float?",
        
        # Question with time context
        "How did Berkshire perform in 2020?",
        
        # Complex comparison question
        "How did Berkshire's insurance business compare between 2008 and 2020?",
    ]
    
    print("\n📋 Test Questions:")
    for i, q in enumerate(test_questions, 1):
        print(f"  {i}. {q}")
    
    choice = input("\nEnter 1-3 to test, or type your own question: ").strip()
    
    if choice in ['1', '2', '3']:
        question = test_questions[int(choice) - 1]
    elif choice:
        question = choice
    else:
        print("No input provided.")
        return
    
    result = agent.run(question)
    
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result["answer"])
    
    print("\n📊 Agent Summary:")
    print(f"   Total documents retrieved: {result['total_documents']}")
    print(f"   Retrieval steps: {len(result['retrieval_steps'])}")
    for step in result["retrieval_steps"]:
        print(f"   - {step['tool']}: {step['documents_retrieved']} docs")


if __name__ == "__main__":
    main()
