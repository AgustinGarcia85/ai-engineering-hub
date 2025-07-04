import os
import asyncio
from typing import TypedDict, List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class RAGConfig:
    """Configuration for the RAG agent"""
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    model_name: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_search_results: int = 5
    vector_db_path: str = "./chroma_db"


class AgentState(TypedDict):
    """State for the RAG agent"""
    query: str
    web_search_results: List[str]
    retrieved_docs: List[Document]
    generated_answer: str
    conversation_history: List[Dict[str, str]]
    metadata: Dict[str, Any]


class LangGraphRAGAgent:
    """LangGraph-based RAG Agent with web search capabilities"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Initialize LLM and embeddings
        self.llm = ChatOpenAI(
            api_key=config.openai_api_key,
            model=config.model_name,
            temperature=0.1
        )
        
        self.embeddings = OpenAIEmbeddings(
            api_key=config.openai_api_key,
            model=config.embedding_model
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory=config.vector_db_path,
            embedding_function=self.embeddings
        )
        
        # Initialize web search
        self.web_search = DuckDuckGoSearchRun()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("web_search", self._web_search_node)
        workflow.add_node("vector_retrieval", self._vector_retrieval_node)
        workflow.add_node("answer_generation", self._answer_generation_node)
        workflow.add_node("quality_check", self._quality_check_node)
        
        # Define the graph flow
        workflow.set_entry_point("web_search")
        workflow.add_edge("web_search", "vector_retrieval")
        workflow.add_edge("vector_retrieval", "answer_generation")
        workflow.add_edge("answer_generation", "quality_check")
        workflow.add_edge("quality_check", END)
        
        # Compile the graph
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    async def _web_search_node(self, state: AgentState) -> AgentState:
        """Perform web search to get relevant information"""
        query = state["query"]
        
        try:
            # Perform web search
            search_results = self.web_search.run(query)
            
            # Extract and clean search results
            web_results = []
            if search_results:
                # Parse search results and extract meaningful content
                results_list = search_results.split('\n')[:self.config.max_search_results]
                for result in results_list:
                    if result.strip():
                        web_results.append(result.strip())
            
            state["web_search_results"] = web_results
            state["metadata"] = state.get("metadata", {})
            state["metadata"]["web_search_count"] = len(web_results)
            
        except Exception as e:
            print(f"Web search error: {e}")
            state["web_search_results"] = []
            state["metadata"] = state.get("metadata", {})
            state["metadata"]["web_search_error"] = str(e)
        
        return state
    
    async def _vector_retrieval_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents from vector store"""
        query = state["query"]
        
        try:
            # Perform similarity search
            retrieved_docs = self.vector_store.similarity_search(
                query, k=5
            )
            
            state["retrieved_docs"] = retrieved_docs
            state["metadata"] = state.get("metadata", {})
            state["metadata"]["retrieved_docs_count"] = len(retrieved_docs)
            
        except Exception as e:
            print(f"Vector retrieval error: {e}")
            state["retrieved_docs"] = []
            state["metadata"] = state.get("metadata", {})
            state["metadata"]["retrieval_error"] = str(e)
        
        return state
    
    async def _answer_generation_node(self, state: AgentState) -> AgentState:
        """Generate answer using retrieved information"""
        query = state["query"]
        web_results = state.get("web_search_results", [])
        retrieved_docs = state.get("retrieved_docs", [])
        conversation_history = state.get("conversation_history", [])
        
        # Prepare context from web search and vector retrieval
        context_parts = []
        
        # Add web search results
        if web_results:
            web_context = "\n".join(web_results)
            context_parts.append(f"Web Search Results:\n{web_context}")
        
        # Add retrieved documents
        if retrieved_docs:
            doc_context = "\n".join([doc.page_content for doc in retrieved_docs])
            context_parts.append(f"Retrieved Documents:\n{doc_context}")
        
        context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
        
        # Prepare conversation history
        history_text = ""
        if conversation_history:
            history_text = "\nConversation History:\n"
            for turn in conversation_history[-3:]:  # Last 3 turns
                history_text += f"Human: {turn.get('human', '')}\nAssistant: {turn.get('assistant', '')}\n"
        
        # Create prompt
        prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant with access to real-time web search and a knowledge base.
Your task is to provide accurate, comprehensive, and well-sourced answers to user questions.

{history}

Context Information:
{context}

User Question: {query}

Instructions:
1. Use the provided context to answer the question comprehensively
2. If the web search results contain recent information, prioritize it
3. If there are contradictions between sources, acknowledge them
4. Cite your sources when possible
5. If the context doesn't fully answer the question, say so
6. Provide a clear, well-structured response

Answer:
""")
        
        try:
            # Generate response
            messages = prompt.format_messages(
                history=history_text,
                context=context,
                query=query
            )
            
            response = await self.llm.ainvoke(messages)
            state["generated_answer"] = response.content
            
        except Exception as e:
            print(f"Answer generation error: {e}")
            state["generated_answer"] = f"I apologize, but I encountered an error while generating the answer: {str(e)}"
            state["metadata"] = state.get("metadata", {})
            state["metadata"]["generation_error"] = str(e)
        
        return state
    
    async def _quality_check_node(self, state: AgentState) -> AgentState:
        """Perform quality check on the generated answer"""
        query = state["query"]
        answer = state["generated_answer"]
        
        # Simple quality checks
        quality_score = 0
        quality_issues = []
        
        # Check answer length
        if len(answer) < 50:
            quality_issues.append("Answer too short")
        else:
            quality_score += 1
        
        # Check if answer addresses the query
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(query_words.intersection(answer_words))
        
        if overlap >= len(query_words) * 0.3:
            quality_score += 1
        else:
            quality_issues.append("Answer may not fully address the query")
        
        # Check for error messages
        if "error" not in answer.lower() and "apologize" not in answer.lower():
            quality_score += 1
        else:
            quality_issues.append("Answer contains error indicators")
        
        state["metadata"] = state.get("metadata", {})
        state["metadata"]["quality_score"] = quality_score
        state["metadata"]["quality_issues"] = quality_issues
        
        return state
    
    async def query(self, question: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Process a query through the RAG pipeline"""
        initial_state = {
            "query": question,
            "web_search_results": [],
            "retrieved_docs": [],
            "generated_answer": "",
            "conversation_history": conversation_history or [],
            "metadata": {}
        }
        
        # Run the graph
        config = {"configurable": {"thread_id": "default"}}
        result = await self.graph.ainvoke(initial_state, config)
        
        return {
            "answer": result["generated_answer"],
            "web_results": result["web_search_results"],
            "retrieved_docs": [doc.page_content for doc in result["retrieved_docs"]],
            "metadata": result["metadata"]
        }
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """Add documents to the vector store"""
        # Split documents into chunks
        docs = []
        for i, doc in enumerate(documents):
            chunks = self.text_splitter.split_text(doc)
            for chunk in chunks:
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                docs.append(Document(page_content=chunk, metadata=metadata))
        
        # Add to vector store
        self.vector_store.add_documents(docs)
        print(f"Added {len(docs)} document chunks to the vector store")
    
    def add_documents_from_urls(self, urls: List[str]):
        """Add documents from web URLs"""
        documents = []
        metadatas = []
        
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text()
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                if text:
                    documents.append(text)
                    metadatas.append({"source": url})
                    print(f"Successfully processed: {url}")
                
            except Exception as e:
                print(f"Error processing {url}: {e}")
        
        if documents:
            self.add_documents(documents, metadatas)


# Example usage and testing
async def main():
    """Example usage of the LangGraph RAG Agent"""
    # Initialize the agent
    config = RAGConfig()
    if not config.openai_api_key:
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    agent = LangGraphRAGAgent(config)
    
    # Add some sample documents (optional)
    sample_docs = [
        "LangGraph is a framework for building stateful, multi-actor applications with Large Language Models (LLMs). It extends LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner.",
        "RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them to generate more informed and accurate responses."
    ]
    
    agent.add_documents(sample_docs)
    
    # Test queries
    test_queries = [
        "What is LangGraph and how does it work?",
        "What are the latest developments in AI and machine learning?",
        "How does RAG improve language model performance?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        
        result = await agent.query(query)
        
        print(f"Answer: {result['answer']}")
        print(f"\nWeb Results Found: {len(result['web_results'])}")
        print(f"Retrieved Docs: {len(result['retrieved_docs'])}")
        print(f"Metadata: {result['metadata']}")


if __name__ == "__main__":
    asyncio.run(main())