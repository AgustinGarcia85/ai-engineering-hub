#!/usr/bin/env python3
"""
Command Line Interface for LangGraph RAG Agent
"""

import asyncio
import argparse
import sys
import os
from typing import List
from langgraph_rag_agent import LangGraphRAGAgent, RAGConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CLIAgent:
    """Command Line Interface for the RAG Agent"""
    
    def __init__(self):
        self.agent = None
        self.conversation_history = []
    
    def initialize_agent(self, config: RAGConfig):
        """Initialize the RAG agent"""
        try:
            self.agent = LangGraphRAGAgent(config)
            print("✅ RAG Agent initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Error initializing RAG Agent: {str(e)}")
            return False
    
    async def interactive_mode(self):
        """Start interactive chat mode"""
        print("\n🤖 LangGraph RAG Agent - Interactive Mode")
        print("=" * 50)
        print("Type 'quit', 'exit', or 'q' to stop")
        print("Type 'clear' to clear conversation history")
        print("Type 'help' for more commands")
        print("=" * 50)
        
        while True:
            try:
                query = input("\n👤 You: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if query.lower() == 'clear':
                    self.conversation_history = []
                    print("🗑️ Conversation history cleared!")
                    continue
                
                if query.lower() == 'help':
                    self.show_help()
                    continue
                
                if not query:
                    continue
                
                print("\n🔍 Processing your query...")
                result = await self.agent.query(query, self.conversation_history)
                
                print(f"\n🤖 Assistant: {result['answer']}")
                
                # Show metadata
                metadata = result['metadata']
                print(f"\n📊 Metadata:")
                print(f"   • Web Results: {metadata.get('web_search_count', 0)}")
                print(f"   • Retrieved Docs: {metadata.get('retrieved_docs_count', 0)}")
                print(f"   • Quality Score: {metadata.get('quality_score', 0)}/3")
                
                if metadata.get('quality_issues'):
                    print(f"   • Issues: {', '.join(metadata['quality_issues'])}")
                
                # Update conversation history
                self.conversation_history.append({
                    "human": query,
                    "assistant": result['answer']
                })
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
    
    async def single_query(self, query: str):
        """Process a single query"""
        print(f"\n🔍 Processing query: {query}")
        
        try:
            result = await self.agent.query(query, self.conversation_history)
            
            print(f"\n🤖 Answer:")
            print(result['answer'])
            
            print(f"\n📊 Metadata:")
            metadata = result['metadata']
            print(f"   • Web Results: {metadata.get('web_search_count', 0)}")
            print(f"   • Retrieved Docs: {metadata.get('retrieved_docs_count', 0)}")
            print(f"   • Quality Score: {metadata.get('quality_score', 0)}/3")
            
            if metadata.get('quality_issues'):
                print(f"   • Issues: {', '.join(metadata['quality_issues'])}")
            
        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
    
    def add_documents_from_files(self, file_paths: List[str]):
        """Add documents from text files"""
        documents = []
        metadatas = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
                    metadatas.append({"source": file_path})
                    print(f"✅ Loaded: {file_path}")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {str(e)}")
        
        if documents:
            self.agent.add_documents(documents, metadatas)
            print(f"\n📚 Added {len(documents)} documents to the knowledge base")
    
    def add_documents_from_urls(self, urls: List[str]):
        """Add documents from URLs"""
        try:
            self.agent.add_documents_from_urls(urls)
            print(f"\n🌐 Processed {len(urls)} URLs")
        except Exception as e:
            print(f"❌ Error processing URLs: {str(e)}")
    
    def show_help(self):
        """Show help information"""
        help_text = """
🤖 LangGraph RAG Agent - Help

Available Commands:
  quit, exit, q    - Exit the application
  clear           - Clear conversation history
  help            - Show this help message

Features:
  • Web Search    - Searches the web for current information
  • Knowledge Base - Retrieves from your local document collection
  • Conversation  - Maintains context across multiple questions
  • Quality Check - Evaluates answer quality and completeness

Tips:
  • Ask specific questions for better results
  • The agent combines web search with your documents
  • Conversation history helps with follow-up questions
  • Use 'clear' if you want to start a fresh conversation
        """
        print(help_text)

def main():
    parser = argparse.ArgumentParser(description="LangGraph RAG Agent CLI")
    parser.add_argument("--query", "-q", type=str, help="Single query to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive mode")
    parser.add_argument("--files", "-f", nargs="+", help="Text files to add to knowledge base")
    parser.add_argument("--urls", "-u", nargs="+", help="URLs to add to knowledge base")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-small", help="Embedding model to use")
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Initialize configuration
    config = RAGConfig(
        openai_api_key=api_key,
        model_name=args.model,
        embedding_model=args.embedding_model
    )
    
    # Initialize CLI agent
    cli = CLIAgent()
    if not cli.initialize_agent(config):
        sys.exit(1)
    
    # Add documents from files if provided
    if args.files:
        cli.add_documents_from_files(args.files)
    
    # Add documents from URLs if provided
    if args.urls:
        cli.add_documents_from_urls(args.urls)
    
    # Process single query
    if args.query:
        asyncio.run(cli.single_query(args.query))
    
    # Start interactive mode
    elif args.interactive or not args.query:
        asyncio.run(cli.interactive_mode())

if __name__ == "__main__":
    main()