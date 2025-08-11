import streamlit as st
import asyncio
import os
from typing import List, Dict
from langgraph_rag_agent import LangGraphRAGAgent, RAGConfig
import time

# Set page config
st.set_page_config(
    page_title="LangGraph RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1e88e5;
        background-color: #f8f9fa;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #1976d2;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left-color: #388e3c;
    }
    .metadata-box {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "rag_agent" not in st.session_state:
        st.session_state.rag_agent = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

def setup_agent():
    """Setup the RAG agent with configuration"""
    config = RAGConfig(
        openai_api_key=st.session_state.get("openai_api_key", ""),
        model_name=st.session_state.get("model_name", "gpt-4o-mini"),
        embedding_model=st.session_state.get("embedding_model", "text-embedding-3-small"),
        chunk_size=st.session_state.get("chunk_size", 1000),
        chunk_overlap=st.session_state.get("chunk_overlap", 200),
        max_search_results=st.session_state.get("max_search_results", 5)
    )
    
    try:
        st.session_state.rag_agent = LangGraphRAGAgent(config)
        st.success("✅ RAG Agent initialized successfully!")
        return True
    except Exception as e:
        st.error(f"❌ Error initializing RAG Agent: {str(e)}")
        return False

def display_chat_message(message: Dict, is_user: bool = True):
    """Display a chat message with proper styling"""
    message_class = "user-message" if is_user else "assistant-message"
    role = "👤 You" if is_user else "🤖 Assistant"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <strong>{role}:</strong><br>
        {message.get('content', '')}
    </div>
    """, unsafe_allow_html=True)

def display_metadata(metadata: Dict):
    """Display metadata information"""
    if metadata:
        with st.expander("📊 Query Metadata", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Web Results", 
                    metadata.get("web_search_count", 0),
                    help="Number of web search results found"
                )
            
            with col2:
                st.metric(
                    "Retrieved Docs", 
                    metadata.get("retrieved_docs_count", 0),
                    help="Number of documents retrieved from vector store"
                )
            
            with col3:
                quality_score = metadata.get("quality_score", 0)
                st.metric(
                    "Quality Score", 
                    f"{quality_score}/3",
                    help="Quality assessment of the generated answer"
                )
            
            if metadata.get("quality_issues"):
                st.warning("⚠️ Quality Issues: " + ", ".join(metadata["quality_issues"]))

async def process_query(query: str):
    """Process a user query through the RAG agent"""
    if not st.session_state.rag_agent:
        st.error("Please initialize the RAG agent first!")
        return None
    
    try:
        # Add user message to chat
        st.session_state.chat_messages.append({"role": "user", "content": query})
        
        # Process query
        with st.spinner("🔍 Searching web and knowledge base..."):
            result = await st.session_state.rag_agent.query(
                query, 
                st.session_state.conversation_history
            )
        
        # Add assistant response to chat
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": result["answer"],
            "metadata": result["metadata"]
        })
        
        # Update conversation history
        st.session_state.conversation_history.append({
            "human": query,
            "assistant": result["answer"]
        })
        
        return result
    
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 LangGraph RAG Agent</h1>', unsafe_allow_html=True)
    st.markdown("### AI Assistant with Web Search & Knowledge Base")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.get("openai_api_key", ""),
            help="Enter your OpenAI API key"
        )
        st.session_state.openai_api_key = openai_api_key
        
        # Model settings
        st.subheader("🤖 Model Settings")
        st.session_state.model_name = st.selectbox(
            "Chat Model",
            ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=1
        )
        
        st.session_state.embedding_model = st.selectbox(
            "Embedding Model",
            ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
            index=0
        )
        
        # RAG settings
        st.subheader("📚 RAG Settings")
        st.session_state.chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
        st.session_state.chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200)
        st.session_state.max_search_results = st.slider("Max Web Results", 3, 10, 5)
        
        # Initialize agent button
        if st.button("🚀 Initialize Agent", type="primary"):
            if not openai_api_key:
                st.error("Please provide an OpenAI API key!")
            else:
                setup_agent()
        
        # Agent status
        if st.session_state.rag_agent:
            st.success("✅ Agent Ready")
        else:
            st.warning("⚠️ Agent Not Initialized")
        
        # Document management
        st.header("📄 Document Management")
        
        # Add documents from text
        with st.expander("Add Text Documents"):
            doc_text = st.text_area("Document Content", height=100)
            doc_metadata = st.text_input("Metadata (JSON format)", placeholder='{"source": "manual"}')
            
            if st.button("Add Document"):
                if doc_text and st.session_state.rag_agent:
                    try:
                        import json
                        metadata = json.loads(doc_metadata) if doc_metadata else {}
                        st.session_state.rag_agent.add_documents([doc_text], [metadata])
                        st.success("Document added successfully!")
                    except Exception as e:
                        st.error(f"Error adding document: {str(e)}")
        
        # Add documents from URLs
        with st.expander("Add from URLs"):
            urls_text = st.text_area("URLs (one per line)", height=100)
            
            if st.button("Process URLs"):
                if urls_text and st.session_state.rag_agent:
                    urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
                    try:
                        st.session_state.rag_agent.add_documents_from_urls(urls)
                        st.success(f"Processed {len(urls)} URLs!")
                    except Exception as e:
                        st.error(f"Error processing URLs: {str(e)}")
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.chat_messages = []
            st.session_state.conversation_history = []
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat messages
        if st.session_state.chat_messages:
            for message in st.session_state.chat_messages:
                is_user = message["role"] == "user"
                display_chat_message(message, is_user)
                
                # Display metadata for assistant messages
                if not is_user and "metadata" in message:
                    display_metadata(message["metadata"])
        else:
            st.info("👋 Welcome! Ask me anything and I'll search the web and my knowledge base to help you.")
        
        # Query input
        with st.form("query_form", clear_on_submit=True):
            query = st.text_input(
                "Your Question:",
                placeholder="Ask me anything...",
                help="Enter your question and I'll search both the web and my knowledge base for answers."
            )
            submit_button = st.form_submit_button("Send", type="primary")
        
        if submit_button and query:
            if not st.session_state.rag_agent:
                st.error("Please initialize the RAG agent first!")
            else:
                # Process query asynchronously
                result = asyncio.run(process_query(query))
                if result:
                    st.rerun()
    
    with col2:
        st.header("📊 Statistics")
        
        # Conversation stats
        total_messages = len(st.session_state.chat_messages)
        user_messages = len([m for m in st.session_state.chat_messages if m["role"] == "user"])
        
        st.metric("Total Messages", total_messages)
        st.metric("Your Questions", user_messages)
        st.metric("Agent Responses", total_messages - user_messages)
        
        # Recent metadata
        if st.session_state.chat_messages:
            recent_messages = [m for m in st.session_state.chat_messages[-5:] if m["role"] == "assistant"]
            if recent_messages:
                st.subheader("Recent Performance")
                for i, msg in enumerate(recent_messages):
                    metadata = msg.get("metadata", {})
                    with st.expander(f"Query {len(recent_messages) - i}"):
                        st.json(metadata)

if __name__ == "__main__":
    main()