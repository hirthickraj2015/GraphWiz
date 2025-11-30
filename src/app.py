"""
GraphWiz Ireland - Advanced GraphRAG Chat Application
Complete rewrite with hybrid search, GraphRAG, Groq LLM, and instant responses
"""

import streamlit as st
import os
import time
from rag_engine import IrelandRAGEngine
import json
from pathlib import Path

# Load environment variables from .env file
env_file = Path(__file__).parent.parent / '.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()


# Page configuration
st.set_page_config(
    page_title="GraphWiz Ireland - Intelligent Q&A",
    page_icon="🇮🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5em;
        background: linear-gradient(90deg, #169B62 0%, #FF883E 50%, #FFFFFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .answer-box {
        background-color: #f0f7f4;
        color: #1a1a1a;
        padding: 1.5em;
        border-radius: 10px;
        border-left: 5px solid #169B62;
        margin: 1em 0;
    }
    .citation-box {
        background-color: #f8f9fa;
        color: #2c3e50;
        padding: 0.5em;
        border-radius: 5px;
        margin: 0.3em 0;
        font-size: 0.9em;
    }
    .metric-card {
        background-color: #ffffff;
        color: #1a1a1a;
        padding: 1em;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #169B62;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
        border: none;
    }
    .stButton>button:hover {
        background-color: #127a4d;
    }
</style>
""", unsafe_allow_html=True)


# Initialize RAG Engine (cached)
@st.cache_resource
def load_rag_engine():
    """Load and cache RAG engine"""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("⚠️ GROQ_API_KEY not found in environment variables. Please set it to use the application.")
            st.info("Get your free API key at: https://console.groq.com/")
            st.stop()

        engine = IrelandRAGEngine(
            chunks_file="dataset/wikipedia_ireland/chunks.json",
            graphrag_index_file="dataset/wikipedia_ireland/graphrag_index.json",
            groq_api_key=groq_api_key,
            groq_model="llama-3.3-70b-versatile",
            use_cache=True
        )
        return engine
    except FileNotFoundError as e:
        st.error(f"⚠️ Data files not found: {e}")
        st.info("Please run the data extraction and processing pipeline first:\n"
                "1. python src/wikipedia_extractor.py\n"
                "2. python src/text_processor.py\n"
                "3. python src/graphrag_builder.py\n"
                "4. python src/hybrid_retriever.py")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading RAG engine: {e}")
        st.stop()


# Main header
st.markdown('<h1 class="main-header">🇮🇪 GraphWiz Ireland</h1>', unsafe_allow_html=True)
st.markdown("""
<p style="text-align: center; font-size: 1.2em; color: #666; margin-bottom: 2em;">
    Intelligent Q&A System powered by GraphRAG, Hybrid Search, and Groq LLM
</p>
""", unsafe_allow_html=True)

# Load RAG engine
with st.spinner("🚀 Loading GraphWiz Engine..."):
    engine = load_rag_engine()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # Retrieval settings
    st.markdown("#### Retrieval Configuration")
    top_k = st.slider("Number of sources to retrieve", 3, 15, 5, help="More sources = more context but slower")
    semantic_weight = st.slider("Semantic search weight", 0.0, 1.0, 0.7, 0.1, help="Higher = prioritize meaning over keywords")
    keyword_weight = 1.0 - semantic_weight

    # Advanced options
    with st.expander("Advanced Options"):
        use_community = st.checkbox("Use community context", value=True, help="Include related topic clusters")
        show_debug = st.checkbox("Show debug information", value=False, help="Display retrieval details")

    st.markdown("---")

    # Statistics
    st.markdown("#### 📊 System Statistics")
    stats = engine.get_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Knowledge Chunks", f"{stats['total_chunks']:,}")
    with col2:
        st.metric("Topic Communities", stats['total_communities'])

    cache_stats = stats['cache_stats']
    st.metric("Cache Hit Rate", cache_stats['hit_rate'])
    st.caption(f"Hits: {cache_stats['cache_hits']} | Misses: {cache_stats['cache_misses']}")

    if st.button("🗑️ Clear Cache"):
        engine.clear_cache()
        st.success("Cache cleared!")
        st.rerun()

    st.markdown("---")

    # Info
    st.markdown("#### ℹ️ About")
    st.info("""
    **GraphWiz Ireland** uses:
    - 🔍 Hybrid search (semantic + keyword)
    - 🕸️ GraphRAG with community detection
    - ⚡ Groq LLM (ultra-fast inference)
    - 💾 Smart caching for instant responses
    - 📚 Comprehensive Wikipedia data
    """)

    st.markdown("---")
    st.caption("Built with Streamlit, FAISS, NetworkX, Groq, and spaCy")


# Suggested questions
st.markdown("### 💡 Try These Questions")
suggested_questions = [
    "What is the capital of Ireland?",
    "When did Ireland join the European Union?",
    "Who is the current president of Ireland?",
    "What is the oldest university in Ireland?",
    "Tell me about the history of Dublin",
    "What are the major cities in Ireland?",
    "Explain the Irish language and its history",
    "What is Ireland's economy based on?",
    "Describe Irish mythology and folklore",
    "What are the main political parties in Ireland?"
]

# Display suggested questions as buttons in columns
cols = st.columns(3)
for idx, question in enumerate(suggested_questions):
    with cols[idx % 3]:
        if st.button(question, key=f"suggested_{idx}", use_container_width=True):
            st.session_state.question = question

# Question input
st.markdown("### 🔍 Ask Your Question")
question = st.text_input(
    "Enter your question about Ireland:",
    value=st.session_state.get('question', ''),
    placeholder="e.g., What is the history of Irish independence?",
    key="question_input"
)

# Search button and results
if st.button("🔎 Search", type="primary") or question:
    if question and question.strip():
        # Display searching indicator
        with st.spinner("🔍 Searching knowledge base..."):
            # Query the RAG engine
            result = engine.answer_question(
                question=question,
                top_k=top_k,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                use_community_context=use_community,
                return_debug_info=show_debug
            )

        # Display results
        st.markdown("---")

        # Response time and cache status
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            cache_indicator = "💾 Cached" if result['cached'] else "🔄 Fresh"
            st.caption(f"{cache_indicator} | Response time: {result['response_time']:.2f}s")
        with col2:
            st.caption(f"Retrieval: {result['retrieval_time']:.2f}s")
        with col3:
            st.caption(f"Generation: {result['generation_time']:.2f}s")

        # Answer
        st.markdown("### 💬 Answer")
        st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

        # Citations
        st.markdown("### 📚 Citations & Sources")
        for cite in result['citations']:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    f'<div class="citation-box">'
                    f'<strong>[{cite["id"]}]</strong> '
                    f'<a href="{cite["url"]}" target="_blank">{cite["source"]}</a>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col2:
                st.caption(f"Score: {cite['relevance_score']:.3f}")

        # Related topics (communities)
        if result.get('communities'):
            st.markdown("### 🏷️ Related Topics")
            for comm in result['communities']:
                st.info(f"**Topic Cluster:** {', '.join(comm['top_entities'])}")

        # Debug information
        if show_debug and result.get('debug'):
            st.markdown("---")
            st.markdown("### 🔧 Debug Information")

            with st.expander("Retrieved Chunks Details", expanded=False):
                for chunk in result['debug']['retrieved_chunks']:
                    st.markdown(f"""
                    **Rank {chunk['rank']}:** {chunk['source']}
                    - Semantic: {chunk['semantic_score']} | Keyword: {chunk['keyword_score']} | Combined: {chunk['combined_score']}
                    - Community: {chunk['community']}
                    - Preview: {chunk['text_preview']}
                    """)
                    st.markdown("---")

            cache_stats = result['debug']['cache_stats']
            st.metric("Overall Cache Hit Rate", cache_stats['hit_rate'])

    else:
        st.warning("⚠️ Please enter a question to search.")

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #666; font-size: 0.9em;">
    GraphWiz Ireland | Powered by Wikipedia, GraphRAG, and Groq |
    <a href="https://github.com/yourusername/graphwiz" target="_blank">GitHub</a>
</p>
""", unsafe_allow_html=True)
