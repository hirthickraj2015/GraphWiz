# 🇮🇪 GraphWiz Ireland

Advanced GraphRAG-powered Q&A system for Ireland knowledge using Wikipedia data, hybrid search, and Groq LLM.

## Quick Start

```bash
# 1. Setup (auto-detects UV or pip)
./setup.sh

# 2. Activate environment
source .venv/bin/activate  # or: source venv/bin/activate

# 3. Build knowledge base (2-4 hours, one-time)
python build_graphwiz.py

# 4. Launch app
streamlit run src/app.py
```

**Note:** Your GROQ_API_KEY is already configured in `.env` file.

## Features

- **Comprehensive Data**: 10,000+ Ireland Wikipedia articles with full content
- **GraphRAG**: Community detection & hierarchical summarization
- **Hybrid Search**: Semantic (FAISS) + Keyword (BM25) retrieval
- **Fast LLM**: Groq API with Llama 3.3 70B (sub-second responses)
- **Citations**: Every answer includes Wikipedia sources
- **Caching**: Instant responses for repeated questions

## Requirements

- Python 3.10+
- 16GB RAM recommended (8GB minimum)
- 15GB free storage
- GROQ_API_KEY in `.env` file (free from https://console.groq.com/)

## Project Structure

```
GraphWiz/
├── src/
│   ├── app.py                  # Streamlit UI
│   ├── wikipedia_extractor.py  # Data extraction
│   ├── text_processor.py       # Text chunking
│   ├── graphrag_builder.py     # GraphRAG builder
│   ├── hybrid_retriever.py     # Search engine
│   ├── groq_llm.py             # LLM integration
│   └── rag_engine.py           # RAG pipeline
├── build_graphwiz.py           # Pipeline orchestrator
├── setup.sh                    # One-command setup
└── requirements.txt            # Dependencies
```

## Configuration

Edit settings in `src/app.py` or via UI sidebar:
- `top_k`: Number of chunks to retrieve (3-15, default: 5)
- `semantic_weight`: Semantic search weight (0-1, default: 0.7)
- `use_community_context`: Include topic clusters (default: True)

## Example Questions

- What is the capital of Ireland?
- Tell me about the Easter Rising
- Who was Michael Collins?
- What are the provinces of Ireland?
- Explain Irish mythology

## License

MIT License
