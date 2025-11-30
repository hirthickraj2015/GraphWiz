#!/usr/bin/env python3
"""
GraphWiz Ireland - Complete Pipeline Orchestrator
Runs the entire data extraction, processing, and indexing pipeline
"""

import sys
import os

# Fix macOS threading conflicts - MUST be set before importing numerical libraries
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['VECLIB_MAXIMUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

import time
import json
from datetime import datetime

# Load environment variables from .env file
from pathlib import Path
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def print_banner(text):
    """Print a fancy banner"""
    line = "=" * 80
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")


def check_environment():
    """Check if the environment is set up correctly"""
    print_banner("STEP 0: Environment Check")

    # Check if GROQ_API_KEY is set
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("❌ GROQ_API_KEY environment variable not set!")
        print("\n📝 To fix this:")
        print("   1. Get a free API key from: https://console.groq.com/")
        print("   2. Set the environment variable:")
        print("      - Linux/Mac: export GROQ_API_KEY='your-key-here'")
        print("      - Windows: set GROQ_API_KEY=your-key-here")
        print("\n   Or add it to a .env file in the project root.")
        return False
    else:
        print("✅ GROQ_API_KEY is set")

    # Check if required directories exist
    required_dirs = ["src", "dataset"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"📁 Created directory: {dir_name}")
        else:
            print(f"✅ Directory exists: {dir_name}")

    # Check Python version
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ required, you have {sys.version_info.major}.{sys.version_info.minor}")
        return False
    else:
        print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    return True


def step1_extract_wikipedia():
    """Step 1: Extract Wikipedia articles about Ireland"""
    print_banner("STEP 1: Wikipedia Data Extraction")
    print("This will extract ALL Ireland-related Wikipedia articles.")
    print("Estimated time: 2-4 hours depending on network speed")
    print("Estimated storage: 5-10 GB")

    # Check for existing checkpoint or completed data
    import os.path
    checkpoint_file = "dataset/wikipedia_ireland/checkpoint_articles.json"
    final_file = "dataset/wikipedia_ireland/ireland_articles.json"
    progress_file = "dataset/wikipedia_ireland/extraction_progress.json"

    if os.path.exists(final_file):
        print("✅ Data already extracted, skipping")
        return True

    if os.path.exists(checkpoint_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        print(f"📍 CHECKPOINT FOUND: {progress['completed']}/{progress['total']} articles")
        print(f"   Resuming extraction from checkpoint...")
    else:
        print("\n→ Starting fresh extraction with auto-checkpoint every 100 articles...")

    start_time = time.time()

    try:
        from src.wikipedia_extractor import IrelandWikipediaExtractor

        extractor = IrelandWikipediaExtractor(output_dir="dataset/wikipedia_ireland")
        articles = extractor.run_full_extraction()

        elapsed = time.time() - start_time
        print(f"\n✅ Wikipedia extraction completed in {elapsed/60:.1f} minutes")
        print(f"   Extracted {len(articles)} articles")
        return True

    except KeyboardInterrupt:
        print(f"\n⚠️  Extraction interrupted by user")
        print(f"   Progress saved to checkpoint file")
        print(f"   Run again to resume from checkpoint")
        return False
    except Exception as e:
        print(f"\n❌ Wikipedia extraction failed: {e}")
        print(f"   Progress saved to checkpoint file (if any)")
        print(f"   Run again to resume from checkpoint")
        return False


def step2_process_text():
    """Step 2: Process and chunk text"""
    print_banner("STEP 2: Text Processing and Chunking")
    print("This will process articles into intelligent chunks with entity extraction.")
    print("Estimated time: 30-60 minutes")

    # Check if already done
    import os.path
    if os.path.exists("dataset/wikipedia_ireland/chunks.json"):
        print("✅ Chunks already created, skipping")
        return True

    print("\n→ Starting text processing...")

    start_time = time.time()

    try:
        from src.text_processor import AdvancedTextProcessor
        import json

        # Load articles
        articles_file = "dataset/wikipedia_ireland/ireland_articles.json"
        if not os.path.exists(articles_file):
            print(f"❌ Articles file not found: {articles_file}")
            print("   Please run Step 1 (Wikipedia extraction) first")
            return False

        with open(articles_file, 'r') as f:
            articles = json.load(f)

        processor = AdvancedTextProcessor(chunk_size=512, chunk_overlap=128)
        chunks = processor.process_all_articles(articles)
        processor.save_chunks(chunks, output_path="dataset/wikipedia_ireland/chunks.json")

        elapsed = time.time() - start_time
        print(f"\n✅ Text processing completed in {elapsed/60:.1f} minutes")
        print(f"   Created {len(chunks)} chunks")
        return True

    except Exception as e:
        print(f"\n❌ Text processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step3_build_graphrag():
    """Step 3: Build GraphRAG index"""
    print_banner("STEP 3: GraphRAG Index Construction")
    print("This will build the GraphRAG index with community detection.")
    print("Estimated time: 20-40 minutes")

    # Check if already done
    import os.path
    if os.path.exists("dataset/wikipedia_ireland/graphrag_index.json"):
        print("✅ GraphRAG index already built, skipping")
        return True

    print("\n→ Starting GraphRAG construction...")

    start_time = time.time()

    try:
        from src.graphrag_builder import GraphRAGBuilder

        chunks_file = "dataset/wikipedia_ireland/chunks.json"
        if not os.path.exists(chunks_file):
            print(f"❌ Chunks file not found: {chunks_file}")
            print("   Please run Step 2 (Text processing) first")
            return False

        builder = GraphRAGBuilder(
            chunks_file=chunks_file,
            output_dir="dataset/wikipedia_ireland"
        )

        graphrag_index = builder.build_hierarchical_index()
        builder.save_graphrag_index(graphrag_index)

        elapsed = time.time() - start_time
        print(f"\n✅ GraphRAG index built in {elapsed/60:.1f} minutes")
        return True

    except Exception as e:
        print(f"\n❌ GraphRAG building failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step4_build_hybrid_index():
    """Step 4: Build hybrid retrieval indexes"""
    print_banner("STEP 4: Hybrid Search Index Construction")
    print("This will build HNSW semantic index and BM25 keyword index.")
    print("Estimated time: 5-10 minutes")

    # Check if already done
    import os.path
    if os.path.exists("dataset/wikipedia_ireland/hybrid_hnsw_index.bin"):
        print("✅ Hybrid indexes already built, skipping")
        return True

    print("\n→ Starting hybrid index construction...")

    start_time = time.time()

    try:
        from src.hybrid_retriever import HybridRetriever

        chunks_file = "dataset/wikipedia_ireland/chunks.json"
        graphrag_file = "dataset/wikipedia_ireland/graphrag_index.json"

        if not os.path.exists(chunks_file):
            print(f"❌ Chunks file not found: {chunks_file}")
            return False
        if not os.path.exists(graphrag_file):
            print(f"❌ GraphRAG index not found: {graphrag_file}")
            return False

        retriever = HybridRetriever(
            chunks_file=chunks_file,
            graphrag_index_file=graphrag_file
        )

        retriever.build_semantic_index()
        retriever.build_keyword_index()
        retriever.save_indexes(output_dir="dataset/wikipedia_ireland")

        elapsed = time.time() - start_time
        print(f"\n✅ Hybrid indexes built in {elapsed/60:.1f} minutes")
        return True

    except Exception as e:
        print(f"\n❌ Hybrid index building failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step5_test_system():
    """Step 5: Test the complete system"""
    print_banner("STEP 5: System Testing")
    print("Running end-to-end tests...\n")

    try:
        from src.rag_engine import IrelandRAGEngine

        groq_api_key = os.getenv("GROQ_API_KEY")
        engine = IrelandRAGEngine(
            chunks_file="dataset/wikipedia_ireland/chunks.json",
            graphrag_index_file="dataset/wikipedia_ireland/graphrag_index.json",
            groq_api_key=groq_api_key
        )

        # Test question
        test_question = "What is the capital of Ireland?"
        print(f"Test question: {test_question}\n")

        result = engine.answer_question(test_question, top_k=3)

        print(f"Answer: {result['answer']}\n")
        print(f"Response time: {result['response_time']:.2f}s")
        print(f"Citations: {len(result['citations'])}")
        print(f"\n✅ System test passed!")

        return True

    except Exception as e:
        print(f"\n❌ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main pipeline orchestrator"""
    print("\n" + "=" * 80)
    print("  🇮🇪 GRAPHWIZ IRELAND - COMPLETE PIPELINE")
    print("  Advanced GraphRAG System Builder")
    print("=" * 80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    pipeline_start = time.time()

    # Step 0: Environment check
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues and try again.")
        sys.exit(1)

    # Pipeline steps
    steps = [
        ("Wikipedia Extraction", step1_extract_wikipedia),
        ("Text Processing", step2_process_text),
        ("GraphRAG Building", step3_build_graphrag),
        ("Hybrid Index Building", step4_build_hybrid_index),
        ("System Testing", step5_test_system)
    ]

    completed_steps = 0
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Pipeline failed at: {step_name}")
            print(f"   Completed {completed_steps}/{len(steps)} steps")
            sys.exit(1)
        completed_steps += 1

    # Success!
    pipeline_elapsed = time.time() - pipeline_start
    print_banner("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total time: {pipeline_elapsed/3600:.1f} hours ({pipeline_elapsed/60:.1f} minutes)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📝 Next steps:")
    print("   1. Set your GROQ_API_KEY if not already set")
    print("   2. Run the Streamlit app:")
    print("      streamlit run src/app.py")
    print("\n   Or test the RAG engine:")
    print("      python src/rag_engine.py")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
