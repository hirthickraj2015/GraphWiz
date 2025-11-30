"""
Complete RAG Engine
Integrates hybrid retrieval, GraphRAG, and Groq LLM for Ireland Q&A
"""

import json
import time
from typing import List, Dict, Optional
from hybrid_retriever import HybridRetriever, RetrievalResult
from groq_llm import GroqLLM
import hashlib


class IrelandRAGEngine:
    """Complete RAG engine for Ireland knowledge base"""

    def __init__(
        self,
        chunks_file: str = "dataset/wikipedia_ireland/chunks.json",
        graphrag_index_file: str = "dataset/wikipedia_ireland/graphrag_index.json",
        groq_api_key: Optional[str] = None,
        groq_model: str = "llama-3.3-70b-versatile",
        use_cache: bool = True
    ):
        """Initialize RAG engine"""
        print("[INFO] Initializing Ireland RAG Engine...")

        # Initialize retriever
        self.retriever = HybridRetriever(
            chunks_file=chunks_file,
            graphrag_index_file=graphrag_index_file
        )

        # Try to load pre-built indexes, otherwise build them
        try:
            self.retriever.load_indexes()
        except:
            print("[INFO] Pre-built indexes not found, building new ones...")
            self.retriever.build_semantic_index()
            self.retriever.build_keyword_index()
            self.retriever.save_indexes()

        # Initialize LLM
        self.llm = GroqLLM(api_key=groq_api_key, model=groq_model)

        # Cache for instant responses
        self.use_cache = use_cache
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        print("[SUCCESS] RAG Engine ready!")

    def _hash_query(self, query: str) -> str:
        """Create hash of query for caching"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        use_community_context: bool = True,
        return_debug_info: bool = False
    ) -> Dict:
        """
        Answer a question about Ireland using GraphRAG

        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
            use_community_context: Whether to include community summaries
            return_debug_info: Whether to return detailed debug information

        Returns:
            Dict with answer, citations, and metadata
        """
        start_time = time.time()

        # Check cache
        query_hash = self._hash_query(question)
        if self.use_cache and query_hash in self.cache:
            self.cache_hits += 1
            cached_result = self.cache[query_hash].copy()
            cached_result['cached'] = True
            cached_result['response_time'] = time.time() - start_time
            return cached_result

        self.cache_misses += 1

        # Step 1: Hybrid retrieval
        retrieval_start = time.time()
        retrieved_chunks = self.retriever.hybrid_search(
            query=question,
            top_k=top_k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight
        )
        retrieval_time = time.time() - retrieval_start

        # Step 2: Prepare contexts for LLM
        contexts = []
        for result in retrieved_chunks:
            context = {
                'text': result.text,
                'source_title': result.source_title,
                'source_url': result.source_url,
                'combined_score': result.combined_score,
                'semantic_score': result.semantic_score,
                'keyword_score': result.keyword_score,
                'community_id': result.community_id
            }
            contexts.append(context)

        # Step 3: Add community context if enabled
        community_summaries = []
        if use_community_context:
            # Get unique communities from results
            communities = set(result.community_id for result in retrieved_chunks if result.community_id >= 0)

            for comm_id in list(communities)[:2]:  # Use top 2 communities
                comm_context = self.retriever.get_community_context(comm_id)
                if comm_context:
                    community_summaries.append({
                        'community_id': comm_id,
                        'num_chunks': comm_context.get('num_chunks', 0),
                        'top_entities': [e['entity'] for e in comm_context.get('top_entities', [])[:5]],
                        'sources': comm_context.get('sources', [])[:3]
                    })

        # Step 4: Generate answer with citations
        generation_start = time.time()
        llm_result = self.llm.generate_with_citations(
            question=question,
            contexts=contexts,
            max_contexts=top_k
        )
        generation_time = time.time() - generation_start

        # Step 5: Build response
        response = {
            'question': question,
            'answer': llm_result['answer'],
            'citations': llm_result['citations'],
            'num_contexts_used': llm_result['num_contexts_used'],
            'communities': community_summaries if use_community_context else [],
            'cached': False,
            'response_time': time.time() - start_time,
            'retrieval_time': retrieval_time,
            'generation_time': generation_time
        }

        # Add debug info if requested
        if return_debug_info:
            response['debug'] = {
                'retrieved_chunks': [
                    {
                        'rank': r.rank,
                        'source': r.source_title,
                        'semantic_score': f"{r.semantic_score:.3f}",
                        'keyword_score': f"{r.keyword_score:.3f}",
                        'combined_score': f"{r.combined_score:.3f}",
                        'community': r.community_id,
                        'text_preview': r.text[:150] + "..."
                    }
                    for r in retrieved_chunks
                ],
                'cache_stats': {
                    'hits': self.cache_hits,
                    'misses': self.cache_misses,
                    'hit_rate': f"{self.cache_hits / (self.cache_hits + self.cache_misses) * 100:.1f}%" if (self.cache_hits + self.cache_misses) > 0 else "0%"
                }
            }

        # Cache the response
        if self.use_cache:
            self.cache[query_hash] = response.copy()

        return response

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_queries = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_queries * 100) if total_queries > 0 else 0

        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_queries': total_queries,
            'hit_rate': f"{hit_rate:.1f}%"
        }

    def clear_cache(self):
        """Clear the response cache"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        print("[INFO] Cache cleared")

    def get_stats(self) -> Dict:
        """Get engine statistics"""
        return {
            'total_chunks': len(self.retriever.chunks),
            'total_communities': len(self.retriever.graphrag_index['communities']),
            'cache_stats': self.get_cache_stats()
        }


if __name__ == "__main__":
    # Test RAG engine
    engine = IrelandRAGEngine()

    # Test questions
    questions = [
        "What is the capital of Ireland?",
        "When did Ireland join the European Union?",
        "Who is the current president of Ireland?",
        "What is the oldest university in Ireland?"
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print(f"Question: {question}")
        print("=" * 80)

        result = engine.answer_question(question, top_k=5, return_debug_info=True)

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nResponse Time: {result['response_time']:.2f}s")
        print(f"  - Retrieval: {result['retrieval_time']:.2f}s")
        print(f"  - Generation: {result['generation_time']:.2f}s")

        print(f"\nCitations:")
        for cite in result['citations']:
            print(f"  [{cite['id']}] {cite['source']} (score: {cite['relevance_score']:.3f})")

        if result.get('communities'):
            print(f"\nRelated Topics:")
            for comm in result['communities']:
                print(f"  - {', '.join(comm['top_entities'][:3])}")

    print("\n" + "=" * 80)
    print("Cache Stats:", engine.get_cache_stats())
    print("=" * 80)
