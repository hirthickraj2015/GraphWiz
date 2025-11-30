"""
Hybrid Retrieval System
Combines semantic search (HNSW) with keyword search (BM25) for optimal retrieval
"""

import json
import numpy as np
import hnswlib
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Represents a retrieval result with metadata"""
    chunk_id: str
    text: str
    source_title: str
    source_url: str
    semantic_score: float
    keyword_score: float
    combined_score: float
    community_id: int
    rank: int


class HybridRetriever:
    """Hybrid retrieval combining semantic and keyword search"""

    def __init__(
        self,
        chunks_file: str,
        graphrag_index_file: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384
    ):
        self.chunks_file = chunks_file
        self.graphrag_index_file = graphrag_index_file
        self.embedding_dim = embedding_dim

        # Load components
        print("[INFO] Loading hybrid retriever components...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunks = self._load_chunks()
        self.graphrag_index = self._load_graphrag_index()

        # Build indexes
        self.hnsw_index = None
        self.bm25 = None
        self.chunk_embeddings = None

        print("[SUCCESS] Hybrid retriever initialized")

    def _load_chunks(self) -> List[Dict]:
        """Load chunks from file"""
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"[INFO] Loaded {len(chunks)} chunks")
        return chunks

    def _load_graphrag_index(self) -> Dict:
        """Load GraphRAG index"""
        with open(self.graphrag_index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)
        print(f"[INFO] Loaded GraphRAG index with {index['metadata']['total_communities']} communities")
        return index

    def build_semantic_index(self):
        """Build HNSW semantic search index"""
        print("[INFO] Building semantic index with HNSW...")

        # Generate embeddings for all chunks
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        print(f"[INFO] Generating embeddings for {len(chunk_texts)} chunks...")

        self.chunk_embeddings = self.embedding_model.encode(
            chunk_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )

        # Build HNSW index with optimized parameters
        import time
        n_chunks = len(self.chunks)

        print(f"[INFO] Building HNSW index for {n_chunks} chunks...")
        start_build = time.time()

        # Initialize HNSW index
        # ef_construction: controls index build time/accuracy tradeoff (higher = more accurate but slower)
        # M: number of bi-directional links per element (higher = better recall but more memory)
        self.hnsw_index = hnswlib.Index(space='cosine', dim=self.embedding_dim)

        # For 86K vectors, optimal parameters for speed + accuracy:
        # M=64 gives excellent recall with reasonable memory
        # ef_construction=200 balances build time and quality
        self.hnsw_index.init_index(
            max_elements=n_chunks,
            ef_construction=200,  # Higher = better quality, slower build
            M=64,  # Higher = better recall, more memory
            random_seed=42
        )

        # Set number of threads for parallel insertion
        self.hnsw_index.set_num_threads(8)

        # Add all vectors to index
        print(f"[INFO] Adding {n_chunks} vectors to index (using 8 threads)...")
        self.hnsw_index.add_items(self.chunk_embeddings, np.arange(n_chunks))

        build_time = time.time() - start_build
        print(f"[SUCCESS] HNSW index built in {build_time:.1f} seconds ({build_time/60:.2f} minutes)")
        print(f"[SUCCESS] Index contains {self.hnsw_index.get_current_count()} vectors")

    def build_keyword_index(self):
        """Build BM25 keyword search index"""
        print("[INFO] Building BM25 keyword index...")

        # Tokenize chunks for BM25
        tokenized_chunks = [chunk['text'].lower().split() for chunk in self.chunks]

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_chunks)

        print(f"[SUCCESS] BM25 index built for {len(tokenized_chunks)} chunks")

    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Semantic search using HNSW"""
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Set ef (exploration factor) for search - higher = more accurate but slower
        # For maximum accuracy, set ef = top_k * 2
        self.hnsw_index.set_ef(max(top_k * 2, 100))

        # Search in HNSW index
        indices, distances = self.hnsw_index.knn_query(query_embedding, k=top_k)

        # Convert cosine distances to similarity scores (1 - distance)
        # HNSW returns distances, we want similarities
        scores = 1 - distances[0]

        # Return (index, score) tuples
        results = [(int(idx), float(score)) for idx, score in zip(indices[0], scores)]
        return results

    def keyword_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Keyword search using BM25"""
        # Tokenize query
        query_tokens = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Return (index, score) tuples
        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        rerank: bool = True
    ) -> List[RetrievalResult]:
        """
        Hybrid search combining semantic and keyword search

        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic scores (0-1)
            keyword_weight: Weight for keyword scores (0-1)
            rerank: Whether to rerank by community relevance
        """
        # Get results from both search methods
        semantic_results = self.semantic_search(query, top_k * 2)  # Get more for fusion
        keyword_results = self.keyword_search(query, top_k * 2)

        # Normalize scores to [0, 1] range
        def normalize_scores(results):
            if not results:
                return []
            scores = [score for _, score in results]
            min_score, max_score = min(scores), max(scores)
            if max_score == min_score:
                return [(idx, 1.0) for idx, _ in results]
            return [(idx, (score - min_score) / (max_score - min_score))
                   for idx, score in results]

        semantic_results = normalize_scores(semantic_results)
        keyword_results = normalize_scores(keyword_results)

        # Combine scores using reciprocal rank fusion
        combined_scores = {}

        for idx, score in semantic_results:
            combined_scores[idx] = {
                'semantic': score * semantic_weight,
                'keyword': 0.0,
                'combined': score * semantic_weight
            }

        for idx, score in keyword_results:
            if idx in combined_scores:
                combined_scores[idx]['keyword'] = score * keyword_weight
                combined_scores[idx]['combined'] += score * keyword_weight
            else:
                combined_scores[idx] = {
                    'semantic': 0.0,
                    'keyword': score * keyword_weight,
                    'combined': score * keyword_weight
                }

        # Sort by combined score
        sorted_indices = sorted(
            combined_scores.items(),
            key=lambda x: x[1]['combined'],
            reverse=True
        )[:top_k]

        # Build retrieval results
        results = []
        for rank, (idx, scores) in enumerate(sorted_indices):
            chunk = self.chunks[idx]
            community_id = self.graphrag_index['node_to_community'].get(chunk['chunk_id'], -1)

            result = RetrievalResult(
                chunk_id=chunk['chunk_id'],
                text=chunk['text'],
                source_title=chunk['source_title'],
                source_url=chunk['source_url'],
                semantic_score=scores['semantic'],
                keyword_score=scores['keyword'],
                combined_score=scores['combined'],
                community_id=community_id,
                rank=rank + 1
            )
            results.append(result)

        return results

    def get_community_context(self, community_id: int) -> Dict:
        """Get context from a community"""
        if str(community_id) in self.graphrag_index['communities']:
            return self.graphrag_index['communities'][str(community_id)]
        return {}

    def save_indexes(self, output_dir: str = "dataset/wikipedia_ireland"):
        """Save indexes for fast loading"""
        print("[INFO] Saving indexes...")

        # Save HNSW index
        self.hnsw_index.save_index(f"{output_dir}/hybrid_hnsw_index.bin")

        # Save BM25 and embeddings
        with open(f"{output_dir}/hybrid_indexes.pkl", 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'embeddings': self.chunk_embeddings
            }, f)

        print(f"[SUCCESS] Indexes saved to {output_dir}")

    def load_indexes(self, output_dir: str = "dataset/wikipedia_ireland"):
        """Load pre-built indexes"""
        print("[INFO] Loading pre-built indexes...")

        # Load HNSW index
        self.hnsw_index = hnswlib.Index(space='cosine', dim=self.embedding_dim)
        self.hnsw_index.load_index(f"{output_dir}/hybrid_hnsw_index.bin")
        self.hnsw_index.set_num_threads(8)  # Enable multi-threading for search

        # Load BM25 and embeddings
        with open(f"{output_dir}/hybrid_indexes.pkl", 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.chunk_embeddings = data['embeddings']

        print("[SUCCESS] Indexes loaded successfully")


if __name__ == "__main__":
    # Build and save indexes
    retriever = HybridRetriever(
        chunks_file="dataset/wikipedia_ireland/chunks.json",
        graphrag_index_file="dataset/wikipedia_ireland/graphrag_index.json"
    )

    retriever.build_semantic_index()
    retriever.build_keyword_index()
    retriever.save_indexes()

    # Test hybrid search
    query = "What is the capital of Ireland?"
    results = retriever.hybrid_search(query, top_k=5)

    print("\nHybrid Search Results:")
    for result in results:
        print(f"\nRank {result.rank}: {result.source_title}")
        print(f"Score: {result.combined_score:.3f} (semantic: {result.semantic_score:.3f}, keyword: {result.keyword_score:.3f})")
        print(f"Text: {result.text[:200]}...")
