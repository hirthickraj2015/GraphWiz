"""
GraphRAG Builder with Community Detection and Hierarchical Summarization
Implements Microsoft GraphRAG approach for knowledge graphs
"""

import json
import networkx as nx
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class GraphRAGBuilder:
    """Build GraphRAG with community detection and hierarchical summaries"""

    def __init__(self, chunks_file: str, output_dir: str = "dataset/wikipedia_ireland"):
        self.chunks_file = chunks_file
        self.output_dir = output_dir
        self.graph = nx.Graph()
        self.entity_graph = nx.DiGraph()
        self.chunks = []
        self.entity_to_chunks = defaultdict(list)
        self.chunk_to_entities = defaultdict(list)

    def load_chunks(self):
        """Load processed chunks"""
        print(f"[INFO] Loading chunks from {self.chunks_file}")
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        print(f"[SUCCESS] Loaded {len(self.chunks)} chunks")

    def build_entity_graph(self):
        """Build graph from entities across chunks"""
        print("[INFO] Building entity graph from chunks...")

        # Extract all entities and their co-occurrences
        for chunk_idx, chunk in enumerate(tqdm(self.chunks, desc="Processing chunks")):
            chunk_id = chunk['chunk_id']
            entities = chunk.get('entities', [])

            # Track which chunks contain which entities
            for entity in entities:
                entity_key = f"{entity['text']}|{entity['label']}"
                self.entity_to_chunks[entity_key].append(chunk_id)
                self.chunk_to_entities[chunk_id].append(entity_key)

                # Add entity as node if not exists
                if not self.entity_graph.has_node(entity_key):
                    self.entity_graph.add_node(
                        entity_key,
                        text=entity['text'],
                        label=entity['label'],
                        chunk_count=0
                    )

                # Update chunk count
                self.entity_graph.nodes[entity_key]['chunk_count'] += 1

            # Create edges between co-occurring entities in same chunk
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    key1 = f"{entity1['text']}|{entity1['label']}"
                    key2 = f"{entity2['text']}|{entity2['label']}"

                    if self.entity_graph.has_edge(key1, key2):
                        self.entity_graph[key1][key2]['weight'] += 1
                    else:
                        self.entity_graph.add_edge(key1, key2, weight=1)

        print(f"[SUCCESS] Entity graph: {self.entity_graph.number_of_nodes()} nodes, "
              f"{self.entity_graph.number_of_edges()} edges")

    def build_semantic_chunk_graph(self, similarity_threshold: float = 0.3):
        """Build graph of semantically similar chunks"""
        print("[INFO] Building semantic similarity graph...")

        # Extract chunk texts
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        chunk_ids = [chunk['chunk_id'] for chunk in self.chunks]

        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(chunk_texts)

        # Compute pairwise cosine similarity (in batches to save memory)
        batch_size = 500
        for i in tqdm(range(0, len(chunk_texts), batch_size), desc="Computing similarity"):
            end_i = min(i + batch_size, len(chunk_texts))
            batch_similarities = cosine_similarity(tfidf_matrix[i:end_i], tfidf_matrix)

            for local_idx, chunk_idx in enumerate(range(i, end_i)):
                chunk_id = chunk_ids[chunk_idx]

                # Add chunk as node
                if not self.graph.has_node(chunk_id):
                    self.graph.add_node(
                        chunk_id,
                        text=chunk_texts[chunk_idx],
                        source_title=self.chunks[chunk_idx]['source_title'],
                        source_url=self.chunks[chunk_idx]['source_url'],
                        section=self.chunks[chunk_idx]['section'],
                        word_count=self.chunks[chunk_idx]['word_count']
                    )

                # Add edges to similar chunks
                for other_idx, similarity in enumerate(batch_similarities[local_idx]):
                    if chunk_idx != other_idx and similarity > similarity_threshold:
                        other_chunk_id = chunk_ids[other_idx]
                        if not self.graph.has_edge(chunk_id, other_chunk_id):
                            self.graph.add_edge(chunk_id, other_chunk_id, weight=float(similarity))

        print(f"[SUCCESS] Chunk graph: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")

    def detect_communities(self, resolution: float = 1.0) -> Dict[str, int]:
        """Detect communities using Louvain algorithm"""
        print("[INFO] Detecting communities with Louvain algorithm...")

        from networkx.algorithms import community as nx_comm

        # Use Louvain for community detection
        communities = nx_comm.louvain_communities(self.graph, resolution=resolution, seed=42)

        # Create node to community mapping
        node_to_community = {}
        for comm_id, community_nodes in enumerate(communities):
            for node in community_nodes:
                node_to_community[node] = comm_id

        print(f"[SUCCESS] Detected {len(communities)} communities")

        # Add community attribute to nodes
        for node, comm_id in node_to_community.items():
            self.graph.nodes[node]['community'] = comm_id

        return node_to_community

    def generate_community_summaries(self, node_to_community: Dict[str, int], max_chunks_per_summary: int = 20) -> Dict[int, Dict]:
        """Generate hierarchical summaries for each community"""
        print("[INFO] Generating community summaries...")

        communities = defaultdict(list)
        for node, comm_id in node_to_community.items():
            communities[comm_id].append(node)

        community_summaries = {}

        for comm_id, chunk_ids in tqdm(communities.items(), desc="Summarizing communities"):
            # Gather all text from chunks in this community (limit to avoid huge summaries)
            sample_chunk_ids = chunk_ids[:max_chunks_per_summary]
            chunk_texts = []
            sources = set()

            for chunk_id in sample_chunk_ids:
                chunk_data = self.graph.nodes.get(chunk_id, {})
                chunk_texts.append(chunk_data.get('text', ''))
                sources.add(chunk_data.get('source_title', 'Unknown'))

            # Extract most common entities in this community
            community_entities = []
            for chunk_id in chunk_ids:
                community_entities.extend(self.chunk_to_entities.get(chunk_id, []))

            entity_counter = Counter(community_entities)
            top_entities = entity_counter.most_common(20)

            # Generate summary metadata (would use LLM for actual summary in production)
            combined_text = " ".join(chunk_texts)
            summary = {
                "community_id": comm_id,
                "num_chunks": len(chunk_ids),
                "num_sources": len(sources),
                "sources": list(sources)[:10],
                "top_entities": [{"entity": ent[0].split('|')[0], "count": ent[1]} for ent in top_entities],
                "combined_text_sample": combined_text[:2000],  # First 2000 chars as preview
                "total_text_length": len(combined_text),
                "chunk_ids": chunk_ids[:100]  # Limit stored chunk IDs
            }

            community_summaries[comm_id] = summary

        print(f"[SUCCESS] Generated {len(community_summaries)} community summaries")
        return community_summaries

    def build_hierarchical_index(self) -> Dict:
        """Build complete hierarchical index for GraphRAG"""
        print("=" * 80)
        print("BUILDING GRAPHRAG HIERARCHICAL INDEX")
        print("=" * 80)

        # Step 1: Load chunks
        self.load_chunks()

        # Step 2: Build entity graph
        self.build_entity_graph()

        # Step 3: Build semantic chunk graph
        self.build_semantic_chunk_graph(similarity_threshold=0.25)

        # Step 4: Detect communities
        node_to_community = self.detect_communities(resolution=1.0)

        # Step 5: Generate community summaries
        community_summaries = self.generate_community_summaries(node_to_community)

        # Step 6: Build complete index
        graphrag_index = {
            "metadata": {
                "total_chunks": len(self.chunks),
                "total_entities": self.entity_graph.number_of_nodes(),
                "total_communities": len(set(node_to_community.values())),
                "chunk_graph_edges": self.graph.number_of_edges(),
                "entity_graph_edges": self.entity_graph.number_of_edges()
            },
            "communities": community_summaries,
            "entity_to_chunks": dict(self.entity_to_chunks),
            "chunk_to_entities": dict(self.chunk_to_entities),
            "node_to_community": node_to_community
        }

        return graphrag_index

    def save_graphrag_index(self, graphrag_index: Dict):
        """Save GraphRAG index and graphs"""
        print("[INFO] Saving GraphRAG index...")

        # Save main index as JSON
        index_path = f"{self.output_dir}/graphrag_index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(graphrag_index, f, ensure_ascii=False, indent=2)
        print(f"[SUCCESS] Saved GraphRAG index to {index_path}")

        # Save graphs as pickle (more efficient for networkx graphs)
        graphs_path = f"{self.output_dir}/graphrag_graphs.pkl"
        with open(graphs_path, 'wb') as f:
            pickle.dump({
                'chunk_graph': self.graph,
                'entity_graph': self.entity_graph
            }, f)
        print(f"[SUCCESS] Saved graphs to {graphs_path}")

        # Save human-readable statistics
        stats = {
            "total_chunks": graphrag_index["metadata"]["total_chunks"],
            "total_entities": graphrag_index["metadata"]["total_entities"],
            "total_communities": graphrag_index["metadata"]["total_communities"],
            "communities": []
        }

        for comm_id, comm_data in graphrag_index["communities"].items():
            stats["communities"].append({
                "id": comm_id,
                "num_chunks": comm_data["num_chunks"],
                "num_sources": comm_data["num_sources"],
                "top_sources": comm_data["sources"][:5],
                "top_entities": [e["entity"] for e in comm_data["top_entities"][:10]]
            })

        stats_path = f"{self.output_dir}/graphrag_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[SUCCESS] Saved statistics to {stats_path}")

        print("=" * 80)
        print("GRAPHRAG INDEX BUILDING COMPLETE!")
        print("=" * 80)


if __name__ == "__main__":
    builder = GraphRAGBuilder(
        chunks_file="dataset/wikipedia_ireland/chunks.json"
    )
    graphrag_index = builder.build_hierarchical_index()
    builder.save_graphrag_index(graphrag_index)
