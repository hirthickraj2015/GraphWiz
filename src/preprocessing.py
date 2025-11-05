# build_index.py
import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# --------------------
# CONFIG
# --------------------
NEO4J_URI = "neo4j+s://d3c3a325.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "uZHvPLzNNUlqVewtkhKXSpME3TkSJSZy2LOx70d5puc"
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TOP_K = 5
INDEX_FILE = "data/faiss_index_ivf.bin"
META_FILE = "data/nodes_meta.json"
EMB_ARRAY_FILE = "data/embeddings.npy"
CONTEXT_FILE = "data/nodes_context.json"
NUM_THREADS = 8   # adjust based on CPU cores

# --------------------
# Neo4j driver
# --------------------
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# --------------------
# Utilities
# --------------------
def fetch_all_nodes():
    uris = []
    with driver.session(database=NEO4J_DATABASE) as session:
        for record in session.run("MATCH (n:Entity) RETURN n.uri AS uri"):
            uri = record.get("uri")
            if uri:
                uris.append(uri)
    return uris

def fetch_node_context(uri, neighbor_limit=20):
    try:
        query = """
        MATCH (n {uri:$uri})
        OPTIONAL MATCH (n)-[r]->(m)
        OPTIONAL MATCH (m2)-[r2]-(n)
        RETURN collect(DISTINCT {out_rel:type(r), out_obj:m.uri})[0..$limit] AS outs,
               collect(DISTINCT {in_rel:type(r2), in_subj:m2.uri})[0..$limit] AS ins
        """
        with driver.session(database=NEO4J_DATABASE) as session:
            rec = session.run(query, uri=uri, limit=neighbor_limit).single()
            outs = rec["outs"] if rec and rec.get("outs") else []
            ins = rec["ins"] if rec and rec.get("ins") else []
            lines = []
            for o in outs:
                rel = o.get("out_rel") or "RELATED_TO"
                obj = o.get("out_obj") or ""
                lines.append(f"{rel} -> {obj}")
            for i in ins:
                rel = i.get("in_rel") or "RELATED_TO"
                subj = i.get("in_subj") or ""
                lines.append(f"{subj} <- {rel}")
            return lines
    except Exception as e:
        print(f"[WARN] Failed to fetch context for {uri}: {e}")
        return []

def build_embeddings_parallel(uris):
    model = SentenceTransformer(EMBED_MODEL_NAME)
    metas, texts, contexts = [], [], {}

    def process(uri):
        text = uri  # can extend to more descriptive text
        ctx = fetch_node_context(uri)
        contexts[uri] = ctx
        return {"uri": uri, "text": text}, text

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(process, uri): uri for uri in uris}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Embedding nodes"):
            meta, text = f.result()
            metas.append(meta)
            texts.append(text)

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings, metas, contexts

def build_faiss_index(embeddings, nlist=100):
    # IVF (Inverted File) index for millions of vectors
    quantizer = faiss.IndexFlatL2(EMBEDDING_DIM)
    index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, nlist, faiss.METRIC_L2)
    if not index.is_trained:
        index.train(embeddings)
    index.add(embeddings)
    return index

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("Fetching all node URIs from Neo4j...")
    uris = fetch_all_nodes()
    print(f"Total nodes: {len(uris)}")

    print("Building embeddings and contexts...")
    embeddings, metas, contexts = build_embeddings_parallel(uris)

    print("Saving embeddings and metadata...")
    np.save(EMB_ARRAY_FILE, embeddings)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)
    with open(CONTEXT_FILE, "w", encoding="utf-8") as f:
        json.dump(contexts, f, ensure_ascii=False, indent=2)

    print("Building FAISS IVF index...")
    index = build_faiss_index(embeddings, nlist=256)
    faiss.write_index(index, INDEX_FILE)

    print(" Preprocessing complete. FAISS index, embeddings, metadata, and context saved!")
