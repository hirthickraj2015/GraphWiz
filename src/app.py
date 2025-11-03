# app.py
import json
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --------------------
# Config
# --------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TOP_K = 5
INDEX_FILE = "data/faiss_index_ivf.bin"
META_FILE = "data/nodes_meta.json"
EMB_ARRAY_FILE = "data/embeddings.npy"
CONTEXT_FILE = "data/nodes_context.json"
HF_MODEL = "google/flan-t5-small"  # optional for answer generation

# --------------------
# Load everything into memory
# --------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource
def load_faiss_and_meta():
    index = faiss.read_index(INDEX_FILE)
    embeddings = np.load(EMB_ARRAY_FILE)
    metas = json.load(open(META_FILE, "r", encoding="utf-8"))
    contexts = json.load(open(CONTEXT_FILE, "r", encoding="utf-8"))
    return index, metas, embeddings, contexts

@st.cache_resource
def load_gen_model():
    try:
        return pipeline("text2text-generation", model=HF_MODEL)
    except Exception:
        return None

# --------------------
# Semantic search
# --------------------
def semantic_search(query, embed_model, idx, metas):
    q_emb = embed_model.encode(query, convert_to_numpy=True)
    D, I = idx.search(np.array([q_emb]), TOP_K)
    results = []
    for k, i in enumerate(I[0]):
        if i < 0 or i >= len(metas):
            continue
        results.append((metas[i], float(D[0][k])))
    return results

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="⚡ Lightning GraphRAG", layout="wide")
st.title("🇮🇪 Ireland KG — Lightning GraphRAG")

st.sidebar.markdown("### ⚙️ Options")
st.sidebar.markdown("Index & embeddings must be prebuilt via `build_index.py`.")

# Load all in-memory
embed_model = load_embedding_model()
idx, metas, embeddings, contexts = load_faiss_and_meta()
gen_model = load_gen_model()

question = st.text_input("🔍 Ask a question about Ireland:")

if st.button("Search") and question:
    with st.spinner("Searching..."):
        # Semantic search
        results = semantic_search(question, embed_model, idx, metas)
        uris = [meta["uri"] for meta, _ in results]

        # Fetch context
        context_list = []
        for uri in uris:
            ctx_lines = contexts.get(uri, [])
            context_list.append(f"{uri}: {', '.join(ctx_lines)}")

        # Optional answer generation
        if gen_model:
            prompt = f"Answer the question based on the context below:\n\n{chr(10).join(context_list)}\n\nQuestion: {question}\nAnswer:"
            answer = gen_model(prompt, max_length=200, do_sample=False)[0]['generated_text']
        else:
            answer = "No generation model configured. Showing raw context:\n\n" + "\n".join(context_list)

    st.markdown("### ✅ Answer")
    st.write(answer)

    st.markdown("---")
    st.markdown("### 🔗 References")
    for meta, score in results:
        st.markdown(f"- [{meta['uri']}]({meta['uri']}) — Score: `{score:.4f}`")

    st.markdown("---")
    st.markdown("### 📄 Context Used")
    st.code("\n".join(context_list))
