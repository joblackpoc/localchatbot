# chat/rag.py
import os
import json
import uuid
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "rag_data"
DATA_DIR.mkdir(exist_ok=True)

# filenames
DOCS_FILE = DATA_DIR / "docs.json"      # {id: {"text":..., "meta":...}}
INDEX_FILE = DATA_DIR / "faiss.index"   # faiss index file
EMB_MODEL_NAME = "all-MiniLM-L6-v2"     # small and CPU friendly

# load embedding model (global)
_embed_model = SentenceTransformer(EMB_MODEL_NAME)

# helper to load or create docs metadata
def _load_docs() -> Dict[str, Dict]:
    if DOCS_FILE.exists():
        with open(DOCS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_docs(docs: Dict[str, Dict]):
    with open(DOCS_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

# create or load faiss index
def _load_index(dim: int):
    if INDEX_FILE.exists():
        index = faiss.read_index(str(INDEX_FILE))
        return index
    else:
        index = faiss.IndexFlatIP(dim)  # inner product (we'll normalize)
        return index

def _save_index(index):
    faiss.write_index(index, str(INDEX_FILE))

def _normalize(vecs: np.ndarray):
    # normalize rows for cosine similarity with inner product index
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms

# a simple text splitter: split paragraphs and further chunks if long
def splitter(text: str, chunk_size_chars: int = 800, overlap_chars: int = 100) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    for p in paragraphs:
        if len(p) <= chunk_size_chars:
            chunks.append(p)
        else:
            start = 0
            while start < len(p):
                end = start + chunk_size_chars
                chunks.append(p[start:end])
                start = max(0, end - overlap_chars)
    return chunks

# ingest plain text (string) and return number of chunks added
def ingest_text(text: str, meta: Dict = None) -> int:
    if meta is None:
        meta = {}
    docs = _load_docs()

    # split and embed
    chunks = splitter(text)
    if not chunks:
        return 0

    embeddings = _embed_model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    embeddings = _normalize(embeddings.astype("float32"))

    # load index (dim from model)
    dim = embeddings.shape[1]
    index = _load_index(dim)

    # if index is empty but not initialized with correct dim, ensure correct dim
    if isinstance(index, faiss.IndexFlatIP):
        # OK
        pass

    # add vectors and metadata
    start_id = len(docs)
    ids = []
    for i, chunk in enumerate(chunks):
        doc_id = str(uuid.uuid4())
        docs[doc_id] = {"text": chunk, "meta": meta}
        ids.append(doc_id)

    # add to index
    index.add(embeddings)
    _save_index(index)
    _save_docs(docs)
    return len(chunks)

# query: returns top k documents (texts + meta)
def query(question: str, k: int = 5):
    docs = _load_docs()
    if not docs:
        return []

    q_emb = _embed_model.encode([question], convert_to_numpy=True)
    q_emb = _normalize(q_emb.astype("float32"))

    index = _load_index(q_emb.shape[1])
    # if index has zero vectors, return empty
    if index.ntotal == 0:
        return []

    # search
    D, I = index.search(q_emb, k)  # distances and indices
    # faiss returns indices as 0..n-1 in insertion order
    # We stored docs in insertion order so map by order: but we used UUID keys, so we need stable ordering
    # Reconstruct insertion-order list of keys from saved docs
    doc_keys = list(docs.keys())
    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0 or idx >= len(doc_keys):
            continue
        key = doc_keys[idx]
        entry = docs.get(key)
        if entry:
            results.append({"id": key, "text": entry["text"], "meta": entry.get("meta", {}), "score": float(score)})
    return results

# helper to clear index (for admin)
def clear_index():
    if INDEX_FILE.exists():
        INDEX_FILE.unlink()
    if DOCS_FILE.exists():
        DOCS_FILE.unlink()
    # recreate directory files
    return True
