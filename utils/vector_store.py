"""Splits text into chunks and performs semantic search using
Sentence Transformers (embeddings) + FAISS (fast similarity search)."""

import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Loaded once and reused everywhere (downloads the model the first time you run this)
_EMBEDDER_NAME = "all-MiniLM-L6-v2"
_embedder = None


def get_embedder():
    """Load the sentence embedding model once and reuse it."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_EMBEDDER_NAME)
    return _embedder


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):
    """
    Split text into overlapping chunks (by characters, on sentence boundaries
    where possible), so each chunk is small enough for accurate search but
    still keeps full sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current.strip())
            # start new chunk, keep a bit of overlap from the end of the last one
            current = current[-overlap:] + " " + sentence if current else sentence

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 20]  # drop tiny leftover fragments


def build_index(chunks: list[str]):
    """
    Turn a list of text chunks into a FAISS index that supports fast
    semantic search. Returns the index (search happens against it).
    """
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    # Normalize vectors so we can use inner product as cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def search_index(query: str, index, chunks: list[str], top_k: int = 4):
    """
    Find the chunks most relevant to the query.
    Returns a list of the top_k most relevant chunk texts.
    """
    embedder = get_embedder()
    query_vector = np.array(embedder.encode([query])).astype("float32")
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])
    return results