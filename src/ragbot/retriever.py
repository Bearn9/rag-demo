"""Query-time retrieval: loads the FAISS index built by ingest.py and searches it.

This module never re-embeds the corpus — it only reads the artifacts that
`python -m ragbot.ingest` already produced.
"""

import json
from functools import lru_cache
from typing import TypedDict

import faiss
from sentence_transformers import SentenceTransformer

from ragbot import config


class Chunk(TypedDict):
    text: str
    source: str
    section: str
    score: float


@lru_cache(maxsize=1)
def _load_index() -> tuple[faiss.Index, list[dict]]:
    if not config.INDEX_PATH.exists() or not config.CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Vector store not found. Run `python -m ragbot.ingest` first to build it "
            f"(expected {config.INDEX_PATH} and {config.CHUNKS_PATH})."
        )
    index = faiss.read_index(str(config.INDEX_PATH))
    chunks = json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8"))
    return index, chunks


@lru_cache(maxsize=1)
def _load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


def retrieve(query: str, k: int = config.RETRIEVAL_TOP_K) -> list[Chunk]:
    """Embed `query` and return the top-k most similar chunks, highest score first."""
    index, chunks = _load_index()
    model = _load_embedding_model()

    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, k)

    results: list[Chunk] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append(
            Chunk(
                text=chunk["text"],
                source=chunk["source"],
                section=chunk["section"],
                score=float(score),
            )
        )
    return results
