"""Standalone ingestion CLI: chunk the knowledge base, embed it, and persist a FAISS index.

Run with:  python -m ragbot.ingest

This is intentionally decoupled from the live app — the Streamlit app and the
LangGraph nodes only ever *read* the artifacts this script produces (via
retriever.py). Re-run this script whenever data/knowledge_base/ changes.
"""

import csv
import json
import re

import faiss
from sentence_transformers import SentenceTransformer

from ragbot import config


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown text on '##' headers, returning (section_title, section_body) pairs."""
    parts = re.split(r"(?m)^##\s+(.+)$", text)
    # parts[0] is any preamble before the first "##" header (the "# Title" line).
    sections = []
    preamble = parts[0].strip()
    if preamble:
        title_match = re.match(r"(?m)^#\s+(.+)$", preamble)
        title = title_match.group(1).strip() if title_match else "Overview"
        sections.append((title, preamble))
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i + 1].strip()
        sections.append((title, body))
    return sections


def _split_long_section(title: str, body: str) -> list[str]:
    """Further split a section body into ~CHUNK_MAX_CHARS pieces with overlap, on paragraph
    boundaries where possible."""
    if len(body) <= config.CHUNK_MAX_CHARS:
        return [body]

    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > config.CHUNK_MAX_CHARS and current:
            chunks.append(current)
            overlap = current[-config.CHUNK_OVERLAP_CHARS :]
            current = f"{overlap}\n\n{para}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def chunk_document(source: str, text: str) -> list[dict]:
    """Chunk a single markdown document into {text, source, section, chunk_id} dicts."""
    chunks = []
    for section_title, section_body in _split_into_sections(text):
        for i, piece in enumerate(_split_long_section(section_title, section_body)):
            chunks.append(
                {
                    "text": piece,
                    "source": source,
                    "section": section_title,
                    "chunk_id": f"{source}::{section_title}::{i}",
                }
            )
    return chunks


def chunk_csv_document(source: str, path) -> list[dict]:
    """Turn each row of a CSV into a standalone chunk, rendered as a plain-English
    sentence so the embedding model has real prose to match against (rather than
    raw column/value pairs)."""
    chunks = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            label = next(iter(row.values()), str(i))
            fields = ", ".join(f"{key}: {value}" for key, value in row.items())
            chunks.append(
                {
                    "text": f"{source.replace('_', ' ').removesuffix('.csv')} — {label}. {fields}.",
                    "source": source,
                    "section": label,
                    "chunk_id": f"{source}::{label}",
                }
            )
    return chunks


def build_index() -> None:
    config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    doc_paths = sorted(config.DATA_DIR.glob("*.md")) + sorted(config.DATA_DIR.glob("*.csv"))
    if not doc_paths:
        raise SystemExit(f"No markdown or CSV files found in {config.DATA_DIR}")

    all_chunks: list[dict] = []
    for path in doc_paths:
        if path.suffix == ".csv":
            all_chunks.extend(chunk_csv_document(path.name, path))
        else:
            text = path.read_text(encoding="utf-8")
            all_chunks.extend(chunk_document(path.name, text))

    print(f"Loaded {len(doc_paths)} documents, split into {len(all_chunks)} chunks.")

    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(config.INDEX_PATH))

    config.CHUNKS_PATH.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"Indexed {len(all_chunks)} chunks from {len(doc_paths)} documents "
        f"into {config.INDEX_PATH.relative_to(config.ROOT_DIR)}"
    )


if __name__ == "__main__":
    build_index()
