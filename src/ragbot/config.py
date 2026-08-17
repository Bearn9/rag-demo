"""Environment loading, model names, and filesystem paths shared across ragbot."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "knowledge_base"
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"
INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.json"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "claude-haiku-4-5-20251001")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "claude-sonnet-5")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_MAX_CHARS = 500
CHUNK_OVERLAP_CHARS = 50
RETRIEVAL_TOP_K = 4
