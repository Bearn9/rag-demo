"""Environment loading, model names, demo limits, and filesystem paths shared across ragbot."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "knowledge_base"
FINANCIALS_CSV_PATH = DATA_DIR / "economy_quarterly_financials.csv"
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"
INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.json"
HARNESS_DOC_PATH = ROOT_DIR / "docs" / "harness.md"
EVAL_DATASET_PATH = ROOT_DIR / "evals" / "dataset.jsonl"
EVAL_RESULTS_PATH = ROOT_DIR / "evals" / "results" / "latest.json"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# The agent model drives the whole tool-use loop: it picks tools, reads their
# results, and writes the final cited answer.
AGENT_MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5")
AGENT_EFFORT = os.environ.get("AGENT_EFFORT", "low")
MAX_AGENT_STEPS = int(os.environ.get("MAX_AGENT_STEPS", "5"))
# Prior user/assistant turns (text only) passed back to the agent for follow-ups.
MEMORY_TURNS = 6

# Offline eval only: grades answers for faithfulness/correctness.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-5")

# USD per million tokens (input, output), used for the per-step cost display
# and the demo's daily budget.
PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Approximate request cost. Cache reads bill at 0.1x input, 5-minute cache writes at 1.25x."""
    input_price, output_price = PRICING.get(model, PRICING["claude-sonnet-5"])
    billed_input = input_tokens + 0.1 * cache_read_tokens + 1.25 * cache_write_tokens
    return (billed_input * input_price + output_tokens * output_price) / 1_000_000


# Public-demo guardrails. The Anthropic console spend limit is the hard backstop;
# these just keep a single visitor (or a busy day) from burning through it.
MAX_QUESTIONS_PER_SESSION = int(os.environ.get("MAX_QUESTIONS_PER_SESSION", "15"))
DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "2.0"))
MAX_INPUT_CHARS = 500

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_MAX_CHARS = 500
CHUNK_OVERLAP_CHARS = 50
RETRIEVAL_TOP_K = 4
