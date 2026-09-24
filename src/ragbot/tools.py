"""The agent's tools: JSON schemas for Claude plus the pure-Python implementations.

    search_knowledge_base  -> semantic search over the FAISS index (retriever.py)
    query_financials       -> exact computation over the quarterly financials CSV
    calculator             -> safe arithmetic (AST-based, never eval())
    explain_harness        -> the assistant's own design docs + live config values

Every implementation returns a JSON-serialisable dict. `run_tool` dispatches by name
and turns any exception into an {"error": ...} result that the agent can read.
"""

import ast
import csv
import json
import math
import operator
import re
from functools import lru_cache

from ragbot import config, retriever

# ---------------------------------------------------------------------------
# query_financials
# ---------------------------------------------------------------------------

FINANCIAL_METRICS = [
    "Revenue (USD)",
    "Operating Expenses (USD)",
    "Net Profit (USD)",
    "Profit Margin (%)",
    "ARR (USD)",
    "Headcount",
    "Customer Count",
    "Churn Rate (%)",
    "Cash Reserves (USD)",
]
FINANCIAL_OPERATIONS = ["values", "sum", "mean", "min", "max", "change", "pct_change"]
FINANCIALS_SOURCE = config.FINANCIALS_CSV_PATH.name


@lru_cache(maxsize=1)
def _load_financials() -> list[dict]:
    with config.FINANCIALS_CSV_PATH.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        {"Quarter": row["Quarter"], **{m: float(row[m]) for m in FINANCIAL_METRICS}}
        for row in rows
    ]


def available_quarters() -> list[str]:
    return [row["Quarter"] for row in _load_financials()]


def _normalise_quarter(label: str) -> str:
    """Accept 'Q1 2025', 'q1-2025', '2025 Q1', etc. and return the canonical 'Q1 2025'."""
    match = re.search(r"[qQ]([1-4])\D*(\d{4})|(\d{4})\D*[qQ]([1-4])", label)
    if not match:
        raise ValueError(f"Unrecognised quarter {label!r}; expected e.g. 'Q1 2025'.")
    quarter, year = (match.group(1), match.group(2)) if match.group(1) else (match.group(4), match.group(3))
    return f"Q{quarter} {year}"


def _clean_number(value: float) -> float | int:
    return int(value) if float(value).is_integer() else round(value, 4)


def query_financials(metrics: list[str], quarters: list[str], operation: str) -> dict:
    """Compute `operation` for each metric over the selected quarters.

    `quarters` may be ["all"]. `change` and `pct_change` compare the last selected
    quarter to the first (in chronological order).
    """
    unknown = [m for m in metrics if m not in FINANCIAL_METRICS]
    if unknown or not metrics:
        raise ValueError(f"Unknown metric(s) {unknown}; choose from {FINANCIAL_METRICS}.")
    if operation not in FINANCIAL_OPERATIONS:
        raise ValueError(f"Unknown operation {operation!r}; choose from {FINANCIAL_OPERATIONS}.")

    rows = _load_financials()
    if not quarters or any(q.strip().lower() == "all" for q in quarters):
        selected = rows
    else:
        wanted = {_normalise_quarter(q) for q in quarters}
        selected = [row for row in rows if row["Quarter"] in wanted]
        missing = wanted - {row["Quarter"] for row in selected}
        if missing:
            raise ValueError(
                f"No data for {sorted(missing)}; available quarters are {available_quarters()}."
            )

    results = {}
    for metric in metrics:
        series = [row[metric] for row in selected]
        if operation == "values":
            value = {row["Quarter"]: _clean_number(row[metric]) for row in selected}
        elif operation == "sum":
            value = sum(series)
        elif operation == "mean":
            value = sum(series) / len(series)
        elif operation == "min":
            idx = series.index(min(series))
            value = {"quarter": selected[idx]["Quarter"], "value": _clean_number(series[idx])}
        elif operation == "max":
            idx = series.index(max(series))
            value = {"quarter": selected[idx]["Quarter"], "value": _clean_number(series[idx])}
        else:
            if len(series) < 2:
                raise ValueError(f"{operation} needs at least two quarters.")
            first, last = series[0], series[-1]
            value = last - first if operation == "change" else (last - first) / first * 100
        results[metric] = _clean_number(value) if isinstance(value, float) else value

    return {
        "operation": operation,
        "quarters": [row["Quarter"] for row in selected],
        "results": results,
        "source": FINANCIALS_SOURCE,
    }


# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {"round": round, "min": min, "max": max, "abs": abs, "sqrt": math.sqrt}
_MAX_EXPRESSION_CHARS = 200
_MAX_EXPONENT = 100


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"Exponent too large (max {_MAX_EXPONENT}).")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*(_evaluate(arg) for arg in node.args))
    raise ValueError(f"Unsupported expression element: {type(node).__name__}.")


def calculator(expression: str) -> dict:
    """Safely evaluate an arithmetic expression like '0.15 * 250' or 'round(560000/1865000*100, 1)'."""
    # Drop thousands separators and "$" signs; "15%" becomes "(15/100)".
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expression).replace("$", "")
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", cleaned).strip()
    if len(cleaned) > _MAX_EXPRESSION_CHARS:
        raise ValueError(f"Expression too long (max {_MAX_EXPRESSION_CHARS} characters).")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Could not parse expression {expression!r}.") from exc
    result = _evaluate(tree)
    return {"expression": expression, "result": _clean_number(float(result))}


# ---------------------------------------------------------------------------
# explain_harness
# ---------------------------------------------------------------------------

HARNESS_TOPICS = [
    "overview",
    "architecture",
    "tools",
    "models",
    "retrieval",
    "evaluation",
    "guardrails",
    "tech_stack",
]
HARNESS_SOURCE = config.HARNESS_DOC_PATH.name


@lru_cache(maxsize=1)
def _harness_sections() -> dict[str, str]:
    text = config.HARNESS_DOC_PATH.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^##\s+(\S+)\s*$", text)
    return {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts), 2)}


def live_config() -> dict:
    return {
        "agent_model": config.AGENT_MODEL,
        "agent_effort": config.AGENT_EFFORT,
        "max_agent_steps": config.MAX_AGENT_STEPS,
        "memory_turns": config.MEMORY_TURNS,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "retrieval_top_k": config.RETRIEVAL_TOP_K,
        "judge_model": config.JUDGE_MODEL,
        "max_questions_per_session": config.MAX_QUESTIONS_PER_SESSION,
        "daily_budget_usd": config.DAILY_BUDGET_USD,
        "tools": [tool["name"] for tool in TOOL_DEFINITIONS],
    }


def _eval_headline() -> dict | None:
    if not config.EVAL_RESULTS_PATH.exists():
        return None
    try:
        results = json.loads(config.EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return {"run_at": results.get("run_at"), "headline": results.get("headline")}


def explain_harness(topic: str) -> dict:
    sections = _harness_sections()
    if topic not in sections:
        raise ValueError(f"Unknown topic {topic!r}; choose from {HARNESS_TOPICS}.")
    result = {"topic": topic, "content": sections[topic], "live_config": live_config(), "source": HARNESS_SOURCE}
    if topic == "evaluation":
        headline = _eval_headline()
        if headline:
            result["latest_eval_results"] = headline
    return result


# ---------------------------------------------------------------------------
# search_knowledge_base
# ---------------------------------------------------------------------------


def search_knowledge_base(query: str) -> dict:
    chunks = retriever.retrieve(query, k=config.RETRIEVAL_TOP_K)
    return {"query": query, "chunks": chunks}


# ---------------------------------------------------------------------------
# Tool schemas + dispatch
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Semantic search over Bearn Analytics' internal documents: HR policies (PTO, "
            "benefits, parental leave), IT setup (laptops, VPN, accounts), security policy "
            "(passwords, MFA), the remote-work handbook, product FAQs (pricing, billing, "
            "features, API limits), the expense/travel policy, and the annual report (funding, "
            "company history). Returns the most relevant text chunks with their source file. "
            "Use this for any factual question about Bearn that isn't a quarterly-metric calculation."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused search query, rephrased for retrieval if helpful.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "query_financials",
        "description": (
            "Exact computation over Bearn's quarterly financials (Q1 2023 to Q2 2025). Use it for "
            "any question about revenue, expenses, net profit, profit margin, ARR, headcount, "
            "customer count, churn or cash reserves, including comparisons and growth. Operations: "
            "values (raw numbers per quarter), sum, mean, min, max (with the quarter), change "
            "(last minus first selected quarter) and pct_change (percentage growth from first to "
            "last). Pass quarters like 'Q1 2025', or ['all']."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "metrics": {
                    "type": "array",
                    "items": {"type": "string", "enum": FINANCIAL_METRICS},
                    "description": "One or more metrics (CSV column names).",
                },
                "quarters": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Quarters such as 'Q1 2025', or ['all'] for every quarter.",
                },
                "operation": {"type": "string", "enum": FINANCIAL_OPERATIONS},
            },
            "required": ["metrics", "quarters", "operation"],
            "additionalProperties": False,
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate an arithmetic expression exactly, e.g. '0.15 * 250' or "
            "'round((560000 - 490000) / 490000 * 100, 2)'. Supports + - * / // ** and "
            "round, min, max, abs, sqrt; a trailing % means percent (15% = 0.15). Use it for any arithmetic instead of computing in your head."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
    {
        "name": "explain_harness",
        "description": (
            "Returns documentation about how this assistant itself works (the Bearn Agent Demo "
            "harness), plus live configuration values such as model names and limits. Use it when "
            "the user asks about the assistant's design: its architecture, tools, models, "
            "retrieval, evaluation, guardrails or tech stack."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string", "enum": HARNESS_TOPICS}},
            "required": ["topic"],
            "additionalProperties": False,
        },
    },
]

_IMPLEMENTATIONS = {
    "search_knowledge_base": search_knowledge_base,
    "query_financials": query_financials,
    "calculator": calculator,
    "explain_harness": explain_harness,
}


def run_tool(name: str, tool_input: dict) -> tuple[dict, bool]:
    """Run a tool by name. Returns (result, is_error)."""
    impl = _IMPLEMENTATIONS.get(name)
    if impl is None:
        return {"error": f"Unknown tool {name!r}."}, True
    try:
        return impl(**tool_input), False
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        return {"error": str(exc)}, True


def sources_from_result(name: str, result: dict) -> list[str]:
    """Source files a tool result draws on, for citation checks and the UI."""
    if name == "search_knowledge_base":
        return [chunk["source"] for chunk in result.get("chunks", [])]
    if "source" in result:
        return [result["source"]]
    return []
