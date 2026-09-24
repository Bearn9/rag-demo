"""Pure evaluation metrics: no LLM calls, so they're unit-testable without an API key.

scripts/run_eval.py runs the agent over evals/dataset.jsonl and uses these functions to
score each case and aggregate the results.
"""

import json
import re
from collections import Counter

from ragbot import config

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_dataset(path=config.EVAL_DATASET_PATH) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                cases.append(json.loads(line))
    return cases


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def hit_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """1.0 if any expected source appears in the top-k retrieved sources."""
    return 1.0 if set(retrieved[:k]) & set(expected) else 0.0


def reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    """1 / rank of the first relevant source (0 if none). Averaged over cases this is MRR."""
    for rank, source in enumerate(retrieved, start=1):
        if source in expected:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Fraction of the expected sources that appear in the top-k retrieved sources."""
    if not expected:
        return 1.0
    return len(set(retrieved[:k]) & set(expected)) / len(set(expected))


# ---------------------------------------------------------------------------
# Answer checks
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[source:\s*([^\]]+)\]", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_SCALE_RE = re.compile(r"\s?(thousand|million|billion|[kmb])\b", re.IGNORECASE)
_REFUSAL_RE = re.compile(
    r"(isn't|is not|not) (in|included in|covered (in|by)|available in|part of) (the|our|my|bearn)"
    r"|(don't|do not|doesn't|does not) (have|contain|include|cover|mention)"
    r"|(couldn't|could not|can't|cannot|wasn't able to|was not able to|unable to) (find|locate|help|answer|provide|share|write|assist)"
    r"|no (information|details|data|record)"
    r"|outside (of )?(what|(the|that|this|my) scope|my)|i'll (have to )?pass"
    r"|(i can )?only (help|assist|answer)|not (something|a request) i can"
    r"|(isn't|is not|not) (really )?(something|what) i('m| am| can| have)"
    r"|(isn't|aren't|is not|are not|not) (yet )?available"
    r"|(is|are|was|were)(n't| not) mentioned|not mentioned|no [\w\- ]{1,30} (is|are) mentioned"
    r"|(no|isn't any|is no) (mention|record|data|information)",
    re.IGNORECASE,
)


def cited_sources(answer: str) -> list[str]:
    """Filenames cited as [source: file] (a citation may list several, comma-separated)."""
    sources = []
    for match in _CITATION_RE.findall(answer):
        sources.extend(part.strip() for part in match.split(",") if part.strip())
    return sources


def _normalise(text: str) -> str:
    return re.sub(r"[\s\-‐‑–]+", " ", text.lower())


def contains_all(answer: str, substrings: list[str]) -> bool:
    """Case-insensitive; hyphens and spaces are interchangeable ('real-time' ~ 'real time')."""
    normalised = _normalise(answer)
    return all(_normalise(s) in normalised for s in substrings)


_SCALES = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}


def numbers_in(text: str) -> list[float]:
    """Every number in `text`; '4.6 million' / '$4.6M' also yield 4600000."""
    values = []
    for match in _NUMBER_RE.finditer(text):
        try:
            value = float(match.group().replace(",", ""))
        except ValueError:
            continue
        values.append(value)
        suffix = _SCALE_RE.match(text, match.end())
        if suffix:
            values.append(value * _SCALES[suffix.group(1).lower()])
    return values


def numeric_match(answer: str, expected: float, tolerance: float) -> bool:
    """True if any number in the answer is within `tolerance` of `expected`."""
    return any(abs(value - expected) <= tolerance for value in numbers_in(answer))


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(answer))


def tools_match(used: list[str], expected: list[str] | None) -> bool | None:
    """Expected tools must all be used; [] means no tools at all. None means don't check."""
    if expected is None:
        return None
    if not expected:
        return not used
    return set(expected) <= set(used)


def citation_scores(cited: list[str], retrieved: list[str], expected: list[str]) -> tuple[float | None, float | None]:
    """(precision, recall). Precision: share of cited files that tools actually returned.
    Recall: share of the expected source files that were cited."""
    precision = (sum(1 for c in cited if c in retrieved) / len(cited)) if cited else None
    recall = (len(set(cited) & set(expected)) / len(set(expected))) if expected else None
    return precision, recall


def score_case(case: dict, answer: str, tools_used: list[str], sources: list[str]) -> dict:
    """Run every check that applies to this case. `passed` requires all of them."""
    checks: dict[str, bool] = {}

    tool_ok = tools_match(tools_used, case.get("expected_tools"))
    if tool_ok is not None:
        checks["tools"] = tool_ok
    if case.get("answer_contains"):
        checks["answer_contains"] = contains_all(answer, case["answer_contains"])
    if case.get("expected_number") is not None:
        checks["number"] = numeric_match(answer, case["expected_number"], case.get("tolerance", 0.01))
    refused = is_refusal(answer)
    if case.get("should_refuse"):
        checks["refusal"] = refused

    cited = cited_sources(answer)
    precision, recall = citation_scores(cited, sources, case.get("expected_sources") or [])
    if case.get("expected_sources"):
        checks["cites_expected_source"] = bool(recall)

    return {
        "checks": checks,
        "passed": all(checks.values()),
        "refused": refused,
        "cited": cited,
        "citation_precision": precision,
        "citation_recall": recall,
    }


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (pct in 0..100)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, round(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def refusal_precision_recall(should_refuse: list[bool], refused: list[bool]) -> tuple[float | None, float | None]:
    true_positive = sum(1 for s, r in zip(should_refuse, refused) if s and r)
    predicted = sum(refused)
    actual = sum(should_refuse)
    precision = true_positive / predicted if predicted else None
    recall = true_positive / actual if actual else None
    return precision, recall


TOOL_LABELS = ["none", "search_knowledge_base", "query_financials", "calculator", "explain_harness"]


def primary_tool(tools: list[str]) -> str:
    """The first tool called in a turn, or 'none', used for the confusion matrix."""
    return tools[0] if tools else "none"


def confusion_matrix(expected: list[str], predicted: list[str], labels: list[str] = TOOL_LABELS) -> list[list[int]]:
    """matrix[i][j] = number of cases whose expected label is labels[i] and predicted is labels[j]."""
    counts = Counter(zip(expected, predicted))
    return [[counts[(e, p)] for p in labels] for e in labels]
