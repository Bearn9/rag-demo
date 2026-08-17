"""End-to-end graph tests: routing + retrieval + generation, using the same
EVAL_CASES table as scripts/run_eval.py. These hit the Anthropic API and are
skipped automatically when ANTHROPIC_API_KEY isn't set (e.g. forked CI runs).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ragbot import config
from ragbot.graph import get_graph
from run_eval import EVAL_CASES

pytestmark = pytest.mark.skipif(
    not config.ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY not set"
)


@pytest.mark.parametrize("query,expected_route,expected_source,expected_substring", EVAL_CASES)
def test_eval_case(query, expected_route, expected_source, expected_substring):
    graph = get_graph()
    final_state = graph.invoke({"query": query})

    assert final_state["route"] == expected_route

    if expected_source is not None:
        sources = [c["source"] for c in (final_state.get("retrieved_chunks") or [])]
        assert expected_source in sources

    if expected_substring is not None:
        assert expected_substring.lower() in final_state["answer"].lower()
