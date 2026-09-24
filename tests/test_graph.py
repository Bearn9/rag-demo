"""End-to-end agent tests: the full tool-use loop against the Anthropic API, on a
sample of cases from evals/dataset.jsonl. Skipped automatically when ANTHROPIC_API_KEY
isn't set (e.g. forked CI runs). The full dataset runs via scripts/run_eval.py.
"""

import pytest

from ragbot import config, evaluation
from ragbot.graph import run, turn_totals

pytestmark = pytest.mark.skipif(not config.ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY not set")

SAMPLE_IDS = [
    "chat-hi",
    "harness-vector-db",
    "calc-hotel-high-cost",
    "fact-vpn",
    "fin-profit-growth",
    "turns-revenue-followup",
    "oos-ceo-salary",
]
CASES = {c["id"]: c for c in evaluation.load_dataset() if c["id"] in SAMPLE_IDS}


@pytest.mark.parametrize("case_id", SAMPLE_IDS)
def test_agent_case(case_id):
    case = CASES[case_id]
    history = []
    for turn in case["turns"]:
        state = run(turn, history=history)
        history += [{"role": "user", "content": turn}, {"role": "assistant", "content": state["answer"]}]

    totals = turn_totals(state["steps"])
    scored = evaluation.score_case(case, state["answer"], totals["tools_used"], state.get("sources") or [])
    assert scored["passed"], f"failed checks {scored['checks']}; answer: {state['answer']!r}"
    assert totals["steps"] <= 2 * config.MAX_AGENT_STEPS
