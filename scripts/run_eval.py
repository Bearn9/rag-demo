"""CLI eval harness: runs a table of sample queries through the graph and checks
that routing and answers behave as expected. Requires ANTHROPIC_API_KEY.

Run with:  python scripts/run_eval.py

The same EVAL_CASES table backs tests/test_graph.py via pytest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragbot.graph import get_graph  # noqa: E402

# (query, expected_route, expected_source_or_None, expected_answer_substring_or_None)
EVAL_CASES = [
    ("hi", "direct", None, None),
    ("what can you do?", "direct", None, None),
    ("thanks!", "direct", None, None),
    (
        "How many PTO days do new employees get?",
        "retrieve",
        "hr_policies_pto.md",
        "15",
    ),
    (
        "What VPN tool does the company use?",
        "retrieve",
        "it_setup_vpn_access.md",
        "Tailscale",
    ),
    (
        "What's the data refresh rate on the Enterprise plan?",
        "retrieve",
        "product_faq_features.md",
        "real-time",
    ),
    (
        "Who led the company's seed funding round?",
        "retrieve",
        "economy_annual_report_summary.md",
        "Cascade Ventures",
    ),
]


def run_case(query, expected_route, expected_source, expected_substring):
    graph = get_graph()
    final_state = graph.invoke({"query": query})

    ok = True
    if final_state["route"] != expected_route:
        ok = False
        print(f"  FAIL route: expected {expected_route!r}, got {final_state['route']!r}")

    if expected_source is not None:
        sources = [c["source"] for c in (final_state.get("retrieved_chunks") or [])]
        if expected_source not in sources:
            ok = False
            print(f"  FAIL source: expected {expected_source!r} in {sources}")

    if expected_substring is not None:
        if expected_substring.lower() not in final_state["answer"].lower():
            ok = False
            print(f"  FAIL substring: expected {expected_substring!r} in answer")

    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {query!r} -> route={final_state['route']!r}")
    return ok


def main():
    results = [run_case(*case) for case in EVAL_CASES]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} cases passed.")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
