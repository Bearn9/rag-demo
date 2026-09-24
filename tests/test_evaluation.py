"""Unit tests for the pure eval metrics — no API key required."""

import pytest

from ragbot import evaluation


def test_retrieval_metrics():
    retrieved = ["a.md", "b.md", "c.md", "d.md"]
    assert evaluation.hit_at_k(retrieved, ["c.md"], k=2) == 0.0
    assert evaluation.hit_at_k(retrieved, ["c.md"], k=3) == 1.0
    assert evaluation.reciprocal_rank(retrieved, ["c.md"]) == pytest.approx(1 / 3)
    assert evaluation.reciprocal_rank(retrieved, ["z.md"]) == 0.0
    assert evaluation.recall_at_k(retrieved, ["a.md", "d.md"], k=2) == 0.5


def test_cited_sources_handles_multiple_citation_styles():
    answer = "Tailscale [source: it_setup_vpn_access.md]. Also [source: a.md, b.md]"
    assert evaluation.cited_sources(answer) == ["it_setup_vpn_access.md", "a.md", "b.md"]


@pytest.mark.parametrize(
    "answer,expected,tolerance,ok",
    [
        ("Net profit grew by **14.29%**.", 14.29, 0.05, True),
        ("It was $1,865,000 in Q2.", 1865000, 1, True),
        ("Cash reserves were $4.6 million.", 4600000, 1, True),
        ("Cash reserves were $4.6M.", 4600000, 1, True),
        ("About 12%.", 14.29, 0.05, False),
    ],
)
def test_numeric_match(answer, expected, tolerance, ok):
    assert evaluation.numeric_match(answer, expected, tolerance) is ok


@pytest.mark.parametrize(
    "answer,refused",
    [
        ("That information isn't in the knowledge base.", True),
        ("I couldn't find anything about pet insurance.", True),
        ("The documents don't mention a separate sick-day policy.", True),
        ("I can only help with questions about Bearn Analytics.", True),
        ("Writing poetry falls outside that scope, so I'll pass on this one.", True),
        ("That information isn't something I have access to.", True),
        ("I can't write general creative content like an ocean poem.", True),
        ("Data only covers Q1 2023 to Q2 2025, so Q3 2025 figures aren't available yet.", True),
        ("Benefits include medical and dental, but no pet insurance is mentioned.", True),
        ("Production services are available without the VPN.", False),
        ("New employees get 15 PTO days per year.", False),
    ],
)
def test_is_refusal(answer, refused):
    assert evaluation.is_refusal(answer) is refused


def test_tools_match():
    assert evaluation.tools_match([], []) is True
    assert evaluation.tools_match(["calculator"], []) is False
    assert evaluation.tools_match(["search_knowledge_base", "calculator"], ["calculator"]) is True
    assert evaluation.tools_match(["search_knowledge_base"], ["query_financials"]) is False
    assert evaluation.tools_match(["anything"], None) is None


def test_score_case_requires_every_applicable_check():
    case = {
        "expected_tools": ["search_knowledge_base"],
        "expected_sources": ["it_setup_vpn_access.md"],
        "answer_contains": ["Tailscale"],
    }
    good = evaluation.score_case(
        case, "We use Tailscale [source: it_setup_vpn_access.md].", ["search_knowledge_base"], ["it_setup_vpn_access.md"]
    )
    assert good["passed"] and good["citation_precision"] == 1.0 and good["citation_recall"] == 1.0

    uncited = evaluation.score_case(case, "We use Tailscale.", ["search_knowledge_base"], ["it_setup_vpn_access.md"])
    assert not uncited["passed"] and uncited["checks"]["cites_expected_source"] is False


def test_aggregates():
    assert evaluation.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert evaluation.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert evaluation.mean([1.0, None, 3.0]) == 2.0
    assert evaluation.refusal_precision_recall([True, True, False], [True, False, True]) == (0.5, 0.5)
    matrix = evaluation.confusion_matrix(["none", "calculator"], ["none", "none"], labels=["none", "calculator"])
    assert matrix == [[1, 0], [1, 0]]


def test_dataset_is_well_formed():
    cases = evaluation.load_dataset()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    for case in cases:
        assert case["turns"] and case["category"] and case.get("reference")
        for tool in case.get("expected_tools") or []:
            assert tool in evaluation.TOOL_LABELS
