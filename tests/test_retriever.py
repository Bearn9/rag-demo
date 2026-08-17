"""Deterministic retrieval tests — no LLM calls, no API key required.

These rely on the vector store built by `python -m ragbot.ingest` already
existing (it's committed to the repo; re-run ingest if data/knowledge_base/
changes).
"""

from ragbot import retriever


def test_pto_query_retrieves_pto_doc():
    results = retriever.retrieve("how many PTO days do new employees get", k=3)
    assert results, "expected at least one retrieved chunk"
    assert results[0]["source"] == "hr_policies_pto.md"


def test_vpn_query_retrieves_vpn_doc_not_laptop_doc():
    results = retriever.retrieve("what VPN tool does the company use", k=3)
    assert results[0]["source"] == "it_setup_vpn_access.md"


def test_enterprise_refresh_query_retrieves_features_doc():
    results = retriever.retrieve("what is the data refresh rate on the enterprise plan", k=3)
    assert results[0]["source"] == "product_faq_features.md"


def test_results_are_ordered_by_descending_score():
    results = retriever.retrieve("password requirements", k=4)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_funding_query_retrieves_annual_report_doc():
    results = retriever.retrieve("who led the seed funding round", k=3)
    assert results[0]["source"] == "economy_annual_report_summary.md"


def test_hotel_cap_query_retrieves_expense_policy_doc():
    results = retriever.retrieve("what is the hotel spend cap for business travel", k=3)
    assert results[0]["source"] == "finance_expense_policy.md"
