"""Deterministic tool tests — no LLM calls, no API key required."""

import pytest

from ragbot import config, tools

# --- query_financials -------------------------------------------------------


def test_financials_values_for_single_quarter():
    result = tools.query_financials(["Net Profit (USD)"], ["Q2 2025"], "values")
    assert result["results"]["Net Profit (USD)"] == {"Q2 2025": 560000}
    assert result["source"] == "economy_quarterly_financials.csv"


def test_financials_pct_change_between_quarters():
    result = tools.query_financials(["Net Profit (USD)"], ["Q1 2025", "Q2 2025"], "pct_change")
    # (560000 - 490000) / 490000 * 100
    assert result["results"]["Net Profit (USD)"] == pytest.approx(14.2857, abs=1e-3)


def test_financials_change_uses_chronological_order():
    # Quarters given out of order still compare first -> last chronologically.
    result = tools.query_financials(["Revenue (USD)"], ["Q2 2025", "Q1 2023"], "change")
    assert result["results"]["Revenue (USD)"] == 1865000 - 980000


def test_financials_sum_for_year():
    quarters = ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"]
    result = tools.query_financials(["Revenue (USD)"], quarters, "sum")
    assert result["results"]["Revenue (USD)"] == 1240000 + 1390000 + 1510000 + 1670000


def test_financials_max_over_all_quarters_reports_quarter():
    result = tools.query_financials(["Churn Rate (%)"], ["all"], "max")
    assert result["results"]["Churn Rate (%)"] == {"quarter": "Q1 2023", "value": 3.4}
    assert len(result["quarters"]) == 10


def test_financials_accepts_loose_quarter_formats():
    result = tools.query_financials(["Headcount"], ["q3-2024", "2025 Q1"], "values")
    assert result["results"]["Headcount"] == {"Q3 2024": 49, "Q1 2025": 55}


def test_financials_rejects_unknown_metric_and_missing_quarter():
    with pytest.raises(ValueError):
        tools.query_financials(["EBITDA"], ["all"], "sum")
    with pytest.raises(ValueError):
        tools.query_financials(["Revenue (USD)"], ["Q3 2025"], "values")


# --- calculator -------------------------------------------------------------


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("0.15 * 250", 37.5),
        ("15% * 250", 37.5),
        ("$1,865,000 - $1,720,000", 145000),
        ("round((560000 - 490000) / 490000 * 100, 2)", 14.29),
        ("2 ** 10", 1024),
        ("max(3, 7) - abs(-2)", 5),
    ],
)
def test_calculator_evaluates_arithmetic(expression, expected):
    assert tools.calculator(expression)["result"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('secrets.txt')",
        "(1).__class__",
        "x + 1",
        "9 ** 999999",
        "[1, 2, 3]",
    ],
)
def test_calculator_rejects_unsafe_or_unsupported_input(expression):
    with pytest.raises(ValueError):
        tools.calculator(expression)


def test_run_tool_turns_errors_into_error_results():
    result, is_error = tools.run_tool("calculator", {"expression": "1 / 0"})
    assert is_error and "error" in result
    result, is_error = tools.run_tool("does_not_exist", {})
    assert is_error


# --- explain_harness --------------------------------------------------------


def test_every_harness_topic_has_a_section():
    for topic in tools.HARNESS_TOPICS:
        result = tools.explain_harness(topic)
        assert result["content"], f"empty section for {topic}"
        assert result["source"] == "harness.md"


def test_explain_harness_includes_live_config():
    result = tools.explain_harness("models")
    assert result["live_config"]["agent_model"] == config.AGENT_MODEL
    assert set(result["live_config"]["tools"]) == {
        "search_knowledge_base",
        "query_financials",
        "calculator",
        "explain_harness",
    }


# --- search_knowledge_base ---------------------------------------------------


def test_search_knowledge_base_returns_sourced_chunks():
    result = tools.search_knowledge_base("what VPN tool does the company use")
    assert len(result["chunks"]) == config.RETRIEVAL_TOP_K
    assert result["chunks"][0]["source"] == "it_setup_vpn_access.md"
    assert tools.sources_from_result("search_knowledge_base", result)[0] == "it_setup_vpn_access.md"


def test_tool_schemas_are_strict():
    for tool in tools.TOOL_DEFINITIONS:
        assert tool["strict"] is True
        assert tool["input_schema"]["additionalProperties"] is False
