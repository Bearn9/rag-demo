"""Streamlit UI for the Bearn Agent Demo.

Run with:  streamlit run app.py
"""

import json
import sys
import threading
from datetime import date
from pathlib import Path

# Make `ragbot` importable without `pip install -e .` (e.g. on Streamlit Community Cloud).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import anthropic  # noqa: E402
import streamlit as st  # noqa: E402

from ragbot import config, tools  # noqa: E402
from ragbot.graph import get_graph, initial_state, turn_totals  # noqa: E402

REPO_URL = "https://github.com/Bearn9/rag-demo"

st.set_page_config(page_title="Bearn Agent Demo", page_icon="\N{COMPASS}", layout="wide")

SUGGESTIONS = [
     ("Multi-document lookup", "How does the home-office stipend compare to the wellness stipend?"),
     ("Finance math", "How much did net profit grow from Q1 to Q2 2025, in percent?"),
     ("Multilingual", "Forstår du dansk?"),
]

TOOL_ICONS = {
    "search_knowledge_base": "\N{LEFT-POINTING MAGNIFYING GLASS}",
    "query_financials": "\N{CHART WITH UPWARDS TREND}",
    "calculator": "\N{ABACUS}",
    "explain_harness": "\N{GEAR}",
}


@st.cache_resource(show_spinner=False)
def _graph():
    return get_graph()


@st.cache_resource(show_spinner=False)
def _daily_spend() -> dict:
    """Process-wide spend tracker shared by every visitor, reset each day."""
    return {"date": date.today().isoformat(), "usd": 0.0, "lock": threading.Lock()}


def _spent_today() -> float:
    tracker = _daily_spend()
    with tracker["lock"]:
        if tracker["date"] != date.today().isoformat():
            tracker["date"], tracker["usd"] = date.today().isoformat(), 0.0
        return tracker["usd"]


def _record_spend(usd: float) -> None:
    tracker = _daily_spend()
    with tracker["lock"]:
        tracker["usd"] += usd


def _escape_dollar_math(text: str) -> str:
    """Escape '$' so Streamlit's markdown renderer doesn't mistake dollar amounts
    (e.g. "$490,000 ... $1,720,000") for paired LaTeX math delimiters."""
    return text.replace("$", "\\$")


# ---------------------------------------------------------------------------
# Trace rendering
# ---------------------------------------------------------------------------


def _describe_call(call: dict) -> str:
    name, args = call["name"], call["input"]
    if name == "search_knowledge_base":
        return f"Searching the knowledge base for “{args.get('query', '')}”"
    if name == "query_financials":
        metrics = ", ".join(args.get("metrics", []))
        return f"Computing {args.get('operation')} of {metrics} ({', '.join(args.get('quarters', []))})"
    if name == "calculator":
        return f"Calculating {args.get('expression')}"
    if name == "explain_harness":
        return f"Looking up how I work: {args.get('topic')}"
    return f"Calling {name}"


def _summary_line(steps: list[dict]) -> str:
    totals = turn_totals(steps)
    if totals["tools_used"]:
        chain = " → ".join(f"{TOOL_ICONS.get(t, '')} {t}" for t in totals["tools_used"])
        route = f"Used tools: {chain}"
    else:
        route = "\N{SPEECH BALLOON} Answered directly, no tools needed"
    return (
        f"{route}  ·  {totals['steps']} steps  ·  {totals['latency_ms'] / 1000:.1f}s  ·  "
        f"{totals['input_tokens'] + totals['output_tokens']:,} tokens  ·  ${totals['cost_usd']:.4f}"
    )


def _render_tool_output(step: dict) -> None:
    output = step["output"]
    if step["is_error"]:
        st.error(output.get("error", "Tool error"))
    elif step["name"] == "search_knowledge_base":
        for chunk in output.get("chunks", []):
            with st.container(border=True):
                st.markdown(f"**{chunk['source']}**: {chunk['section']} · similarity {chunk['score']:.2f}")
                st.caption(_escape_dollar_math(chunk["text"][:400] + ("…" if len(chunk["text"]) > 400 else "")))
    elif step["name"] == "query_financials":
        st.json(output["results"], expanded=True)
        st.caption(f"Quarters used: {', '.join(output['quarters'])} · source: {output['source']}")
    elif step["name"] == "calculator":
        st.markdown(f"`{output['expression']}` = **{output['result']}**")
    elif step["name"] == "explain_harness":
        st.caption(f"Section `{output['topic']}` of {output['source']}, plus live config values")
    else:
        st.json(output)


def _render_trace(steps: list[dict]) -> None:
    with st.expander("Agent trace: what the model did, step by step"):
        for i, step in enumerate(steps, start=1):
            if step["kind"] == "model":
                st.markdown(
                    f"**{i}. \N{BRAIN} {step['label']}**  \n"
                    f"`{step['model']}` · {step['latency_ms'] / 1000:.1f}s · "
                    f"{step['input_tokens']:,} in / {step['output_tokens']:,} out tokens · \\${step['cost_usd']:.4f}"
                )
                if step.get("thinking"):
                    st.caption("Reasoning summary: " + _escape_dollar_math(step["thinking"]))
            else:
                icon = TOOL_ICONS.get(step["name"], "\N{WRENCH}")
                st.markdown(f"**{i}. {icon} Tool: `{step['name']}`** · {step['latency_ms']:.0f} ms")
                st.code(json.dumps(step["input"], indent=2, ensure_ascii=False), language="json")
                _render_tool_output(step)

        sources = sorted({s for step in steps if step["kind"] == "tool" for s in step.get("sources", [])})
        if sources:
            st.markdown("**Sources used:** " + ", ".join(f"`{s}`" for s in sources))


def _render_meta(steps: list[dict], show_trace: bool) -> None:
    st.caption(_escape_dollar_math(_summary_line(steps)))
    if show_trace:
        _render_trace(steps)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.questions_asked = 0

with st.sidebar:
    st.markdown("### Bearn Agent Demo")
    st.caption("Claude API - LangGraph - FAISS")
    st.markdown(
        "An agentic RAG Demo, using made up documents from a fictional company Bearn Analytics for retrieval. "
        "For each question, a Claude model decides how to process the query, "
        "whether tool-calling such as document lookup using vector search is relevant. "
        "The agent then generates an answer with the cited references. " 
        "The trace under each answer shows every step and the cost of the query. "
        "More information is available on the GitHub repo"
    )
    show_trace = st.toggle("Show agent trace", value=True)
    # Filled in at the end of the script, after any question in this run has been counted.
    questions_left_slot = st.empty()
    if st.button("New conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.markdown(f"[Source code on GitHub]({REPO_URL})")
    st.caption("All company data is fictional.")

st.title("\N{COMPASS} Bearn Agent Demo")

chat_tab, eval_tab, how_tab = st.tabs(["\N{SPEECH BALLOON} Chat", "\N{BAR CHART} Evaluation", "\N{GEAR} How it works"])

# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------


def _guardrail_message(query: str) -> str | None:
    if not config.ANTHROPIC_API_KEY:
        return "This deployment has no `ANTHROPIC_API_KEY` configured."
    if len(query) > config.MAX_INPUT_CHARS:
        return f"Please keep questions under {config.MAX_INPUT_CHARS} characters."
    if st.session_state.questions_asked >= config.MAX_QUESTIONS_PER_SESSION:
        return (
            f"You've reached this demo's limit of {config.MAX_QUESTIONS_PER_SESSION} questions per session. "
            f"Thanks for trying it! To keep going, clone the [repo]({REPO_URL}) and run it with your own key."
        )
    if _spent_today() >= config.DAILY_BUDGET_USD:
        return (
            "The demo has reached its daily usage budget and will be back tomorrow. "
            f"Meanwhile, the Evaluation tab and the [source code]({REPO_URL}) are still available."
        )
    return None


def _answer(query: str) -> None:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(_escape_dollar_math(query))

    with st.chat_message("assistant"):
        blocked = _guardrail_message(query)
        if blocked:
            st.info(blocked)
            st.session_state.messages.pop()
            return

        st.session_state.questions_asked += 1
        status = st.status("Thinking about which tools to use…", expanded=False)
        final_state: dict = {}
        seen = 0
        try:
            for final_state in _graph().stream(initial_state(query, history), stream_mode="values"):
                steps = final_state.get("steps", [])
                for step in steps[seen:]:
                    if step["kind"] == "model" and step["tool_calls"]:
                        status.update(label=" · ".join(_describe_call(c) for c in step["tool_calls"]) + "…")
                    elif step["kind"] == "model":
                        status.update(label="Done", state="complete")
                    else:
                        status.update(label=f"Got results from {step['name']}, deciding the next step…")
                seen = len(steps)
        except anthropic.APIError as exc:
            status.update(label="Error", state="error")
            st.error(f"The model API returned an error ({type(exc).__name__}). Please try again in a moment.")
            st.session_state.messages.pop()
            return

        steps = final_state.get("steps", [])
        _record_spend(turn_totals(steps)["cost_usd"])
        answer = final_state.get("answer") or "Sorry, something went wrong generating a response."
        st.markdown(_escape_dollar_math(answer))
        _render_meta(steps, show_trace)

    st.session_state.messages.append({"role": "assistant", "content": answer, "steps": steps})


with chat_tab:
    if not st.session_state.messages:
        st.markdown("**Try one of these, or ask your own question below:**")
        cols = st.columns(3)
        for i, (label, question) in enumerate(SUGGESTIONS):
            if cols[i % 3].button(f"**{label}**  \n{question}", key=f"suggest-{i}", width="stretch"):
                st.session_state.pending_query = question

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(_escape_dollar_math(message["content"]))
            if message["role"] == "assistant" and message.get("steps"):
                _render_meta(message["steps"], show_trace)

    typed = st.chat_input("Ask about PTO, benefits, IT, security, pricing, expenses, financials, or how the agent works…")
    query = typed or st.session_state.pop("pending_query", None)
    if query:
        _answer(query)
        if len(st.session_state.messages) == 2:
            st.rerun()  # hide the suggestion buttons after the first answer

# ---------------------------------------------------------------------------
# Evaluation tab (read-only: shows committed results, never calls the API)
# ---------------------------------------------------------------------------


def _pct(value) -> str:
    return "–" if value is None else f"{value * 100:.0f}%"


def _score(value) -> str:
    return "–" if value is None else f"{value:.2f} / 5"


@st.cache_data(show_spinner=False)
def _load_eval_results() -> dict | None:
    if not config.EVAL_RESULTS_PATH.exists():
        return None
    return json.loads(config.EVAL_RESULTS_PATH.read_text(encoding="utf-8"))


with eval_tab:
    results = _load_eval_results()
    st.markdown(
        "The evaluation is run locally before comitting the results to GitHub. This uses LLM-as-judge and deterministic metrics "
        "over a labelled dataset of questions. This tab displays the last comitted results."
    )
    if results is None:
        st.info("No evaluation results have been committed yet. Run `python scripts/run_eval.py --judge` to generate them.")
    else:
        h = results["headline"]
        st.caption(
            f"Run {results['run_at']} · commit `{results['git_sha']}` · agent `{results['agent_model']}`"
            + (f" · judge `{results['judge_model']}`" if results.get("judge_model") else "")
            + f" · {h['cases']} cases · total eval cost ${results['total_eval_cost_usd']:.2f}"
        )
        row1 = st.columns(4)
        row1[0].metric("Case pass rate", _pct(h["pass_rate"]), help="Share of cases where every applicable check passed.")
        row1[1].metric("Tool-selection accuracy", _pct(h["tool_selection_accuracy"]))
        row1[2].metric("Judge faithfulness", _score(h["faithfulness"]), help="Are claims supported by the tool results?")
        row1[3].metric("Judge correctness", _score(h["correctness"]))
        row2 = st.columns(4)
        row2[0].metric("Citation precision", _pct(h["citation_precision"]), help="Cited files that the tools actually returned.")
        row2[1].metric("Refusal recall", _pct(h["refusal_recall"]), help="Out-of-scope questions correctly declined.")
        row2[2].metric("Latency p50 / p95", f"{h['latency_p50_s']:.1f}s / {h['latency_p95_s']:.1f}s")
        row2[3].metric("Cost per question", f"${h['cost_per_question_usd']:.4f}")

        left, right = st.columns(2)
        with left:
            st.subheader("By category")
            st.dataframe(
                [
                    {
                        "Category": r["category"],
                        "Cases": r["cases"],
                        "Pass rate": _pct(r["pass_rate"]),
                        "Tool accuracy": _pct(r["tool_selection_accuracy"]),
                        "Correctness": _score(r["correctness"]),
                    }
                    for r in results["by_category"]
                ],
                hide_index=True,
                width="stretch",
            )
        with right:
            retrieval = results["retrieval"]
            st.subheader("Retrieval quality")
            st.caption(f"{retrieval['cases']} single-turn cases, file level, no LLM · MRR {retrieval['mrr']:.3f}")
            st.dataframe(
                [
                    {"k": r["k"], "hit@k": f"{r['hit_at_k']:.3f}", "recall@k": f"{r['recall_at_k']:.3f}"}
                    for r in retrieval["rows"]
                ],
                hide_index=True,
                width="stretch",
            )

        st.subheader("Tool selection: expected vs actual first tool")
        confusion = results["tool_confusion"]
        st.dataframe(
            [
                {"expected ↓ / actual →": label, **dict(zip(confusion["labels"], row))}
                for label, row in zip(confusion["labels"], confusion["matrix"])
            ],
            hide_index=True,
            width="stretch",
        )

        if results.get("model_comparison"):
            st.subheader("Agent model comparison")
            st.dataframe(
                [
                    {
                        "Model": m["model"],
                        "Pass rate": _pct(m["pass_rate"]),
                        "Tool accuracy": _pct(m["tool_selection_accuracy"]),
                        "Faithfulness": _score(m["faithfulness"]),
                        "Correctness": _score(m["correctness"]),
                        "Latency p50": f"{m['latency_p50_s']:.1f}s" if m["latency_p50_s"] else "–",
                        "Cost / question": f"${m['cost_per_question_usd']:.4f}" if m["cost_per_question_usd"] else "–",
                    }
                    for m in results["model_comparison"]
                ],
                hide_index=True,
                width="stretch",
            )

        failures = [c for c in results["cases"] if not c.get("passed")]
        with st.expander(f"Failed cases ({len(failures)})"):
            for case in failures:
                failed = [k for k, v in case.get("checks", {}).items() if not v]
                st.markdown(f"**{case['id']}** ({case['category']}): failed {', '.join(failed) or case.get('error', '')}")
                st.caption(_escape_dollar_math(f"Q: {case['question']}"))
                st.caption(_escape_dollar_math(f"A: {case.get('answer', '')}"))
                if case.get("judge"):
                    st.caption(f"Judge: {case['judge']['rationale']}")

        docked = [
            c for c in results["cases"]
            if c.get("judge") and min(c["judge"]["faithfulness"], c["judge"]["correctness"]) < 5
        ]
        with st.expander(f"Where the judge docked points ({len(docked)})"):
            for case in docked:
                j = case["judge"]
                st.markdown(
                    f"**{case['id']}** ({case['category']}): faithfulness {j['faithfulness']}/5, "
                    f"correctness {j['correctness']}/5"
                )
                st.caption(_escape_dollar_math(f"Q: {case['question']}"))
                st.caption(_escape_dollar_math(f"Judge: {j['rationale']}"))

        with st.expander("All cases"):
            st.dataframe(
                [
                    {
                        "id": c["id"],
                        "category": c["category"],
                        "passed": c.get("passed"),
                        "tools used": ", ".join(c.get("tools_used") or []) or "none",
                        "faithfulness": (c.get("judge") or {}).get("faithfulness"),
                        "correctness": (c.get("judge") or {}).get("correctness"),
                        "latency (s)": round(c.get("latency_s") or 0, 1),
                    }
                    for c in results["cases"]
                ],
                hide_index=True,
                width="stretch",
            )

    with st.expander("Methodology", expanded=results is None):
        st.markdown(tools.explain_harness("evaluation")["content"])
        st.markdown(
            "Each case in `evals/dataset.jsonl` lists the expected tools, expected source files, "
            "expected substrings or numbers (with a tolerance), and whether the agent should decline. "
            "A case passes only if every applicable check passes. The LLM judge scores each answer "
            "against the tool results it actually saw and a reference note."
        )

# ---------------------------------------------------------------------------
# How it works tab
# ---------------------------------------------------------------------------

with how_tab:
    sections = tools.explain_harness("overview")
    st.markdown(sections["content"])
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR; node [shape=box, style="rounded", fontname="Helvetica"];
            start [label="User question\\n+ recent turns", shape=oval];
            agent [label="agent\\n(Claude + tool definitions)"];
            tools [label="tools\\nsearch_knowledge_base · query_financials\\ncalculator · explain_harness"];
            answer [label="Cited answer\\n+ trace", shape=oval];
            start -> agent;
            agent -> tools [label="stop_reason = tool_use"];
            tools -> agent [label="tool results"];
            agent -> answer [label="final text"];
        }
        """
    )
    for topic in ("architecture", "tools", "retrieval", "guardrails", "tech_stack"):
        with st.expander(topic.replace("_", " ").title()):
            st.markdown(tools.explain_harness(topic)["content"])
    st.caption(f"Live configuration: `{json.dumps(tools.live_config())}`")

remaining = max(0, config.MAX_QUESTIONS_PER_SESSION - st.session_state.questions_asked)
questions_left_slot.caption(f"Questions left this session: {remaining} of {config.MAX_QUESTIONS_PER_SESSION}")
