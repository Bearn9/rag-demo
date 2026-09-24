"""Offline evaluation harness for the Bearn Agent Demo.

Retrieval-only (free, deterministic, no API key; this is what CI runs):
    python scripts/run_eval.py --retrieval-only [--min-hit-at-4 0.9]

Full agent eval (calls Claude; roughly $0.01 per case, plus about $0.02 per case with --judge):
    python scripts/run_eval.py --judge
    python scripts/run_eval.py --judge --compare-models claude-haiku-4-5

Re-apply changed checks to the saved answers without new API calls:
    python scripts/run_eval.py --rescore

A full run writes evals/results/latest.json (read by the app's Evaluation tab and by
the explain_harness tool) and evals/results/REPORT.md. Only the project owner runs
this; demo visitors only ever see the committed results.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import anthropic  # noqa: E402

from ragbot import config, evaluation, llm, retriever  # noqa: E402
from ragbot.graph import run, turn_totals  # noqa: E402

RETRIEVAL_K_VALUES = [1, 2, 4, 6]
REPORT_PATH = config.EVAL_RESULTS_PATH.parent / "REPORT.md"


# ---------------------------------------------------------------------------
# Retrieval eval
# ---------------------------------------------------------------------------


def retrieval_eval(cases: list[dict]) -> dict:
    """File-level hit@k / MRR / recall@k for single-turn cases whose expected sources
    are knowledge-base documents (the harness doc isn't in the index)."""
    kb_files = {p.name for p in config.DATA_DIR.iterdir()}
    usable = [
        c
        for c in cases
        if len(c["turns"]) == 1
        and c.get("expected_sources")
        and set(c["expected_sources"]) <= kb_files
    ]
    max_k = max(RETRIEVAL_K_VALUES)
    ranked = []
    for case in usable:
        chunks = retriever.retrieve(case["turns"][0], k=max_k)
        ranked.append((case, [chunk["source"] for chunk in chunks]))

    rows = []
    for k in RETRIEVAL_K_VALUES:
        rows.append(
            {
                "k": k,
                "hit_at_k": evaluation.mean([evaluation.hit_at_k(r, c["expected_sources"], k) for c, r in ranked]),
                "recall_at_k": evaluation.mean([evaluation.recall_at_k(r, c["expected_sources"], k) for c, r in ranked]),
            }
        )
    misses = [
        {"id": c["id"], "query": c["turns"][0], "expected": c["expected_sources"], "top_4": r[:4]}
        for c, r in ranked
        if not evaluation.hit_at_k(r, c["expected_sources"], config.RETRIEVAL_TOP_K)
    ]
    return {
        "cases": len(ranked),
        "mrr": evaluation.mean([evaluation.reciprocal_rank(r, c["expected_sources"]) for c, r in ranked]),
        "rows": rows,
        "misses_at_configured_k": misses,
        "configured_k": config.RETRIEVAL_TOP_K,
    }


def print_retrieval(result: dict) -> None:
    print(f"\nRetrieval ({result['cases']} cases, file-level)   MRR = {result['mrr']:.3f}")
    for row in result["rows"]:
        print(f"  k={row['k']}: hit@k = {row['hit_at_k']:.3f}   recall@k = {row['recall_at_k']:.3f}")
    for miss in result["misses_at_configured_k"]:
        print(f"  MISS {miss['id']}: expected {miss['expected']}, got {miss['top_4']}")


# ---------------------------------------------------------------------------
# Agent eval
# ---------------------------------------------------------------------------


def _tool_results_text(steps: list[dict], limit: int = 8000) -> str:
    parts = [
        f"{s['name']}({json.dumps(s['input'])}) -> {json.dumps(s['output'], ensure_ascii=False)}"
        for s in steps
        if s["kind"] == "tool"
    ]
    return "\n\n".join(parts)[:limit]


def run_case(case: dict, model: str, judge: bool) -> dict:
    history: list[dict] = []
    cost = 0.0
    state = {}
    latency_s = 0.0
    # Evidence accumulates across turns: a follow-up may rightly answer from an earlier
    # turn's tool results, so citations and the judge consider the whole conversation.
    all_steps: list[dict] = []
    all_sources: list[str] = []
    for turn in case["turns"]:
        started = time.perf_counter()
        state = run(turn, history=history, model=model)
        latency_s = time.perf_counter() - started  # final turn's wall time
        cost += turn_totals(state["steps"])["cost_usd"]
        all_steps += state["steps"]
        all_sources += state.get("sources") or []
        history += [{"role": "user", "content": turn}, {"role": "assistant", "content": state["answer"]}]

    totals = turn_totals(state["steps"])  # tool selection is scored on the final turn
    answer = state["answer"] or ""
    scored = evaluation.score_case(case, answer, totals["tools_used"], all_sources)

    result = {
        "id": case["id"],
        "category": case["category"],
        "question": " → ".join(case["turns"]),
        "answer": answer,
        "tools_used": totals["tools_used"],
        "expected_tools": case.get("expected_tools"),
        "sources": sorted(set(all_sources)),
        "latency_s": latency_s,
        "agent_cost_usd": cost,
        **scored,
    }
    if judge:
        conversation = "\n".join(f"{m['role']}: {m['content']}" for m in history[:-2])
        question = f"{conversation}\nuser: {case['turns'][-1]}" if conversation else case["turns"][-1]
        verdict = llm.judge_answer(question, _tool_results_text(all_steps), case.get("reference", ""), answer)
        result["judge"] = {k: verdict[k] for k in ("faithfulness", "correctness", "rationale")}
        cost += verdict["cost_usd"]
    result["total_cost_usd"] = cost
    return result


def summarise(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    tool_checks = [r["checks"]["tools"] for r in ok if "tools" in r["checks"]]
    refusal_p, refusal_r = evaluation.refusal_precision_recall(
        [r["category"] == "out-of-scope" for r in ok], [r["refused"] for r in ok]
    )
    judged = [r["judge"] for r in ok if "judge" in r]
    return {
        "cases": len(results),
        "errors": len(results) - len(ok),
        "pass_rate": evaluation.mean([float(r["passed"]) for r in ok]),
        "tool_selection_accuracy": evaluation.mean([float(t) for t in tool_checks]),
        "citation_precision": evaluation.mean([r["citation_precision"] for r in ok]),
        "citation_recall": evaluation.mean([r["citation_recall"] for r in ok]),
        "refusal_precision": refusal_p,
        "refusal_recall": refusal_r,
        "faithfulness": evaluation.mean([j["faithfulness"] for j in judged]),
        "correctness": evaluation.mean([j["correctness"] for j in judged]),
        "latency_p50_s": evaluation.percentile([r["latency_s"] for r in ok], 50),
        "latency_p95_s": evaluation.percentile([r["latency_s"] for r in ok], 95),
        "cost_per_question_usd": evaluation.mean([r["agent_cost_usd"] for r in ok]),
    }


def by_category(results: list[dict]) -> list[dict]:
    rows = []
    for category in dict.fromkeys(r["category"] for r in results):
        group = [r for r in results if r["category"] == category and "error" not in r]
        tool_checks = [float(r["checks"]["tools"]) for r in group if "tools" in r["checks"]]
        rows.append(
            {
                "category": category,
                "cases": len(group),
                "pass_rate": evaluation.mean([float(r["passed"]) for r in group]),
                "tool_selection_accuracy": evaluation.mean(tool_checks),
                "correctness": evaluation.mean([r["judge"]["correctness"] for r in group if "judge" in r]),
            }
        )
    return rows


def tool_confusion(results: list[dict]) -> dict:
    scored = [r for r in results if "error" not in r and r.get("expected_tools") is not None]
    expected = [evaluation.primary_tool(r["expected_tools"]) for r in scored]
    predicted = [evaluation.primary_tool(r["tools_used"]) for r in scored]
    return {"labels": evaluation.TOOL_LABELS, "matrix": evaluation.confusion_matrix(expected, predicted)}


def agent_eval(cases: list[dict], model: str, judge: bool, budget: dict) -> list[dict]:
    results = []
    for i, case in enumerate(cases, start=1):
        if budget["spent"] >= budget["max"]:
            print(f"  Budget of ${budget['max']:.2f} reached; stopping after {i - 1} cases.")
            break
        try:
            result = run_case(case, model, judge)
        except (anthropic.APIError, ValueError, KeyError) as exc:  # keep going; record the failure
            result = {"id": case["id"], "category": case["category"], "question": " → ".join(case["turns"]),
                      "error": f"{type(exc).__name__}: {exc}", "passed": False, "total_cost_usd": 0.0}
        budget["spent"] += result["total_cost_usd"]
        results.append(result)
        status = "ERROR" if "error" in result else ("PASS" if result["passed"] else "FAIL")
        failed = [k for k, v in result.get("checks", {}).items() if not v]
        print(f"  [{status}] {i:>2}/{len(cases)} {case['id']:<28} tools={result.get('tools_used')} "
              f"{'failed=' + str(failed) if failed else ''}{result.get('error', '')}")
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(value, kind="pct"):
    if value is None:
        return "–"
    if kind == "pct":
        return f"{value * 100:.0f}%"
    if kind == "score":
        return f"{value:.2f} / 5"
    if kind == "s":
        return f"{value:.1f}s"
    if kind == "usd":
        return f"${value:.4f}"
    return str(value)


def write_report(payload: dict) -> None:
    h = payload["headline"]
    lines = [
        "# Evaluation report",
        "",
        f"Run {payload['run_at']} · commit `{payload['git_sha']}` · agent `{payload['agent_model']}`"
        + (f" · judge `{payload['judge_model']}`" if payload.get("judge_model") else "")
        + f" · {h['cases']} cases · eval cost ${payload['total_eval_cost_usd']:.2f}"
        + (f" · checks re-applied {payload['rescored_at']}" if payload.get("rescored_at") else ""),
        "",
        (
            "Generated by `python scripts/run_eval.py`. Methodology: see the \"evaluation\" section of "
            "[`docs/harness.md`](../../docs/harness.md)."
        ),
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Case pass rate (all checks) | {_fmt(h['pass_rate'])} |",
        f"| Tool-selection accuracy | {_fmt(h['tool_selection_accuracy'])} |",
        f"| Citation precision / recall | {_fmt(h['citation_precision'])} / {_fmt(h['citation_recall'])} |",
        f"| Refusal precision / recall (out-of-scope) | {_fmt(h['refusal_precision'])} / {_fmt(h['refusal_recall'])} |",
        f"| LLM-judge faithfulness | {_fmt(h['faithfulness'], 'score')} |",
        f"| LLM-judge correctness | {_fmt(h['correctness'], 'score')} |",
        f"| Latency p50 / p95 | {_fmt(h['latency_p50_s'], 's')} / {_fmt(h['latency_p95_s'], 's')} |",
        f"| Cost per question | {_fmt(h['cost_per_question_usd'], 'usd')} |",
        "",
        "## Retrieval (no LLM, file-level)",
        "",
        f"{payload['retrieval']['cases']} single-turn cases · MRR {payload['retrieval']['mrr']:.3f}",
        "",
        "| k | hit@k | recall@k |",
        "|---|---|---|",
        *[f"| {r['k']} | {r['hit_at_k']:.3f} | {r['recall_at_k']:.3f} |" for r in payload["retrieval"]["rows"]],
        "",
        "## By category",
        "",
        "| Category | Cases | Pass rate | Tool accuracy | Judge correctness |",
        "|---|---|---|---|---|",
        *[
            f"| {r['category']} | {r['cases']} | {_fmt(r['pass_rate'])} | {_fmt(r['tool_selection_accuracy'])} | "
            f"{_fmt(r['correctness'], 'score')} |"
            for r in payload["by_category"]
        ],
    ]
    if payload.get("model_comparison"):
        lines += [
            "",
            "## Model comparison",
            "",
            "| Agent model | Pass rate | Tool accuracy | Faithfulness | Correctness | Latency p50 | Cost / question |",
            "|---|---|---|---|---|---|---|",
            *[
                f"| `{m['model']}` | {_fmt(m['pass_rate'])} | {_fmt(m['tool_selection_accuracy'])} | "
                f"{_fmt(m['faithfulness'], 'score')} | {_fmt(m['correctness'], 'score')} | "
                f"{_fmt(m['latency_p50_s'], 's')} | {_fmt(m['cost_per_question_usd'], 'usd')} |"
                for m in payload["model_comparison"]
            ],
        ]
    failures = [c for c in payload["cases"] if not c.get("passed")]
    lines += ["", f"## Failed cases ({len(failures)})", ""]
    for c in failures:
        failed = [k for k, v in c.get("checks", {}).items() if not v]
        lines.append(f"- **{c['id']}** ({c['category']}): {c.get('error') or 'failed ' + ', '.join(failed)}. "
                     f"Tools: {c.get('tools_used')}.")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_results(payload: dict) -> None:
    config.EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.EVAL_RESULTS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(payload)
    print(f"Wrote {config.EVAL_RESULTS_PATH.relative_to(config.ROOT_DIR)} and {REPORT_PATH.relative_to(config.ROOT_DIR)}")


def rescore(cases: list[dict], retrieval: dict) -> None:
    """Re-run the deterministic checks over the answers saved in latest.json. The agent
    answers and judge scores are unchanged; only the string/number/tool checks move."""
    payload = json.loads(config.EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in cases}
    for result in payload["cases"]:
        if "error" in result or result["id"] not in by_id:
            continue
        result.update(evaluation.score_case(by_id[result["id"]], result["answer"], result["tools_used"], result["sources"]))
    results = payload["cases"]
    payload.update(
        headline=summarise(results),
        retrieval=retrieval,
        by_category=by_category(results),
        tool_confusion=tool_confusion(results),
        rescored_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
    if payload.get("model_comparison"):
        payload["model_comparison"][0] = {"model": payload["agent_model"], **payload["headline"]}
    print(f"\nRescored {len(results)} saved answers: pass rate {_fmt(payload['headline']['pass_rate'])}")
    _write_results(payload)


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=config.ROOT_DIR, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval-only", action="store_true", help="Only run the free retrieval metrics.")
    parser.add_argument("--min-hit-at-4", type=float, help="Exit non-zero if hit@4 falls below this.")
    parser.add_argument("--agent-model", default=config.AGENT_MODEL)
    parser.add_argument("--compare-models", nargs="*", default=[], help="Extra agent models to summarise.")
    parser.add_argument("--judge", action="store_true", help=f"Grade answers with {config.JUDGE_MODEL}.")
    parser.add_argument("--max-cost", type=float, default=5.0, help="Stop once this much USD has been spent.")
    parser.add_argument("--only", nargs="*", help="Run only these case ids (no results file is written).")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Re-apply the current deterministic checks to the saved answers in latest.json (no API calls).",
    )
    args = parser.parse_args()

    cases = evaluation.load_dataset()
    retrieval = retrieval_eval(cases)
    print_retrieval(retrieval)

    if args.min_hit_at_4 is not None:
        hit4 = next(r["hit_at_k"] for r in retrieval["rows"] if r["k"] == 4)
        if hit4 < args.min_hit_at_4:
            raise SystemExit(f"hit@4 = {hit4:.3f} is below the required {args.min_hit_at_4}")
    if args.retrieval_only:
        return
    if args.rescore:
        rescore(cases, retrieval)
        return

    if not config.ANTHROPIC_API_KEY:
        raise SystemExit("ANTHROPIC_API_KEY is not set; use --retrieval-only for the free metrics.")

    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]
    budget = {"spent": 0.0, "max": args.max_cost}

    print(f"\nAgent eval: {len(cases)} cases on {args.agent_model}" + (" (+ judge)" if args.judge else ""))
    results = agent_eval(cases, args.agent_model, args.judge, budget)
    headline = summarise(results)

    comparison = [{"model": args.agent_model, **headline}]
    for model in args.compare_models:
        print(f"\nComparison run: {model}")
        comparison.append({"model": model, **summarise(agent_eval(cases, model, args.judge, budget))})

    print(f"\nPass rate {_fmt(headline['pass_rate'])} · tool accuracy {_fmt(headline['tool_selection_accuracy'])} · "
          f"faithfulness {_fmt(headline['faithfulness'], 'score')} · p50 {_fmt(headline['latency_p50_s'], 's')} · "
          f"{_fmt(headline['cost_per_question_usd'], 'usd')}/question · total spent ${budget['spent']:.2f}")

    if args.only:
        return

    payload = {
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "git_sha": _git_sha(),
        "agent_model": args.agent_model,
        "judge_model": config.JUDGE_MODEL if args.judge else None,
        "total_eval_cost_usd": budget["spent"],
        "headline": headline,
        "retrieval": retrieval,
        "by_category": by_category(results),
        "tool_confusion": tool_confusion(results),
        "model_comparison": comparison if args.compare_models else [],
        "cases": results,
    }
    _write_results(payload)


if __name__ == "__main__":
    main()
