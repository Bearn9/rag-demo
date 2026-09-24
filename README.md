# Bearn Agent Demo

Live demo: https://bearn-agent-demo.streamlit.app (no API key or setup needed)

This is an internal assistant for Bearn Analytics, a made-up software company. You can ask it about the company's HR policies, IT setup, security rules, product pricing, expense policy or quarterly financials, and it answers from the company's documents and data, citing where each fact came from.

It is built as an agent rather than a fixed pipeline. Claude is given a small set of tools and decides for itself, question by question, which ones to use. A policy question leads to a document search. A question about profit growth leads to an exact calculation over the financial data. "What's 15% of the hotel cap?" leads to a search followed by the calculator. "Hi" needs no tools at all. The app shows each of these decisions as it happens, so you can see what the model did and why the answer says what it says.

All company data in this repo is fictional.

## Trying it

A few questions that show different behaviour:

- *How many PTO days do new employees get?* One search, answered from the PTO policy.
- *How much did net profit grow from Q1 to Q2 2025, in percent?* An exact calculation over the quarterly figures (14.29%).
- *What's 15% of the hotel spend cap in high-cost cities?* A search for the cap ($400), then the calculator ($60).
- *What was revenue in Q1 2025?* followed by *And in Q2 2025?* The follow-up is understood from the conversation.
- *How do you decide which tool to use?* The agent looks up its own documentation and explains.
- *What is the CEO's salary?* It says the information isn't available instead of guessing.

Under each answer there is a trace listing every step: the tool that was called and with what input, what came back (including the retrieved text and similarity scores), and the time, tokens and cost of each model call.

## How it works

```
            ┌──────────────────────────────────────────────────────────┐
question ──▶│ agent: Claude + 4 tool definitions (tool_choice: auto)   │──▶ cited answer
+ recent    └──────────────┬───────────────────────────────▲───────────┘     + trace
  turns                    │ stop_reason == "tool_use"     │ all tool results
                           ▼                               │ in one message
            ┌──────────────────────────────────────────────┴───────────┐
            │ tools                                                    │
            │  search_knowledge_base   FAISS + sentence-transformers   │
            │  query_financials        exact maths over the CSV        │
            │  calculator              AST-based, never eval()         │
            │  explain_harness         docs/harness.md + live config   │
            └──────────────────────────────────────────────────────────┘
```

The core is a two-node LangGraph graph ([`src/ragbot/graph.py`](src/ragbot/graph.py)). The `agent` node sends the conversation and the tool definitions to Claude. If Claude responds with tool calls, the `tools` node runs them and sends all the results back in one message, and the loop continues. When Claude replies with plain text, that text is the answer. There's a cap on the number of model calls per question; on the last one the tools are withheld, so the loop always ends with an answer. The last few turns of the conversation are passed along with each new question, which is what makes follow-ups work.

There is no separate step that classifies the question first. The choice of tools *is* the routing, and it's visible in the trace.

### The tools

All four are defined in [`src/ragbot/tools.py`](src/ragbot/tools.py) with strict JSON schemas. If a tool fails (say, a quarter that doesn't exist), the error goes back to Claude as a tool result, and it can correct itself or explain the problem.

- `search_knowledge_base(query)` is semantic search over the 11 company documents. It returns the top four chunks with their source file, section and similarity score.
- `query_financials(metrics, quarters, operation)` works directly on the quarterly financials CSV. It returns raw values, sums, averages, minimums, maximums, changes and percentage changes. Language models make arithmetic slips when reading numbers out of text, so anything numeric about the company's performance goes through code instead. The inputs are fixed lists of metrics and operations; nothing is executed.
- `calculator(expression)` handles any other arithmetic, such as a percentage of a policy limit. It parses the expression with Python's `ast` module and only allows numbers, the usual operators and a few functions (`round`, `min`, `max`, `abs`, `sqrt`). It never calls `eval()`, and exponents are capped.
- `explain_harness(topic)` returns a section of [`docs/harness.md`](docs/harness.md), a short write-up of how the assistant works, along with the live configuration (model, top-k, limits) and the latest evaluation headline. This lets the assistant answer questions about itself accurately. The file lives outside the knowledge base, so it never shows up in normal document searches.

### Retrieval

The documents in `data/knowledge_base/` are split on their `##` headings, and long sections are split further into roughly 500-character pieces with some overlap. The financials CSV becomes one short plain-English chunk per quarter. The chunks are embedded locally with `all-MiniLM-L6-v2` from sentence-transformers and stored in a FAISS index using cosine similarity. Building the index is a separate command (`python -m ragbot.ingest`). The finished index is committed to the repo, and the app only ever reads it. Search therefore needs no API key and costs nothing.

### Model calls

Claude is called directly through the `anthropic` Python SDK ([`src/ragbot/llm.py`](src/ragbot/llm.py)); LangGraph only supplies the graph. The agent runs on `claude-sonnet-5` with adaptive thinking at low effort, which keeps answers quick. The tool definitions and system prompt are the same on every call, so they're marked for prompt caching. A question that takes three or four model calls pays full price for that shared prefix only once. A typical question costs about a cent and takes around four seconds.

## The hosted demo

The public version runs on Streamlit Community Cloud using my own API key, so visitors don't need one. To keep costs predictable, it has a few limits:

- 15 questions per browser session, and questions are capped at 500 characters;
- a shared daily budget (by default $2), tracked from the costs the trace already computes; after that the demo pauses until the next day;
- a maximum number of model calls per question;
- a monthly spend limit on the API key itself, set in the Anthropic Console, as the final backstop.

The Evaluation tab in the app only reads results that are committed to the repo. Visiting it never calls the API, so nobody can trigger an expensive evaluation run from the demo.

## Evaluation

The evaluation lives in [`scripts/run_eval.py`](scripts/run_eval.py) and runs against a hand-written set of 56 test questions in [`evals/dataset.jsonl`](evals/dataset.jsonl). The questions fall into eight groups:
- small talk;
- questions about the assistant itself;
- calculations;
- single-document facts;
- questions about the financial data;
- questions that need two sources;
- two-turn conversations;
- questions the assistant should decline because the answer isn't in its data.

Each test question records what a correct response looks like, using whichever of these apply:
- which tools should be used;
- which source file should be cited;
- words the answer must contain;
- a number it must include, within a tolerance;
- whether it should decline.

A question passes only if every check that applies to it passes. On top of these checks, a second model (`claude-opus-5`) acts as a judge. It reads the question, the tool results the agent actually received and a short reference answer, then scores the response from 1 to 5 for faithfulness (is every claim backed by the tool results?) and correctness.

Retrieval is also measured on its own, without any model calls: for each question, does the right document appear in the top k search results (hit@k and recall@k), and how high does it rank (MRR)?

Results from the latest run (2026-09-24, 56 questions, $1.15 to run):

| | |
|---|---|
| Questions passing every check | 56 / 56 |
| Correct tool choice | 100% |
| Cited sources that were actually retrieved | 100% |
| Out-of-scope questions correctly declined | 6 / 6 |
| Judge: faithfulness / correctness | 4.91 / 4.96 out of 5 |
| Retrieval: right document in top 4 / MRR | 100% / 0.986 |
| Response time, median / 95th percentile | 4.1 s / 7.1 s |
| Cost per question | $0.0092 |

The full breakdown is in [`evals/results/REPORT.md`](evals/results/REPORT.md) and in the app's Evaluation tab.

A perfect pass rate on a test set I wrote myself shouldn't be read as "the agent is perfect". The set is small, and the check that detects a polite refusal is a pattern match that I widened during development after seeing correct declines it didn't recognise. The judge scores are the more interesting signal. Every point it docked was for a claim that went beyond what the tools had returned. For example, the agent said the company is privately held while declining a stock-price question, without having looked anything up. This is exactly the behaviour the prompt tries to prevent, and it's the next thing to tighten.

To run the evaluation yourself:

```bash
# Retrieval metrics only: free, no API key (CI runs this and fails below 90% hit@4)
python scripts/run_eval.py --retrieval-only --min-hit-at-4 0.9

# Full run with the judge: about $1-2, writes evals/results/latest.json and REPORT.md
python scripts/run_eval.py --judge

# Compare another agent model, with a hard spending cap
python scripts/run_eval.py --judge --compare-models claude-haiku-4-5 --max-cost 5

# Try a few questions without overwriting the saved results
python scripts/run_eval.py --only fin-profit-growth oos-ceo-salary

# Re-apply changed checks to the saved answers (no API calls)
python scripts/run_eval.py --rescore
```

## Running it locally

You'll need Python 3.10+ and an Anthropic API key.

```bash
git clone https://github.com/Bearn9/rag-demo.git
cd rag-demo
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env             # Windows: Copy-Item .env.example .env
# add your ANTHROPIC_API_KEY to .env
streamlit run app.py
```

On Windows, if PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. `faiss-cpu` needs 64-bit Python.

The search index is already built. If you change anything in `data/knowledge_base/`, rebuild it with `python -m ragbot.ingest`.

## Deploying your own copy

1. Fork the repo and create a new app at [share.streamlit.io](https://share.streamlit.io), pointing it at `app.py` on `main`.
2. Under Advanced settings, pick Python 3.11 and add your key as a secret: `ANTHROPIC_API_KEY = "sk-ant-..."`. Streamlit exposes top-level secrets as environment variables, so no code changes are needed. The other settings in `.env.example` can be added the same way.
3. Set a monthly spend limit for the key in the Anthropic Console.

`requirements.txt` installs the CPU-only build of PyTorch, which keeps the deployment small. Free Streamlit apps go to sleep when unused, so the first visit after a while can take half a minute.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/test_tools.py tests/test_evaluation.py tests/test_retriever.py   # no API key needed
pytest tests/test_graph.py                                                    # runs the real agent; needs a key
```

The first set covers the tools (including attempts to sneak code into the calculator), the evaluation metrics and retrieval. `test_graph.py` runs a sample of the evaluation questions through the real agent and is skipped when no key is set. GitHub Actions rebuilds the index, runs the retrieval check and runs the tests on every push.

## Configuration

Everything is set through environment variables (see `.env.example`):

| Variable | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | | required |
| `AGENT_MODEL` | `claude-sonnet-5` | model that runs the agent |
| `AGENT_EFFORT` | `low` | thinking effort, `low` to `max` |
| `MAX_AGENT_STEPS` | `5` | model calls allowed per question |
| `JUDGE_MODEL` | `claude-opus-5` | evaluation judge |
| `MAX_QUESTIONS_PER_SESSION` | `15` | demo limit per visitor |
| `DAILY_BUDGET_USD` | `2.0` | demo spending limit per day |

## Project layout

```
app.py                  Streamlit app: chat with trace, evaluation results, how it works
src/ragbot/
  graph.py              the agent loop
  tools.py              the four tools
  llm.py                calls to Claude (agent step and eval judge)
  prompts.py            system prompts
  retriever.py          search over the FAISS index
  ingest.py             builds the index from data/knowledge_base/
  evaluation.py         evaluation metrics
  config.py             settings, prices, limits
data/knowledge_base/    the 11 fictional company documents
docs/harness.md         how the assistant works (read by explain_harness)
vector_store/           the prebuilt search index
evals/                  test questions and the latest results
scripts/run_eval.py     runs the evaluation
tests/
```

## License

MIT, see [LICENSE](LICENSE).
