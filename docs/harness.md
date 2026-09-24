# Bearn Agent Demo — how the harness works

This file is the agent's self-knowledge. The `explain_harness` tool returns one
section of it (plus live configuration values) when a user asks how the assistant
works. It is deliberately kept out of `data/knowledge_base/`, so it never appears in
knowledge-base search results or skews the retrieval evaluation.

## overview

The Bearn Agent Demo is an agentic retrieval-augmented generation (RAG) assistant for a
fictional company, Bearn Analytics. Instead of a fixed pipeline, a single Claude model
runs in a tool-use loop: for every message it decides whether to call a tool, which
tool, and with what arguments. It can chain several tool calls, read the results,
and then write a final answer that cites its sources. Greetings and small talk are
answered directly with no tool calls. Every step (tool name, inputs, result, latency,
tokens and cost) is shown in the UI's trace panel.

## architecture

The agent loop is a LangGraph StateGraph with two nodes:

- `agent`: calls Claude with the conversation so far and the tool definitions
  (`tool_choice: auto`). If Claude stops with `stop_reason == "tool_use"`, the graph
  routes to `tools`; otherwise the turn is finished and its text is the answer.
- `tools`: executes every tool call from Claude's last turn (they may run in
  parallel) and appends all of the results as one message, then loops back to
  `agent`.

The loop is capped at a fixed number of agent steps, so a confused model can't spin
forever. There is no separate routing classifier: the model's own
choice of tools *is* the routing decision, and the trace shows it explicitly.
Conversation memory comes from passing the last few user/assistant turns back in, so
follow-ups like "and what about Q3?" work.

## tools

How tools get chosen: there is no separate router or classifier. Claude sees the
four tool definitions (name, description, JSON schema) on every turn and decides
itself, in a tool-use loop, whether it needs a tool, which one, and with what
arguments. It can call several tools in sequence or in parallel before answering.
Greetings and small talk need no tools, so Claude answers them directly.

The agent has four tools, each with a strict JSON schema:

- `search_knowledge_base(query)`: semantic search over Bearn's internal documents
  (HR, IT, security, remote-work handbook, product FAQs, expense policy, annual
  report). Returns the top-k chunks with source file, section and similarity score.
- `query_financials(metrics, quarters, operation)`: exact computation over the
  quarterly financials CSV (values, sum, mean, min, max, change, pct_change). It
  exists because LLMs are unreliable at arithmetic over retrieved text; this tool
  makes numeric answers exact and auditable.
- `calculator(expression)`: a safe arithmetic evaluator for any other maths (for
  example a percentage of a policy limit). It parses the expression with Python's
  `ast` module and allows only numbers, arithmetic operators and a few functions. It
  never calls `eval()`.
- `explain_harness(topic)`: returns a section of this document plus live config
  values, so the assistant can describe its own design accurately instead of guessing.

Generation is the final step: once the tools have returned, Claude writes the answer
using only the tool results and cites each fact as [source: filename].

## models

All reasoning and generation uses a single Claude model (see the live configuration
below), called directly through the official `anthropic` Python SDK with no LangChain
model wrappers. Embeddings come from the local `all-MiniLM-L6-v2` sentence-transformers
model, which runs on CPU with no API key. The offline evaluation uses a stronger
Claude model as an LLM judge.

## retrieval

Documents in `data/knowledge_base/` are chunked by markdown `##` header (with long
sections split to about 500 characters with overlap), and the financial CSV becomes
one plain-English chunk per quarter. Chunks are embedded with sentence-transformers
and stored in a FAISS inner-product index over L2-normalised vectors (cosine
similarity). Ingestion is a separate, idempotent CLI (`python -m ragbot.ingest`); the
app only reads the committed index.

## evaluation

The evaluation is run offline by the project owner, never by demo visitors, and the
results are committed to the repo and shown in the app's Evaluation tab. The dataset
(`evals/dataset.jsonl`) covers chit-chat, questions about the harness, calculations,
single-document facts, finance computations, multi-hop questions, multi-turn
follow-ups, and out-of-scope questions the agent must decline. Metrics:

- retrieval: hit@k, MRR and recall@k (deterministic, no API key needed, run in CI);
- tool-selection accuracy with a confusion matrix;
- answer correctness (expected substrings and numbers within tolerance);
- citation precision and recall;
- refusal precision and recall on out-of-scope questions;
- LLM-as-judge faithfulness and correctness scores (1–5);
- latency p50/p95 and cost per question.

## guardrails

The public demo runs on the project owner's API key, so it has limits:

- a per-session question cap and a maximum question length;
- a shared daily spend budget, after which the demo pauses until the next day;
- a cap on agent steps per question;
- a monthly spend limit set in the Anthropic console as a hard backstop.

The agent is instructed to answer company questions only from tool results and to say
plainly when the knowledge base doesn't contain the answer, instead of guessing.

## tech_stack

Python, LangGraph (graph orchestration), the Anthropic Python SDK (Claude tool use),
sentence-transformers and FAISS (local embeddings and vector search), Streamlit (UI,
hosted on Streamlit Community Cloud), pytest and GitHub Actions (tests, and a
retrieval-quality gate in CI).
