"""The agent loop as a LangGraph StateGraph.

    START -> agent -> (conditional) -> tools -> agent -> ... -> END

`agent` calls Claude with the conversation and the tool definitions. If Claude asks
for tools, `tools` runs every requested call and appends all of the results in one
message, then control returns to `agent`. When Claude answers without asking for a
tool (or the step cap is reached), the graph ends. The model's choice of tools *is*
the routing decision; there is no separate classifier.

Each node appends trace records to `steps`, which the UI renders and the eval scores.
"""

import json
import operator
import time
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from ragbot import config, llm, tools


class AgentState(TypedDict, total=False):
    messages: list[dict]  # Anthropic message list: prior turns + this turn's loop
    model: str | None  # overrides config.AGENT_MODEL (used by the eval)
    agent_calls: int
    last_stop_reason: str | None
    steps: Annotated[list[dict], operator.add]
    sources: Annotated[list[str], operator.add]
    answer: str | None


def agent(state: AgentState) -> dict:
    calls = state.get("agent_calls", 0) + 1
    # On the last allowed call, withhold tools so the model must write an answer.
    allow_tools = calls < config.MAX_AGENT_STEPS
    result = llm.agent_step(state["messages"], model=state.get("model"), allow_tools=allow_tools)

    tool_names = [call["name"] for call in result["tool_calls"]]
    used_tools_before = any(step["kind"] == "tool" for step in state.get("steps", []))
    if tool_names:
        label = "Plan: call " + ", ".join(tool_names)
    elif used_tools_before:
        label = "Generate answer from tool results"
    else:
        label = "Answer directly (no tools needed)"

    step = {
        "kind": "model",
        "label": label,
        "model": result["model"],
        "thinking": result["thinking"],
        "tool_calls": result["tool_calls"],
        "latency_ms": result["latency_ms"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "cost_usd": result["cost_usd"],
    }
    update = {
        "messages": state["messages"] + [{"role": "assistant", "content": result["content"]}],
        "agent_calls": calls,
        "last_stop_reason": result["stop_reason"],
        "steps": [step],
    }
    if result["stop_reason"] != "tool_use":
        update["answer"] = result["text"] or "Sorry, I couldn't produce an answer."
    return update


def run_tools(state: AgentState) -> dict:
    last_content = state["messages"][-1]["content"]
    tool_uses = [block for block in last_content if block.get("type") == "tool_use"]

    results, steps, sources = [], [], []
    for block in tool_uses:
        started = time.perf_counter()
        output, is_error = tools.run_tool(block["name"], block["input"])
        latency_ms = (time.perf_counter() - started) * 1000

        step_sources = [] if is_error else tools.sources_from_result(block["name"], output)
        sources.extend(step_sources)
        steps.append(
            {
                "kind": "tool",
                "name": block["name"],
                "input": block["input"],
                "output": output,
                "is_error": is_error,
                "sources": step_sources,
                "latency_ms": latency_ms,
            }
        )
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": json.dumps(output, ensure_ascii=False),
                "is_error": is_error,
            }
        )

    # All results go back in a single user message.
    return {
        "messages": state["messages"] + [{"role": "user", "content": results}],
        "steps": steps,
        "sources": sources,
    }


def _next(state: AgentState) -> Literal["tools", "__end__"]:
    return "tools" if state.get("last_stop_reason") == "tool_use" and not state.get("answer") else END


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _next, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


_compiled_graph = None


def get_graph():
    """Lazily build and cache the compiled graph (module-level singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def initial_state(query: str, history: list[dict] | None = None, model: str | None = None) -> AgentState:
    """Build the graph input: the last few prior turns (text only) plus the new question.

    `history` is a list of {"role": "user"|"assistant", "content": str}.
    """
    prior = (history or [])[-2 * config.MEMORY_TURNS :]
    # The API requires the conversation to start with a user turn.
    while prior and prior[0]["role"] != "user":
        prior = prior[1:]
    messages = [{"role": m["role"], "content": m["content"]} for m in prior]
    messages.append({"role": "user", "content": query})
    state: AgentState = {"messages": messages, "agent_calls": 0, "steps": [], "sources": [], "answer": None}
    if model:
        state["model"] = model
    return state


def run(query: str, history: list[dict] | None = None, model: str | None = None) -> AgentState:
    """Convenience wrapper: run the full loop and return the final state."""
    return get_graph().invoke(initial_state(query, history, model))


def turn_totals(steps: list[dict]) -> dict:
    """Aggregate a turn's trace: step count, wall time, tokens and cost."""
    model_steps = [s for s in steps if s["kind"] == "model"]
    return {
        "steps": len(steps),
        "latency_ms": sum(s["latency_ms"] for s in steps),
        "input_tokens": sum(s["input_tokens"] for s in model_steps),
        "output_tokens": sum(s["output_tokens"] for s in model_steps),
        "cost_usd": sum(s["cost_usd"] for s in model_steps),
        "tools_used": [s["name"] for s in steps if s["kind"] == "tool"],
    }
