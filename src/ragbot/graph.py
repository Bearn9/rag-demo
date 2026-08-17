"""The agentic RAG graph: routes a query to either a knowledge-base lookup or a
direct answer, then generates a response.

    START -> route_query -> (conditional) -> retrieve | generate_direct
    retrieve -> generate_with_context -> END
    generate_direct -> END
"""

from typing import Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from ragbot import llm, retriever
from ragbot.retriever import Chunk


class GraphState(TypedDict):
    query: str
    route: Optional[Literal["retrieve", "direct"]]
    route_reasoning: Optional[str]
    retrieved_chunks: Optional[list[Chunk]]
    answer: Optional[str]


def route_query(state: GraphState) -> dict:
    needs_retrieval, reasoning = llm.classify_query(state["query"])
    return {
        "route": "retrieve" if needs_retrieval else "direct",
        "route_reasoning": reasoning,
    }


def retrieve(state: GraphState) -> dict:
    chunks = retriever.retrieve(state["query"])
    return {"retrieved_chunks": chunks}


def generate_with_context(state: GraphState) -> dict:
    answer = llm.generate_with_context(state["query"], state["retrieved_chunks"] or [])
    return {"answer": answer}


def generate_direct(state: GraphState) -> dict:
    answer = llm.generate_direct(state["query"])
    return {"answer": answer}


def _decide_route(state: GraphState) -> Literal["retrieve", "generate_direct"]:
    return "retrieve" if state["route"] == "retrieve" else "generate_direct"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("route_query", route_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_with_context", generate_with_context)
    graph.add_node("generate_direct", generate_direct)

    graph.add_edge(START, "route_query")
    graph.add_conditional_edges(
        "route_query",
        _decide_route,
        {"retrieve": "retrieve", "generate_direct": "generate_direct"},
    )
    graph.add_edge("retrieve", "generate_with_context")
    graph.add_edge("generate_with_context", END)
    graph.add_edge("generate_direct", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    """Lazily build and cache the compiled graph (module-level singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
