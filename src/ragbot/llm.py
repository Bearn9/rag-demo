"""Thin wrappers around the Anthropic SDK for the two calls the graph needs:
a forced-tool-use routing classification, and free-text answer generation.
"""

from functools import lru_cache

import anthropic

from ragbot import config, prompts


@lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def classify_query(query: str) -> tuple[bool, str]:
    """Ask the router model whether `query` needs a knowledge-base lookup.

    Returns (needs_retrieval, reasoning).
    """
    response = _client().messages.create(
        model=config.ROUTER_MODEL,
        max_tokens=256,
        system=prompts.ROUTER_SYSTEM_PROMPT,
        tools=[prompts.ROUTER_TOOL],
        tool_choice={"type": "tool", "name": "classify_query"},
        messages=[{"role": "user", "content": query}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "classify_query":
            return bool(block.input["needs_retrieval"]), str(block.input["reasoning"])
    raise RuntimeError("Router model did not return a classify_query tool call.")


def generate_with_context(query: str, chunks: list[dict]) -> str:
    """Generate an answer grounded in retrieved knowledge-base chunks."""
    system = prompts.GENERATION_WITH_CONTEXT_SYSTEM_PROMPT.format(
        context=prompts.format_context(chunks)
    )
    response = _client().messages.create(
        model=config.GENERATION_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": query}],
    )
    return _extract_text(response)


def generate_direct(query: str) -> str:
    """Generate a conversational answer with no retrieval context."""
    response = _client().messages.create(
        model=config.GENERATION_MODEL,
        max_tokens=512,
        system=prompts.GENERATION_DIRECT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": query}],
    )
    return _extract_text(response)


def _extract_text(response: anthropic.types.Message) -> str:
    return "".join(block.text for block in response.content if block.type == "text").strip()
