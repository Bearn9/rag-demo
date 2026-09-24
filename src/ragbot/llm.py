"""Thin wrappers around the Anthropic SDK: one agent step of the tool-use loop, and the
offline eval's LLM judge.
"""

import json
import time
from functools import lru_cache

import anthropic

from ragbot import config, prompts, tools


@lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def _supports_adaptive_thinking(model: str) -> bool:
    # Haiku 4.5 predates adaptive thinking and the effort parameter.
    return "haiku" not in model


def agent_step(messages: list[dict], model: str | None = None, allow_tools: bool = True) -> dict:
    """Run one turn of the agent: Claude sees the conversation and either calls tools
    or writes the final answer.

    Returns a dict with the assistant `content` (as plain dicts, ready to append to
    `messages`), `stop_reason`, `text`, `thinking` (reasoning summary, may be empty),
    `tool_calls`, and usage/latency/cost for the trace.
    """
    model = model or config.AGENT_MODEL
    kwargs = {}
    if _supports_adaptive_thinking(model):
        kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        kwargs["output_config"] = {"effort": config.AGENT_EFFORT}

    started = time.perf_counter()
    response = _client().messages.create(
        model=model,
        max_tokens=4096,
        system=prompts.AGENT_SYSTEM_PROMPT,
        tools=tools.TOOL_DEFINITIONS,
        tool_choice={"type": "auto"} if allow_tools else {"type": "none"},
        messages=messages,
        # Caches the stable tools + system prefix; each loop iteration resends it.
        cache_control={"type": "ephemeral"},
        **kwargs,
    )
    latency_ms = (time.perf_counter() - started) * 1000

    usage = response.usage
    cache_read = usage.cache_read_input_tokens or 0
    cache_write = usage.cache_creation_input_tokens or 0

    if response.stop_reason == "refusal":
        text = "Sorry, I can't help with that request."
    else:
        text = _extract_text(response)

    return {
        "content": [block.to_dict() for block in response.content],
        "stop_reason": response.stop_reason,
        "text": text,
        "thinking": "\n".join(
            block.thinking for block in response.content if block.type == "thinking" and block.thinking
        ).strip(),
        "tool_calls": [
            {"id": block.id, "name": block.name, "input": block.input}
            for block in response.content
            if block.type == "tool_use"
        ],
        "model": model,
        "latency_ms": latency_ms,
        "input_tokens": usage.input_tokens + cache_read + cache_write,
        "output_tokens": usage.output_tokens,
        "cost_usd": config.cost_usd(model, usage.input_tokens, usage.output_tokens, cache_read, cache_write),
    }


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "integer"},
        "correctness": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["faithfulness", "correctness", "rationale"],
    "additionalProperties": False,
}


def judge_answer(question: str, tool_results: str, reference: str, answer: str) -> dict:
    """Grade one answer for faithfulness and correctness (1-5 each). Offline eval only."""
    user_content = (
        f"<assistant_instructions>\n{prompts.AGENT_SYSTEM_PROMPT}\n</assistant_instructions>\n\n"
        f"<question>\n{question}\n</question>\n\n"
        f"<tool_results>\n{tool_results or '(no tools were called)'}\n</tool_results>\n\n"
        f"<reference>\n{reference}\n</reference>\n\n"
        f"<answer>\n{answer}\n</answer>"
    )
    response = _client().messages.create(
        model=config.JUDGE_MODEL,
        max_tokens=2048,
        system=prompts.JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
    )
    verdict = json.loads(_extract_text(response))
    verdict["cost_usd"] = config.cost_usd(
        config.JUDGE_MODEL, response.usage.input_tokens, response.usage.output_tokens
    )
    return verdict


def _extract_text(response: anthropic.types.Message) -> str:
    return "".join(block.text for block in response.content if block.type == "text").strip()
