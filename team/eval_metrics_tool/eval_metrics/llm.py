"""Ask an LLM which metrics a paper's Experiments section reports (any LiteLLM provider).

The model is a LiteLLM model string, e.g. "ollama/qwen3:4b", "gpt-4o-mini",
"anthropic/claude-haiku-5-5", "openrouter/...". API keys are read by LiteLLM
from the provider's usual environment variable (OPENAI_API_KEY,
ANTHROPIC_API_KEY, ...); local models such as Ollama need none.
"""

import json
import os
import re

MODEL_ENV = "EVAL_METRICS_MODEL"
API_BASE_ENV = "EVAL_METRICS_API_BASE"

SYSTEM_PROMPT = (
    "You read the Experiments section of an AI/ML research paper and identify "
    "the evaluation metrics the paper itself uses to report its results (e.g. "
    "accuracy, F1, BLEU, demographic parity difference). Ignore metrics that "
    "are only mentioned when discussing other people's work. Reply with only a "
    "JSON array of metric names as strings, most important first, using the "
    "name the paper uses. Reply [] if the section reports no metrics."
)


class LLMError(Exception):
    """The LLM could not be reached or gave an unusable answer."""


def resolve_model(model: str | None) -> str | None:
    return model or os.environ.get(MODEL_ENV) or None


def _parse(reply: str) -> list[str]:
    # Reasoning models (qwen3, deepseek-r1) may wrap thoughts in <think> tags.
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    match = re.search(r"\[.*\]", reply, re.S)
    if not match:
        raise LLMError(f"no JSON array in LLM reply: {reply[:200]!r}")
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise LLMError(f"invalid JSON in LLM reply: {match.group()[:200]!r}") from exc
    names, seen = [], set()
    for item in items:
        if not isinstance(item, str) or not item.strip():
            continue
        name = " ".join(item.split())
        if name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)
    return names


def ask_llm(section: str, model: str, top_k: int = 5, api_base: str | None = None,
            api_key: str | None = None, timeout: float = 120) -> list[str]:
    """Metric names the LLM finds in ``section``, most important first (<= top_k)."""
    try:
        import litellm  # imported lazily: slow to import, unused by --no-llm
    except ImportError as exc:
        raise LLMError("litellm is not installed: pip install -r requirements.txt") from exc
    litellm.suppress_debug_info = True  # no "Give Feedback" banners on errors
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"List at most {top_k} metrics.\n\nExperiments section:\n{section}"},
    ]
    try:
        resp = litellm.completion(
            model=model, messages=messages, temperature=0, timeout=timeout,
            api_base=api_base or os.environ.get(API_BASE_ENV), api_key=api_key,
        )
        reply = resp.choices[0].message.content or ""
    except Exception as exc:  # litellm raises many provider-specific types
        raise LLMError(f"LLM call to {model!r} failed: {exc}") from exc
    return _parse(reply)[:top_k]
