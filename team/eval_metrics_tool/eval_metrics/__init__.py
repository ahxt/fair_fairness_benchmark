"""Extract the main evaluation metrics from a paper PDF.

By default the paper's Experiments section is sent to an LLM (any LiteLLM
provider, including local Ollama). Without a model, or if the call fails, the
rule-based dictionary ranking is used instead.
"""

import warnings

from .extraction import PaperContent, PDFReadError, extract_paper
from .llm import LLMError, ask_llm, resolve_model
from .scoring import MetricScore, rank_metrics
from .sections import RESULTS

__all__ = ["extract_eval_metrics", "rank_pdf", "PDFReadError", "LLMError", "MetricScore"]


def rank_pdf(pdf_path: str) -> list[MetricScore]:
    """Every detected metric with its score and evidence, best first (dictionary only)."""
    return rank_metrics(extract_paper(pdf_path))


def experiments_section(content: PaperContent) -> str:
    """Text of the Experiments section ("" if no such heading was found).

    Headings like "Experiments", "Results", "Evaluation" all count; if the paper
    has several (e.g. "5 Experiments" then "5.2 Results"), they are joined.
    """
    return "\n".join(s.text for s in content.sections if s.kind == RESULTS).strip()


def llm_metrics(content: PaperContent, top_k: int, model: str | None,
                api_base: str | None = None,
                api_key: str | None = None) -> tuple[list[str] | None, str | None]:
    """(names, None) from the LLM, or (None, reason) when the dictionary should be used."""
    model = resolve_model(model)
    if not model:
        return None, ("no LLM model set (pass model= / --model or set "
                      "EVAL_METRICS_MODEL); using dictionary ranking")
    section = experiments_section(content)
    if not section:
        return None, "no Experiments section found; using dictionary ranking"
    try:
        return ask_llm(section, model, top_k, api_base=api_base, api_key=api_key), None
    except LLMError as exc:
        return None, f"{exc}; using dictionary ranking"


def extract_eval_metrics(pdf_path: str, top_k: int = 5, model: str | None = None,
                         api_base: str | None = None, api_key: str | None = None,
                         use_llm: bool = True) -> list[str]:
    """Names of the paper's most important metrics, most important first.

    With an LLM, names are as the LLM reports them; with the dictionary
    fallback (``use_llm=False``, no model, or a failed call) they are canonical
    dictionary names. ``model`` is a LiteLLM model string and defaults to the
    EVAL_METRICS_MODEL environment variable.

    Returns [] if none are found. Raises FileNotFoundError if the file is
    missing and PDFReadError if it is not a readable PDF.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    content = extract_paper(pdf_path)
    if use_llm:
        names, reason = llm_metrics(content, top_k, model, api_base, api_key)
        if names is not None:
            return names
        warnings.warn(reason, stacklevel=2)
    return [s.name for s in rank_metrics(content)[:top_k]]
