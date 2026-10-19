# eval_metrics

Extracts the most important evaluation metrics a research paper PDF reports.
The tool pulls out the paper's Experiments section and sends only that to an
LLM. Any provider works through [LiteLLM](https://docs.litellm.ai/docs/providers),
including local models via Ollama. No API key ships with this project; bring
your own, or use a local model. If no model is set or the call fails, it falls
back to a deterministic, rule-based dictionary ranking.

## Install

```bash
<<<<<<< HEAD:team/eval_metrics_tool/README.md
pip install -r requirements.txt        # runtime: pdfplumber (MIT)
pip install -r requirements-dev.txt    # optional test dependencies
=======
pip install -r requirements.txt        # runtime: pdfplumber (MIT), litellm (MIT)
pip install -r requirements-dev.txt    # adds pytest, reportlab for tests
>>>>>>> 7ab761c (updated eval metric to use an llm on top of the pre-existing dictionary method | eval metrics now finds the experiments section and sends that to an llm of choice | updated readme):eval_metrics_tool/README.md
```

## Choosing a model

`model` / `--model` is a LiteLLM model string. It defaults to the
`EVAL_METRICS_MODEL` environment variable. LiteLLM reads keys from each
provider's usual environment variable.

| Provider | Model string | Setup |
|---|---|---|
| Ollama (local) | `ollama/qwen3:4b` | `ollama pull qwen3:4b`; no key. Remote host: `--api-base http://host:11434` |
| OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-haiku-5-5` | `ANTHROPIC_API_KEY` |
| Any OpenAI-compatible server (LM Studio, vLLM, ...) | `openai/<model>` | `--api-base http://host:port/v1` |

`--api-base` also defaults to `EVAL_METRICS_API_BASE`. You can pass `--api-key`,
but environment variables keep keys out of your shell history.

## Usage

```python
from eval_metrics import extract_eval_metrics
extract_eval_metrics("paper.pdf", top_k=5, model="ollama/qwen3:4b")  # ["F1 score", "Accuracy"]
extract_eval_metrics("paper.pdf", use_llm=False)                     # dictionary only
```

Returns metric names, most important first, or `[]` if none are found. LLM
results use the paper's own wording (they are not normalized). The dictionary
fallback returns canonical names. When the tool falls back, it issues a
`UserWarning` that says why. Raises `FileNotFoundError` for a missing file and
`eval_metrics.PDFReadError` for a file that is not a readable PDF.

```bash
python -m eval_metrics paper.pdf --model ollama/qwen3:4b   # one metric per line
python -m eval_metrics paper.pdf --show-section            # print the Experiments section sent to the LLM
python -m eval_metrics paper.pdf --no-llm --verbose        # dictionary: + score and where found
```

Run both from `team/eval_metrics_tool/`.

## How it works

LLM path:

1. pdfplumber extraction and section splitting (below).
2. Keep only the Experiments section: the text under headings that
   `sections.py` classifies as Experiments / Results / Evaluation (several such
   headings are joined). Nothing else is sent. If there is no such heading,
   the dictionary fallback is used.
3. `llm.py` asks the model for a JSON array of the metrics the paper reports
   (temperature 0) and parses it, ignoring `<think>` blocks from reasoning models.

Dictionary fallback:

1. `extraction.py` reads page text and tables with pdfplumber
   (`page.extract_text()`, `page.extract_tables()`), splits the text into
   sections by heading heuristics (`sections.py`: Abstract, Introduction,
   Related Work, Method, Experiments/Results/Evaluation, Conclusion,
   References), and tags each table with the section its page mostly belongs to.
   Unrecognised text falls into an "other" section.
2. `matcher.py` finds metric mentions using `metrics_dictionary.py`. Matches use
   word boundaries; when patterns overlap the longest wins ("Top-1 accuracy" is
   not also "Accuracy"). Short ambiguous acronyms (EM, IS, R2, AP, ...) are
   case-sensitive and count in running text only when a number follows
   ("EM = 85.2"), or anywhere in a table.
3. `scoring.py` sums weights per canonical metric, sorts by score and breaks
   ties alphabetically. Weights live in the `WEIGHTS` dict at the top of that
   file (heuristic, meant to be tuned):

   | Where | Weight |
   |---|---|
   | Table header / column (or row) label | 5 |
   | Table cell, caption ("Table 1: ...") | 3 |
   | Abstract | 3 |
   | Experiments / Results / Evaluation | 2 |
   | Anywhere else | 1 |
   | References, Related Work (text and tables) | 0 |

Extraction and scoring are separate: `rank_metrics(PaperContent(...))` and
`experiments_section(PaperContent(...))` work on plain strings and lists, so most
tests need no PDFs. LLM tests use a fake `litellm` module (`tests/conftest.py`),
so they need no network or key.

## Adding a metric

Append a `MetricSpec` to `METRICS` in `eval_metrics/metrics_dictionary.py`.
Patterns are regexes without word boundaries (added automatically):

- `patterns`: case-insensitive, for full names ("mean squared error").
- `cs_patterns`: case-sensitive, for distinctive acronyms ("BLEU").
- `context_patterns`: case-sensitive and context-gated, for short or ambiguous acronyms.

When adding tests, include an alias case for the new pattern. A test suite is
not yet committed in this repository.

## Tests

No automated tests are currently committed. The development requirements
include `pytest` and `reportlab` for future tests.

## Known limitations

- The LLM path is only as good as the model. Small local models may miss
  metrics or list ones from other papers. Use `--show-section` to check what it saw.
- The LLM sees only the Experiments section. Metrics reported only in the
  abstract, or in a section with an unrecognised heading, are missed.
- The whole section is sent with no length limit. Long sections can exceed a
  small local model's context window (Ollama defaults to a few thousand tokens).

- Scanned / image-only PDFs are not supported (no OCR); they yield `[]`.
- Two-column layouts may scramble text order, which can break heading and caption detection.
- Only ruled tables are detected (pdfplumber's default). Borderless
  "booktabs"-style tables are missed as tables, though their text still counts as prose.
- Table text also appears in the page text, so table mentions are partly counted twice.
- Metrics referred to only by symbols or nonstandard names are missed.
- Words like "precision" or "recall" in a non-metric sense can still slip through despite basic filters.
- Section detection is heading-regex based and imperfect; a missed "Related Work" heading means its metrics count.
