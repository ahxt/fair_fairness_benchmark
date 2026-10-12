# Initialize metrics record

Create a record of evaluation metrics and initialize it from values reported in
a paper. This tool uses the team's `eval_metrics_tool` to recognize metric names,
then ranks nearby numbers in text and tables. It returns a Python dictionary
such as `{"F1": 0.87, "Accuracy": 92.0}`. Missing or ambiguous values remain
`None` for later review. It does not run the paper's experiments.

## Setup

From the repository root, use Python 3.10 or newer:

```bash
python -m pip install -r team/eval_metrics_tool/requirements.txt
```

## Python API

```python
from team.eval_metrics_tool.eval_metrics import extract_eval_metrics
from team.initialize_metrics_record_tool import initialize_metrics_record, insert_eval_metric

names = extract_eval_metrics("paper.pdf", top_k=5)
record = initialize_metrics_record("paper.pdf", names)
insert_eval_metric(record, "F1", 0.87)  # fill or correct one entry
```

`initialize_metrics_record("paper.pdf", top_k=5)` discovers metric names itself.
It also accepts a UTF-8 `.txt` paper, aliases such as `"F1-score"`, and an
optional `row_label="Ours"` to restrict extraction to a table row. For the FFB
paper's Table 4, `row_label="Bank Age"` selects that dataset and sensitive
attribute, including when PDF extraction leaves the table as plain text.
The course-task spelling `Initialize_metrics_record()` and the name
`create_eval_metrics_record()` are aliases of the same function.

## Command line

```bash
python -m team.initialize_metrics_record_tool paper.pdf --top-k 5
python -m team.initialize_metrics_record_tool paper.pdf --metric F1 --metric Accuracy
python -m team.initialize_metrics_record_tool paper.pdf --metric F1 --row-label Ours
python -m team.initialize_metrics_record_tool paper.pdf --metric Accuracy --metric F1 --row-label "Bank Age"
```

The command prints JSON. Run it from the repository root.

## Selection rules and limits

For each metric, the tool gathers candidate values from every section of the
paper and ruled tables. It scores proximity to the metric name, direct
metric/value statements, result wording, references to the proposed method,
table alignment, and repeated independent mentions. Baselines, citations,
improvement amounts, and likely years or sample counts reduce a candidate's
score. The weights are in `records.py` and can be overridden with `weights`.

A value is assigned only if it reaches `min_score` and leads the next candidate
by `min_margin`. Percentages keep their printed scale (`92%` becomes `92.0`);
for `87.0 ± 1.2`, the central value is stored. These scores are heuristics,
not probabilities. The function returns one scalar per metric, so papers with
many datasets, models, or metric variants can remain ambiguous. The FFB paper
reports many values per metric across datasets and sensitive attributes; an
unqualified call may therefore return `None` rather than choose an arbitrary
one. Scanned PDFs have no OCR, and complex or scrambled PDF tables may need
manual review.

## Tests

The tests use the optional development dependencies (including ReportLab):

```bash
python -m pip install -r team/eval_metrics_tool/requirements-dev.txt
python -m unittest discover -s team/initialize_metrics_record_tool/tests -v
```
