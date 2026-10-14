
# Insert Into Record Tool

## Overview

This tool is part of the Fair Fairness Benchmark (FFB) VIP team's evaluation-metrics workflow.

It provides a Python interface for updating, validating, saving, and loading evaluation metric records created by the team's `initialize_metrics_record_tool`.

The tool does not calculate evaluation metrics or train machine-learning models. It manages existing metric values.

## Features

- Update values of previously initialized metrics
- Validate metric names and numerical values
- Reject unknown metrics and invalid values
- Save metric records to JSON files
- Load and validate previously saved records
- Prevent accidental overwriting of existing JSON files
- Integrate with the team's existing metric record initializer

## Project Structure

```text
team/insert_into_record_tool/
├── __init__.py
├── insert_into_record.py
├── README.md
└── tests/
    └── test_insert_into_record.py
```

## Requirements

- Python 3.10 or newer
- Dependencies from the existing evaluation-metrics tool

Install dependencies from the repository root:

```bash
python -m pip install -r team/eval_metrics_tool/requirements.txt
```

## Usage

All examples should be executed from the repository root.

### 1. Create a metrics record

```python
from team.insert_into_record_tool import MetricsRecord

record = MetricsRecord({
    "Accuracy": 92.0,
    "F1": None,
    "Demographic Parity": 10.5
})
```

The record contains metric names as keys and numerical values as values.

`None` indicates that a metric has not yet been assigned a reliable value.

### 2. Update an existing metric

```python
record.insert_into_record("F1", 0.87)

print(record.record)
```

Expected output:

```python
{
    "Accuracy": 92.0,
    "F1": 0.87,
    "Demographic Parity": 10.5
}
```

Updates modify the record held in memory.

### 3. Save the record to JSON

```python
record.save_json("metrics.json")
```

This creates a JSON file containing the current metric values.

By default, the tool does not overwrite existing files.

To intentionally overwrite an existing file:

```python
record.save_json("metrics.json", overwrite=True)
```

### 4. Load an existing record

```python
loaded = MetricsRecord.load_json("metrics.json")

print(loaded.record)
```

The loaded record is validated before it can be used.

### 5. Integration with the existing initializer

The tool works with records returned by Team 2's `initialize_metrics_record_tool`.

```python
from team.initialize_metrics_record_tool import initialize_metrics_record
from team.insert_into_record_tool import MetricsRecord

initialized = initialize_metrics_record(
    "paper.pdf",
    ["Accuracy", "F1"]
)

record = MetricsRecord(initialized)

record.insert_into_record("F1", 0.90)

record.save_json("updated_metrics.json")
```

This demonstrates how an initialized paper-metrics record can be updated and saved.

## Validation and Error Handling

The tool checks metric names and numerical values before updating the record.

| Situation | Behavior |
| --- | --- |
| Existing metric with valid numeric value | Updates the record |
| Unknown metric name | Raises `KeyError` |
| Empty or whitespace-only metric name | Raises `ValueError` |
| Non-numeric value | Raises `TypeError` |
| Boolean value | Raises `TypeError` |
| NaN or infinity | Raises `ValueError` |
| Missing JSON file | Raises `FileNotFoundError` |
| Malformed JSON file | Raises `JSONDecodeError` |
| Invalid values in loaded record | Raises `TypeError` or `ValueError` |
| Saving to an existing file without overwrite permission | Raises `FileExistsError` |

Only initialized metrics can be updated. The tool does not automatically add new metric names.

## Testing

Run the automated tests from the repository root:

```bash
python -m unittest discover -s team/insert_into_record_tool/tests -v
```

The current test suite contains 14 tests covering:

- Updating existing metrics
- Rejecting unknown metrics
- Rejecting invalid numerical values
- Preserving unrelated metric values
- Saving and loading JSON records
- Handling missing and malformed JSON files
- Preventing accidental overwrites
- Integration with the existing metric initializer

All 14 tests passed during local development.

## Current Limitations

- Only existing metric names can be updated.
- Updates are stored in memory until `save_json()` is called.
- The tool does not independently verify whether a supplied metric value is correct.
- JSON storage is file-based and does not provide concurrent-write protection.
- The tool is not yet connected to the team's LangChain agent.

## Remaining Work

The core Python implementation is complete.

Remaining work includes:

1. Creating a LangChain-compatible tool interface for `insert_into_record(metric, value)`.
2. Implementing a mechanism for selecting and managing the active metric record.
3. Connecting the tool to the existing Ollama/Qwen agent.
4. Adding end-to-end tests for agent-initiated updates and persistence.
5. Reviewing integration behavior and resolving any issues before merging.

## Team Workflow

This tool builds on the existing `insert_eval_metric()` helper in `initialize_metrics_record_tool`.

The helper performs dictionary updates, while `MetricsRecord` adds validation, record management, JSON persistence, and a tested interface for future agent integration.
