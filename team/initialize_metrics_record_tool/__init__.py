"""Initialize a record of evaluation metrics from a paper.

The public function accepts the metric names returned by the team's
``eval_metrics_tool`` or discovers them from the paper itself.
"""

from .records import VALUE_WEIGHTS, create_eval_metrics_record, insert_eval_metric

# The course task names this operation Initialize_metrics_record(). Keep that
# spelling available while using Python's conventional lowercase spelling too.
initialize_metrics_record = create_eval_metrics_record
Initialize_metrics_record = create_eval_metrics_record

__all__ = [
    "Initialize_metrics_record",
    "VALUE_WEIGHTS",
    "create_eval_metrics_record",
    "initialize_metrics_record",
    "insert_eval_metric",
]
