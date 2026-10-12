"""Print the initialized metric record as JSON."""

import argparse
import json

from . import initialize_metrics_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Read reported evaluation values from a paper")
    parser.add_argument("paper", help="PDF or UTF-8 text paper")
    parser.add_argument("--metric", action="append", dest="metrics",
                        help="metric name to initialize; may be repeated")
    parser.add_argument("--top-k", type=int, default=5,
                        help="number of metrics to discover when --metric is absent")
    parser.add_argument("--row-label", help="exact table row label, e.g. 'Ours'")
    args = parser.parse_args()
    record = initialize_metrics_record(
        args.paper, args.metrics, top_k=args.top_k, row_label=args.row_label
    )
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
