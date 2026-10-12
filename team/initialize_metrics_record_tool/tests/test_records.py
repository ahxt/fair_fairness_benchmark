"""Behavioral checks for initializing paper-reported metric values."""

import tempfile
import unittest
from pathlib import Path

from team.initialize_metrics_record_tool import (
    Initialize_metrics_record,
    initialize_metrics_record,
    insert_eval_metric,
)


class MetricsRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paper = Path(self.temp.name) / "paper.txt"

    def test_reported_values_and_aliases(self):
        self.paper.write_text(
            "Results\nOur proposed method achieved F1 = 0.87 and "
            "Accuracy = 92%.\n", encoding="utf-8"
        )
        record = initialize_metrics_record(
            str(self.paper), ["F1-score", "Accuracy", "BLEU"]
        )
        self.assertEqual(record, {"F1": 0.87, "Accuracy": 92.0, "BLEU": None})
        self.assertEqual(Initialize_metrics_record(str(self.paper), ["F1"]), {"F1": 0.87})

    def test_ours_beats_baseline_and_can_be_updated(self):
        self.paper.write_text(
            "Results\nThe baseline scored F1 = 0.80. "
            "Our proposed method achieved F1 = 0.87.\n", encoding="utf-8"
        )
        record = initialize_metrics_record(str(self.paper), ["F1"])
        self.assertEqual(record, {"F1": 0.87})
        self.assertIs(insert_eval_metric(record, "F1", 0.88), record)
        self.assertEqual(record["F1"], 0.88)
        with self.assertRaises(KeyError):
            insert_eval_metric(record, "Accuracy", 92)

    def test_competing_values_abstain(self):
        self.paper.write_text(
            "Results\nF1 = 0.81.\nF1 = 0.82.\n", encoding="utf-8"
        )
        self.assertEqual(initialize_metrics_record(str(self.paper), ["F1"]), {"F1": None})

    def test_pdf_table_exact_row(self):
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

        path = Path(self.temp.name) / "table.pdf"
        table = Table([
            ["Method", "F1", "Accuracy"],
            ["Baseline", "0.80", "89.0"],
            ["Ours", "0.87", "92.0"],
        ])
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))
        SimpleDocTemplate(str(path)).build([table])
        self.assertEqual(
            initialize_metrics_record(
                str(path), ["F1", "Accuracy"], row_label="Ours"
            ),
            {"F1": 0.87, "Accuracy": 92.0},
        )
        with self.assertRaises(ValueError):
            initialize_metrics_record(str(path), ["F1"], row_label="Missing")

    def test_unruled_table_row_with_uncertainty(self):
        self.paper.write_text(
            "Table 4: Results for our benchmark.\n"
            "acc auc ap f1 dp abcc prule eodd eopp\n"
            "Bank Age 91.17±0.57 94.05±0.19 62.41±1.83 54.31±11.14 "
            "10.88±4.27 10.64±1.63 44.48±5.95 10.71±5.68 6.16±4.90\n",
            encoding="utf-8",
        )
        self.assertEqual(
            initialize_metrics_record(
                str(self.paper), ["Accuracy", "F1", "Demographic Parity"],
                row_label="bank age",
            ),
            {"Accuracy": 91.17, "F1": 54.31, "Demographic Parity": 10.88},
        )


if __name__ == "__main__":
    unittest.main()
