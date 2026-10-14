
import unittest
import json
import tempfile
from pathlib import Path

from team.insert_into_record_tool.insert_into_record import MetricsRecord


class TestMetricsRecord(unittest.TestCase):

    def setUp(self):
        self.record = MetricsRecord({
            "Accuracy": 92.0,
            "F1": None
        })

    def test_update_existing_metric(self):
        self.record.insert_into_record("F1", 0.87)
        self.assertEqual(self.record.record["F1"], 0.87)

    def test_reject_unknown_metric(self):
        with self.assertRaises(KeyError):
            self.record.insert_into_record("Precision", 0.90)

    def test_reject_string_value(self):
        with self.assertRaises(TypeError):
            self.record.insert_into_record("Accuracy", "hello")

    def test_reject_nan(self):
        with self.assertRaises(ValueError):
            self.record.insert_into_record("Accuracy", float("nan"))

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.json"

            self.record.insert_into_record("F1", 0.87)
            self.record.save_json(path)

            loaded = MetricsRecord.load_json(path)

            self.assertEqual(loaded.record, {
                "Accuracy": 92.0,
                "F1": 0.87
            })

    def test_reject_infinity(self):
        with self.assertRaises(ValueError):
            self.record.insert_into_record(
                "Accuracy", float("inf")
            )

    def test_reject_boolean(self):
        with self.assertRaises(TypeError):
            self.record.insert_into_record(
                "Accuracy", True
            )

    def test_reject_empty_metric(self):
        with self.assertRaises(ValueError):
            self.record.insert_into_record("  ", 0.87)

    def test_update_preserves_other_metrics(self):
        self.record.insert_into_record("F1", 0.91)

        self.assertEqual(self.record.record["Accuracy"], 92.0)
        self.assertEqual(self.record.record["F1"], 0.91)

    def test_missing_json_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.json"

            with self.assertRaises(FileNotFoundError):
                MetricsRecord.load_json(path)

    def test_corrupted_json_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "corrupted.json"
            path.write_text("{invalid json", encoding="utf-8")

            with self.assertRaises(json.JSONDecodeError):
                MetricsRecord.load_json(path)

    def test_invalid_json_record(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.json"
            path.write_text(
                '{"Accuracy": "hello"}',
                encoding="utf-8"
            )

            with self.assertRaises(TypeError):
                MetricsRecord.load_json(path)

    def test_prevent_accidental_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.json"

            self.record.save_json(path)

            with self.assertRaises(FileExistsError):
                self.record.save_json(path)

            self.record.insert_into_record("F1", 0.87)
            self.record.save_json(path, overwrite=True)

            loaded = MetricsRecord.load_json(path)
            self.assertEqual(loaded.record["F1"], 0.87)

    def test_integration_with_initializer(self):
        from team.initialize_metrics_record_tool import initialize_metrics_record

        with tempfile.TemporaryDirectory() as folder:
            paper = Path(folder) / "paper.txt"

            paper.write_text(
                "Results\n"
                "Our proposed method achieved F1 = 0.87 "
                "and Accuracy = 92%.\n",
                encoding="utf-8"
            )

            initialized = initialize_metrics_record(
                str(paper), ["F1", "Accuracy"]
            )

            record = MetricsRecord(initialized)

            record.insert_into_record("F1", 0.90)

            self.assertEqual(record.record["F1"], 0.90)
            self.assertEqual(record.record["Accuracy"], 92.0)


if __name__ == "__main__":
    unittest.main()
