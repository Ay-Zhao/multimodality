import csv
import tempfile
import unittest
from pathlib import Path

from src.training.run_10fold_cv import (
    best_metrics_from_epoch_csv,
    build_fold_command,
    write_summary,
)


class RunTenFoldCvTests(unittest.TestCase):
    def test_fold_command_overrides_fold_specific_paths(self):
        fold_dir = Path("output") / "fold_03"
        command = build_fold_command(
            [
                "--dataset", "bace",
                "--cv_fold_id", "9",
                "--csv_dir=old-results",
                "--tensorboard_dir", "old-logs",
            ],
            cv_folds=10,
            fold_id=3,
            fold_dir=fold_dir,
        )

        self.assertIn("bace", command)
        self.assertEqual(command[command.index("--cv_fold_id") + 1], "3")
        self.assertEqual(command[command.index("--cv_folds") + 1], "10")
        self.assertNotIn("old-results", command)
        self.assertNotIn("old-logs", command)

    def test_best_epoch_is_selected_by_validation_auc(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            metrics_path = Path(temporary_dir) / "metrics.csv"
            with metrics_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["epoch", "valid_auc_roc", "test_auc_roc"],
                )
                writer.writeheader()
                writer.writerow({"epoch": 0, "valid_auc_roc": 0.70, "test_auc_roc": 0.80})
                writer.writerow({"epoch": 1, "valid_auc_roc": 0.90, "test_auc_roc": 0.75})

            result = best_metrics_from_epoch_csv(metrics_path)

        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["valid_auc_roc"], 0.90)
        self.assertEqual(result["test_auc_roc"], 0.75)

    def test_summary_contains_mean_and_population_standard_deviation(self):
        records = [
            {"valid_auc_roc": 0.70, "test_auc_roc": 0.60},
            {"valid_auc_roc": 0.90, "test_auc_roc": 0.80},
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            summary = write_summary(
                records,
                cv_folds=2,
                destination=Path(temporary_dir) / "summary.csv",
            )

        self.assertAlmostEqual(summary["valid_auc_mean"], 0.80)
        self.assertAlmostEqual(summary["valid_auc_std"], 0.10)
        self.assertAlmostEqual(summary["test_auc_mean"], 0.70)
        self.assertAlmostEqual(summary["test_auc_std"], 0.10)


if __name__ == "__main__":
    unittest.main()
