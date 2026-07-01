"""Launch every scaffold-CV fold and aggregate fold-level AUC metrics.

Example:
    python -m src.training.run_10fold_cv --dataset bace --epochs 100

Arguments that are not owned by this launcher are forwarded to
``src.training.train_GGT_3``. Folds run sequentially so a single GPU is not
shared by multiple training processes.
"""

import argparse
import csv
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTROLLED_TRAINING_OPTIONS = {
    "--cv_fold_id",
    "--csv_dir",
    "--tensorboard_dir",
}


def _remove_controlled_options(arguments):
    """Remove options whose values must be unique for each launched fold."""
    cleaned = []
    skip_next = False
    for argument in arguments:
        if skip_next:
            skip_next = False
            continue
        if argument in CONTROLLED_TRAINING_OPTIONS:
            skip_next = True
            continue
        if any(argument.startswith(option + "=") for option in CONTROLLED_TRAINING_OPTIONS):
            continue
        cleaned.append(argument)
    return cleaned


def build_fold_command(training_arguments, cv_folds, fold_id, fold_dir):
    """Build one isolated training command for a CV fold."""
    training_arguments = _remove_controlled_options(training_arguments)
    return [
        sys.executable,
        "-m",
        "src.training.train_GGT_3",
        *training_arguments,
        "--cv_folds",
        str(cv_folds),
        "--cv_fold_id",
        str(fold_id),
        "--csv_dir",
        str(fold_dir),
        "--tensorboard_dir",
        str(fold_dir / "tensorboard"),
    ]


def best_metrics_from_epoch_csv(metrics_path):
    """Return metrics from the epoch selected by maximum validation AUC."""
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No epoch rows found in {}".format(metrics_path))

    best_row = max(rows, key=lambda row: float(row["valid_auc_roc"]))
    return {
        "best_epoch": int(best_row["epoch"]),
        "valid_auc_roc": float(best_row["valid_auc_roc"]),
        "test_auc_roc": float(best_row["test_auc_roc"]),
    }


def write_fold_results(records, destination):
    """Write one result row for every completed fold."""
    fieldnames = [
        "test_fold",
        "valid_fold",
        "best_epoch",
        "valid_auc_roc",
        "test_auc_roc",
        "metrics_file",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_summary(records, cv_folds, destination):
    """Write aggregate validation and test AUC statistics."""
    valid_aucs = [record["valid_auc_roc"] for record in records]
    test_aucs = [record["test_auc_roc"] for record in records]
    summary = {
        "cv_folds": cv_folds,
        "completed_folds": len(records),
        "valid_auc_mean": statistics.mean(valid_aucs),
        "valid_auc_std": statistics.pstdev(valid_aucs),
        "test_auc_mean": statistics.mean(test_aucs),
        "test_auc_std": statistics.pstdev(test_aucs),
    }
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    return summary


def parse_launcher_arguments():
    parser = argparse.ArgumentParser(
        description="Run all scaffold cross-validation folds sequentially.",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=10,
        help="Number of folds to launch (default: 10).",
    )
    parser.add_argument(
        "--cv_output_dir",
        type=Path,
        default=None,
        help="Output root. Defaults to results/cv_runs/cv<FOLDS>_<timestamp>.",
    )
    parser.add_argument(
        "--cv_valid_fold_offset",
        type=int,
        default=1,
        help="Validation-fold offset passed to every training run (default: 1).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print all fold commands without starting training.",
    )
    return parser.parse_known_args()


def main():
    launcher_args, training_arguments = parse_launcher_arguments()
    if launcher_args.cv_folds < 3:
        raise ValueError("cv_folds must be at least 3 for train/valid/test splitting")
    if launcher_args.cv_valid_fold_offset % launcher_args.cv_folds == 0:
        raise ValueError("cv_valid_fold_offset must select a fold different from the test fold")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = launcher_args.cv_output_dir
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "cv_runs" / (
            "cv{}_{}".format(launcher_args.cv_folds, timestamp)
        )
    output_dir = output_dir.resolve()

    commands = []
    for fold_id in range(launcher_args.cv_folds):
        fold_dir = output_dir / "fold_{:02d}".format(fold_id)
        commands.append(
            (fold_id, fold_dir, build_fold_command(
                training_arguments + [
                    "--cv_valid_fold_offset",
                    str(launcher_args.cv_valid_fold_offset),
                ],
                launcher_args.cv_folds,
                fold_id,
                fold_dir,
            ))
        )

    if launcher_args.dry_run:
        for fold_id, _, command in commands:
            print("[CV launcher] fold {}: {}".format(
                fold_id,
                subprocess.list2cmdline(command),
            ))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_results_path = output_dir / "fold_results.csv"
    records = []

    child_environment = os.environ.copy()
    existing_pythonpath = child_environment.get("PYTHONPATH", "")
    child_environment["PYTHONPATH"] = str(PROJECT_ROOT) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )

    for fold_id, fold_dir, command in commands:
        fold_dir.mkdir(parents=True, exist_ok=True)
        print("\n[CV launcher] starting fold {}/{}".format(
            fold_id + 1,
            launcher_args.cv_folds,
        ))
        subprocess.run(
            command,
            cwd=str(fold_dir),
            env=child_environment,
            check=True,
        )

        metric_files = list(fold_dir.glob("*.epoch_metrics.csv"))
        if len(metric_files) != 1:
            raise RuntimeError(
                "Expected one epoch metrics CSV for fold {}, found {}".format(
                    fold_id,
                    len(metric_files),
                )
            )
        metrics = best_metrics_from_epoch_csv(metric_files[0])
        valid_fold = (
            fold_id + launcher_args.cv_valid_fold_offset
        ) % launcher_args.cv_folds
        records.append({
            "test_fold": fold_id,
            "valid_fold": valid_fold,
            "best_epoch": metrics["best_epoch"],
            "valid_auc_roc": metrics["valid_auc_roc"],
            "test_auc_roc": metrics["test_auc_roc"],
            "metrics_file": str(metric_files[0]),
        })
        write_fold_results(records, fold_results_path)
        print("[CV launcher] fold {} test AUC: {:.6f}".format(
            fold_id,
            metrics["test_auc_roc"],
        ))

    summary_path = output_dir / "cv_summary.csv"
    summary = write_summary(records, launcher_args.cv_folds, summary_path)
    print("\n[CV launcher] completed {} folds".format(len(records)))
    print("[CV launcher] test AUC: {:.6f} +/- {:.6f}".format(
        summary["test_auc_mean"],
        summary["test_auc_std"],
    ))
    print("[CV launcher] fold results: {}".format(fold_results_path))
    print("[CV launcher] summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
