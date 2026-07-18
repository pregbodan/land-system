"""
Evaluate and summarize trained model outputs from JSON artifacts.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file if present, otherwise return None."""
    if not path.exists():
        logger.warning("File not found: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", path, exc)
        return None


def _fmt_metric(value: Any) -> str:
    """Format metric values consistently."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def print_training_summary(training_results: Dict[str, Any]) -> None:
    """Print model-level summary sorted by accuracy."""
    if not training_results:
        print("No training results found.")
        return

    rows = []
    for model_name, metrics in training_results.items():
        rows.append(
            (
                model_name,
                float(metrics.get("accuracy", 0.0)),
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1_score"),
            )
        )

    rows.sort(key=lambda item: item[1], reverse=True)

    print("\nTRAINING RESULTS")
    print("-" * 79)
    print(f"{'Model':<24}{'Accuracy':>12}{'Precision':>12}{'Recall':>12}{'F1':>12}")
    print("-" * 79)
    for model_name, acc, prec, rec, f1 in rows:
        print(
            f"{model_name:<24}"
            f"{_fmt_metric(acc):>12}"
            f"{_fmt_metric(prec):>12}"
            f"{_fmt_metric(rec):>12}"
            f"{_fmt_metric(f1):>12}"
        )
    print("-" * 79)


def print_metadata(metadata: Dict[str, Any]) -> None:
    """Print best-model metadata block."""
    if not metadata:
        print("\nNo model metadata found.")
        return

    print("\nMODEL METADATA")
    print("-" * 79)
    print(f"Best model:   {metadata.get('best_model_name', 'N/A')}")
    print(f"Best accuracy:{_fmt_metric(metadata.get('best_accuracy'))}")
    print(f"Features:     {metadata.get('feature_count', 'N/A')}")
    classes = metadata.get("label_encoder_classes") or []
    print(f"Classes:      {len(classes)}")
    if classes:
        print("Class labels: " + ", ".join(str(c) for c in classes))


def print_cv_summary(cv_results: Dict[str, Any]) -> None:
    """Print cross-validation summary if available."""
    if not cv_results:
        print("\nNo cross-validation results found.")
        return

    print("\nCROSS-VALIDATION")
    print("-" * 79)
    folds_used = cv_results.get("cv_folds_used")
    if folds_used is not None:
        print(f"Folds used:   {folds_used}")
    print(f"Accuracy:     {_fmt_metric(cv_results.get('accuracy_mean'))} (+/- {_fmt_metric(cv_results.get('accuracy_std'))})")
    print(f"Precision:    {_fmt_metric(cv_results.get('precision_mean'))} (+/- {_fmt_metric(cv_results.get('precision_std'))})")
    print(f"Recall:       {_fmt_metric(cv_results.get('recall_mean'))} (+/- {_fmt_metric(cv_results.get('recall_std'))})")
    print(f"F1 Score:     {_fmt_metric(cv_results.get('f1_mean'))} (+/- {_fmt_metric(cv_results.get('f1_std'))})")
    if cv_results.get("message"):
        print(f"Note:         {cv_results['message']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model artifacts")
    parser.add_argument(
        "--results_file",
        type=str,
        default="data/models/training_results.json",
        help="Path to training results JSON",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="data/models/model_metadata.json",
        help="Path to model metadata JSON",
    )
    parser.add_argument(
        "--cv_file",
        type=str,
        default="data/models/cv_results.json",
        help="Path to cross-validation results JSON",
    )
    args = parser.parse_args()

    results_path = Path(args.results_file)
    metadata_path = Path(args.metadata_file)
    cv_path = Path(args.cv_file)

    training_results = _load_json(results_path) or {}
    metadata = _load_json(metadata_path) or {}
    cv_results = _load_json(cv_path) or {}

    print_training_summary(training_results)
    print_metadata(metadata)
    print_cv_summary(cv_results)

    if not training_results and not metadata and not cv_results:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
