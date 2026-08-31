"""Fit and calibrate the SwapShield fusion classifier without touching test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.swapshield_ml.fusion import load_feature_rows, train_fusion_model, training_report_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SwapShield fusion on train and calibrate on validation")
    parser.add_argument("features", type=Path)
    parser.add_argument("model_output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--seed", type=int, default=5050)
    parser.add_argument("--false-positive-cost", type=float, default=80)
    parser.add_argument("--false-negative-cost", type=float, default=6200)
    parser.add_argument("--min-image-quality", type=float, default=0.46)
    args = parser.parse_args()

    rows = load_feature_rows(args.features)
    model, report = train_fusion_model(
        rows,
        seed=args.seed,
        false_positive_cost=args.false_positive_cost,
        false_negative_cost=args.false_negative_cost,
        min_image_quality=args.min_image_quality,
    )
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(json.dumps(model.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rendered_report = json.dumps(report, indent=2, sort_keys=True)
    print(rendered_report)
    if args.report_output:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(training_report_markdown(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
