"""Select a validation threshold and report the untouched real-image test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.swapshield_ml.real_evaluation import (
    evaluate_real_predictions,
    load_real_predictions,
    real_report_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cached real-image predictions")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--false-positive-cost", type=float, default=80)
    parser.add_argument("--false-negative-cost", type=float, default=6200)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()

    predictions = load_real_predictions(args.predictions)
    report = evaluate_real_predictions(
        predictions,
        false_positive_cost=args.false_positive_cost,
        false_negative_cost=args.false_negative_cost,
        bootstrap_samples=args.bootstrap_samples,
    )
    rendered_json = json.dumps(report, indent=2, sort_keys=True)
    print(rendered_json)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered_json + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(real_report_markdown(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
