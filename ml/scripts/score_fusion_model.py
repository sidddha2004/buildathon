"""Apply a locked JSON fusion model to cached validation/test features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.swapshield_ml.fusion import load_feature_rows, load_fusion_model, score_feature_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Score cached SwapShield features with a trained fusion model")
    parser.add_argument("model", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-split", action="append", choices=["train", "validation", "test"])
    parser.add_argument("--exclude-case", action="append", default=[])
    args = parser.parse_args()

    model = load_fusion_model(args.model)
    rows = load_feature_rows(args.inputs)
    scored = score_feature_rows(
        model,
        rows,
        include_splits=set(args.include_split or ["validation", "test"]),
        exclude_case_ids=set(args.exclude_case),
    )
    if not scored:
        raise SystemExit("no rows matched the requested splits")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in scored:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "scored_cases": len(scored)}, indent=2))


if __name__ == "__main__":
    main()
