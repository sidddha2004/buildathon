"""Download a bounded ABO subset and build SwapShield's real-image manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.swapshield_ml.abo_subset import DEFAULT_CATEGORIES, materialize_abo_subset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download selected ABO 360 views and create an item-disjoint SwapShield benchmark"
    )
    parser.add_argument("--output", type=Path, default=Path("data/real/abo"))
    parser.add_argument("--items", type=int, default=120, help="unique product identities (default: 120)")
    parser.add_argument("--views-per-item", type=int, default=3)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--seed", type=int, default=5050)
    parser.add_argument("--workers", type=int, default=6, help="parallel image downloads (1-16)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing generated manifest")
    args = parser.parse_args()

    summary = materialize_abo_subset(
        args.output,
        item_count=args.items,
        views_per_item=args.views_per_item,
        categories=args.categories,
        seed=args.seed,
        workers=args.workers,
        overwrite_manifest=args.overwrite,
    )
    concise = {key: value for key, value in summary.items() if key != "selection"}
    print("\nABO subset ready")
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
