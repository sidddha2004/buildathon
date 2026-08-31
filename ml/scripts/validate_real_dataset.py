"""Validate the real-image manifest before any expensive GPU work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.swapshield_ml.real_dataset import load_real_manifest, validate_real_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SwapShield's item-disjoint real-image manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    records = load_real_manifest(args.manifest)
    summary = validate_real_manifest(
        records,
        dataset_root=args.manifest.parent,
        check_files=args.check_files,
        require_evaluation_splits=not args.allow_incomplete,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
