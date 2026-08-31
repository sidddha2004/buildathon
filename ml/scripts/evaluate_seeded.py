import argparse
import json

from ml.swapshield_ml.evaluation import evaluate_seeded_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SwapShield's deterministic POC baseline")
    parser.add_argument("--threshold", type=float, default=0.68)
    parser.add_argument("--seed", type=int, default=5050)
    parser.add_argument("--samples", type=int, default=480)
    args = parser.parse_args()

    summary = evaluate_seeded_baseline(
        threshold=args.threshold,
        seed=args.seed,
        sample_size=args.samples,
    )
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

