"""Dependency-free evaluation for versioned real-image predictions."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Any, Iterable


class PredictionValidationError(ValueError):
    """Raised when cached predictions cannot support an honest benchmark."""


@dataclass(frozen=True, slots=True)
class RealPrediction:
    case_id: str
    split: str
    target: int
    probability: float
    decision: str
    category: str
    slices: tuple[str, ...]
    latency_ms: float

    @classmethod
    def from_mapping(cls, raw: Any, *, line_number: int) -> "RealPrediction":
        if not isinstance(raw, dict):
            raise PredictionValidationError(f"prediction line {line_number} must be a JSON object")
        required = {
            "case_id",
            "split",
            "target",
            "probability",
            "decision",
            "category",
            "slices",
            "latency_ms",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise PredictionValidationError(f"prediction line {line_number} is missing {missing}")
        probability = raw["probability"]
        latency = raw["latency_ms"]
        target = raw["target"]
        slices = raw["slices"]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise PredictionValidationError("probability must be numeric")
        if not 0 <= float(probability) <= 1:
            raise PredictionValidationError("probability must be between 0 and 1")
        if target not in {0, 1}:
            raise PredictionValidationError("target must be 0 or 1")
        if raw["split"] not in {"validation", "test"}:
            raise PredictionValidationError("real evaluation accepts validation and test predictions only")
        if raw["decision"] not in {"approve", "review", "recapture"}:
            raise PredictionValidationError("decision is unsupported")
        if not isinstance(slices, list) or any(not isinstance(item, str) for item in slices):
            raise PredictionValidationError("slices must be a list of strings")
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or latency < 0:
            raise PredictionValidationError("latency_ms must be non-negative")
        return cls(
            case_id=str(raw["case_id"]),
            split=str(raw["split"]),
            target=int(target),
            probability=float(probability),
            decision=str(raw["decision"]),
            category=str(raw["category"]),
            slices=tuple(dict.fromkeys(slices)),
            latency_ms=float(latency),
        )


def load_real_predictions(path: str | Path) -> list[RealPrediction]:
    predictions: list[RealPrediction] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PredictionValidationError(f"prediction line {line_number} is invalid JSON") from exc
            prediction = RealPrediction.from_mapping(raw, line_number=line_number)
            if prediction.case_id in seen:
                raise PredictionValidationError(f"duplicate prediction {prediction.case_id!r}")
            seen.add(prediction.case_id)
            predictions.append(prediction)
    if not predictions:
        raise PredictionValidationError("prediction file is empty")
    return predictions


def _confusion(records: Iterable[RealPrediction], threshold: float) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for item in records:
        predicted = item.probability >= threshold
        tp += int(predicted and item.target == 1)
        fp += int(predicted and item.target == 0)
        tn += int(not predicted and item.target == 0)
        fn += int(not predicted and item.target == 1)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _classification_metrics(records: list[RealPrediction], threshold: float) -> dict[str, Any]:
    confusion = _confusion(records, threshold)
    tp, fp, tn, fn = (confusion[key] for key in ("tp", "fp", "tn", "fn"))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "cases": len(records),
        "positive_rate": sum(item.target for item in records) / max(len(records), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": confusion,
    }


def _average_precision(records: list[RealPrediction]) -> float:
    positives = sum(item.target for item in records)
    if positives == 0:
        return 0.0
    ordered = sorted(records, key=lambda item: item.probability, reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, item in enumerate(ordered, start=1):
        if item.target:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def _expected_calibration_error(records: list[RealPrediction], bins: int = 10) -> float:
    if not records:
        return 0.0
    total = len(records)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [
            item
            for item in records
            if lower <= item.probability < upper or (index == bins - 1 and item.probability == 1)
        ]
        if not bucket:
            continue
        confidence = sum(item.probability for item in bucket) / len(bucket)
        observed = sum(item.target for item in bucket) / len(bucket)
        error += len(bucket) / total * abs(confidence - observed)
    return error


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = floor(position)
    upper = ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def select_cost_threshold(
    validation: list[RealPrediction],
    *,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict[str, float | int]:
    if not validation or len({item.target for item in validation}) < 2:
        raise PredictionValidationError("validation predictions need both genuine and substitution cases")
    candidates = [index / 100 for index in range(1, 100)]
    ranked: list[tuple[float, int, float, dict[str, int]]] = []
    for threshold in candidates:
        confusion = _confusion(validation, threshold)
        cost = confusion["fp"] * false_positive_cost + confusion["fn"] * false_negative_cost
        ranked.append((cost, confusion["fp"] + confusion["fn"], -threshold, confusion))
    cost, _, negative_threshold, confusion = min(ranked, key=lambda item: item[:3])
    return {
        "threshold": -negative_threshold,
        "expected_cost": cost,
        "false_positives": confusion["fp"],
        "false_negatives": confusion["fn"],
    }


def _bootstrap_intervals(
    records: list[RealPrediction], threshold: float, *, samples: int, seed: int
) -> dict[str, list[float]]:
    generator = random.Random(seed)
    values = {"precision": [], "recall": [], "f1": []}
    for _ in range(samples):
        resampled = [records[generator.randrange(len(records))] for _ in records]
        metrics = _classification_metrics(resampled, threshold)
        for name in values:
            values[name].append(float(metrics[name]))
    return {
        name: [_percentile(metric_values, 0.025), _percentile(metric_values, 0.975)]
        for name, metric_values in values.items()
    }


def _slice_report(records: list[RealPrediction], threshold: float, *, minimum_cases: int = 5) -> dict[str, Any]:
    groups: dict[str, list[RealPrediction]] = {}
    for item in records:
        groups.setdefault(f"category:{item.category}", []).append(item)
        for slice_name in item.slices:
            groups.setdefault(f"slice:{slice_name}", []).append(item)
    return {
        name: _classification_metrics(group, threshold)
        for name, group in sorted(groups.items())
        if len(group) >= minimum_cases and len({item.target for item in group}) == 2
    }


def evaluate_real_predictions(
    predictions: Iterable[RealPrediction],
    *,
    false_positive_cost: float = 80,
    false_negative_cost: float = 6200,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 5050,
) -> dict[str, Any]:
    materialized = list(predictions)
    validation = [item for item in materialized if item.split == "validation"]
    test = [item for item in materialized if item.split == "test"]
    if not validation or not test:
        raise PredictionValidationError("predictions must include locked validation and test splits")
    if len({item.target for item in test}) < 2:
        raise PredictionValidationError("test predictions need both genuine and substitution cases")

    selection = select_cost_threshold(
        validation,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    )
    threshold = float(selection["threshold"])
    metrics = _classification_metrics(test, threshold)
    confusion = metrics["confusion"]
    recaptures = [item for item in test if item.decision == "recapture"]
    genuine = [item for item in test if item.target == 0]
    genuine_recaptures = [item for item in recaptures if item.target == 0]
    latencies = [item.latency_ms for item in test]
    metrics.update(
        {
            "average_precision": _average_precision(test),
            "calibration_error": _expected_calibration_error(test),
            "false_positive_cost": confusion["fp"] * false_positive_cost,
            "missed_substitution_loss": confusion["fn"] * false_negative_cost,
            "recapture_rate": len(recaptures) / len(test),
            "genuine_recapture_rate": len(genuine_recaptures) / max(len(genuine), 1),
            "latency_ms": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95)},
            "bootstrap_95_ci": _bootstrap_intervals(
                test, threshold, samples=bootstrap_samples, seed=bootstrap_seed
            ),
            "slices": _slice_report(test, threshold),
        }
    )
    return {
        "protocol": "item-disjoint-v1",
        "threshold_selection": {"split": "validation", **selection},
        "cost_assumptions": {
            "false_positive_inr": false_positive_cost,
            "false_negative_inr": false_negative_cost,
        },
        "test": metrics,
    }


def real_report_markdown(report: dict[str, Any]) -> str:
    test = report["test"]
    confusion = test["confusion"]
    selection = report["threshold_selection"]
    return "\n".join(
        [
            "# SwapShield real-image evaluation",
            "",
            f"Operating threshold: **{selection['threshold']:.2f}**, selected on validation only.",
            "",
            "| Metric | Held-out test result |",
            "|---|---:|",
            f"| Cases | {test['cases']} |",
            f"| Precision | {test['precision']:.3f} |",
            f"| Recall | {test['recall']:.3f} |",
            f"| F1 | {test['f1']:.3f} |",
            f"| Average precision (PR) | {test['average_precision']:.3f} |",
            f"| Calibration error | {test['calibration_error']:.3f} |",
            f"| Recapture rate | {test['recapture_rate']:.3f} |",
            f"| p50 latency | {test['latency_ms']['p50']:.0f} ms |",
            f"| p95 latency | {test['latency_ms']['p95']:.0f} ms |",
            "",
            f"Confusion matrix: TP={confusion['tp']}, FP={confusion['fp']}, "
            f"TN={confusion['tn']}, FN={confusion['fn']}.",
            "",
            "This report is generated from the locked item-disjoint test split; synthetic results are separate.",
        ]
    )
