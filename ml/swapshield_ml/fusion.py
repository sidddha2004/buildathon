"""Train and run SwapShield's calibrated, serializable fusion classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Any, Iterable, Sequence

from .contracts import Decision, RiskFeatures, RiskResult


RAW_FEATURES = (
    "vision_similarity",
    "vlm_mismatch",
    "serial_mismatch",
    "weight_delta",
    "image_quality",
)
MODEL_FEATURES = (
    "vision_dissimilarity",
    "vlm_mismatch",
    "serial_mismatch",
    "weight_delta_clipped",
    "quality_deficit",
)
SPLITS = frozenset({"train", "validation", "test"})


class FusionTrainingError(ValueError):
    """Raised when cached features or a model artifact violate the protocol."""


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FusionTrainingError(f"{name} must be numeric")
    parsed = float(value)
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        raise FusionTrainingError(f"{name} must be finite")
    return parsed


def validate_feature_row(raw: Any, *, line_number: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise FusionTrainingError(f"feature line {line_number} must be a JSON object")
    required = {"case_id", "split", "target", "category", "slices", "latency_ms", "features"}
    missing = sorted(required - set(raw))
    if missing:
        raise FusionTrainingError(f"feature line {line_number} is missing {missing}")
    split = raw["split"]
    if split not in SPLITS:
        raise FusionTrainingError(f"feature line {line_number} has unsupported split {split!r}")
    if raw["target"] not in {0, 1}:
        raise FusionTrainingError(f"feature line {line_number} target must be 0 or 1")
    features = raw["features"]
    if not isinstance(features, dict):
        raise FusionTrainingError(f"feature line {line_number} features must be an object")
    for name in RAW_FEATURES:
        value = _finite_number(features.get(name), f"features.{name}")
        if name != "weight_delta" and not 0 <= value <= 1:
            raise FusionTrainingError(f"features.{name} must be between 0 and 1")
        if name == "weight_delta" and value < 0:
            raise FusionTrainingError("features.weight_delta must be non-negative")
    if not isinstance(raw["slices"], list) or any(not isinstance(item, str) for item in raw["slices"]):
        raise FusionTrainingError(f"feature line {line_number} slices must be a string list")
    _finite_number(raw["latency_ms"], "latency_ms")
    return raw


def load_feature_rows(paths: str | Path | Sequence[str | Path]) -> list[dict[str, Any]]:
    selected_paths: Sequence[str | Path]
    if isinstance(paths, (str, Path)):
        selected_paths = [paths]
    else:
        selected_paths = paths
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in selected_paths:
        with Path(source).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise FusionTrainingError(f"{source}:{line_number} is not valid JSON") from exc
                row = validate_feature_row(raw, line_number=line_number)
                case_id = str(row["case_id"])
                if case_id in seen:
                    raise FusionTrainingError(f"duplicate cached feature case {case_id!r}")
                seen.add(case_id)
                rows.append(row)
    if not rows:
        raise FusionTrainingError("cached feature input is empty")
    return rows


def feature_vector(features: dict[str, Any]) -> list[float]:
    for name in RAW_FEATURES:
        _finite_number(features.get(name), f"features.{name}")
    return [
        1.0 - float(features["vision_similarity"]),
        float(features["vlm_mismatch"]),
        float(features["serial_mismatch"]),
        min(float(features["weight_delta"]) / 0.35, 1.0),
        1.0 - float(features["image_quality"]),
    ]


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    exponent = exp(value)
    return exponent / (1.0 + exponent)


@dataclass(frozen=True, slots=True)
class FusionModel:
    seed: int
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    calibration_coefficient: float
    calibration_intercept: float
    threshold: float
    min_image_quality: float
    false_positive_cost: float
    false_negative_cost: float
    train_cases: int
    validation_cases: int
    calibration_folds: int

    def probability(self, features: dict[str, Any]) -> float:
        vector = feature_vector(features)
        standardized = [
            (value - mean) / scale
            for value, mean, scale in zip(vector, self.means, self.scales, strict=True)
        ]
        score = self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, standardized, strict=True)
        )
        calibrated = self.calibration_intercept + self.calibration_coefficient * score
        return _sigmoid(calibrated)

    def decision(self, features: dict[str, Any]) -> str:
        if float(features["image_quality"]) < self.min_image_quality:
            return "recapture"
        return "review" if self.probability(features) >= self.threshold else "approve"

    def score_risk(self, features: RiskFeatures) -> RiskResult:
        features.validate()
        raw = {
            "vision_similarity": features.vision_similarity,
            "vlm_mismatch": features.vlm_mismatch,
            "serial_mismatch": features.serial_mismatch,
            "weight_delta": features.weight_delta,
            "image_quality": features.image_quality,
        }
        probability = self.probability(raw)
        decision = Decision(self.decision(raw))
        reasons: list[str] = []
        if features.vision_similarity < 0.68:
            reasons.append("Visual identity is materially different from dispatch evidence")
        if features.vlm_mismatch > 0.62:
            reasons.append("Vision-language verifier found product-level discrepancies")
        if features.serial_mismatch > 0.70:
            reasons.append("Serial or model identifier does not match the order record")
        if features.weight_delta > 0.18:
            reasons.append("Parcel weight differs beyond the configured tolerance")
        if features.image_quality < self.min_image_quality:
            reasons.append("Image quality is below the evidence threshold; capture clearer product photos")
        if not reasons:
            reasons.append("All available return evidence is consistent with dispatch")
        return RiskResult(probability, round(probability * 100), decision, tuple(reasons))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "swapshield-calibrated-logistic-fusion",
            "seed": self.seed,
            "features": list(MODEL_FEATURES),
            "normalization": {"mean": list(self.means), "scale": list(self.scales)},
            "classifier": {"coefficients": list(self.coefficients), "intercept": self.intercept},
            "calibration": {
                "method": f"platt-oof-{self.calibration_folds}-fold",
                "coefficient": self.calibration_coefficient,
                "intercept": self.calibration_intercept,
            },
            "decision": {
                "threshold": self.threshold,
                "min_image_quality": self.min_image_quality,
            },
            "cost_assumptions": {
                "false_positive_inr": self.false_positive_cost,
                "false_negative_inr": self.false_negative_cost,
            },
            "training": {
                "train_cases": self.train_cases,
                "validation_cases": self.validation_cases,
                "calibration_folds": self.calibration_folds,
            },
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "FusionModel":
        try:
            if raw["schema_version"] != 1 or raw["kind"] != "swapshield-calibrated-logistic-fusion":
                raise FusionTrainingError("unsupported fusion model artifact")
            if tuple(raw["features"]) != MODEL_FEATURES:
                raise FusionTrainingError("fusion model feature order is unsupported")
            means = tuple(float(value) for value in raw["normalization"]["mean"])
            scales = tuple(float(value) for value in raw["normalization"]["scale"])
            coefficients = tuple(float(value) for value in raw["classifier"]["coefficients"])
            if not (len(means) == len(scales) == len(coefficients) == len(MODEL_FEATURES)):
                raise FusionTrainingError("fusion model vector lengths are invalid")
            if any(scale <= 0 for scale in scales):
                raise FusionTrainingError("fusion model normalization scales must be positive")
            return cls(
                seed=int(raw["seed"]),
                means=means,
                scales=scales,
                coefficients=coefficients,
                intercept=float(raw["classifier"]["intercept"]),
                calibration_coefficient=float(raw["calibration"]["coefficient"]),
                calibration_intercept=float(raw["calibration"]["intercept"]),
                threshold=float(raw["decision"]["threshold"]),
                min_image_quality=float(raw["decision"]["min_image_quality"]),
                false_positive_cost=float(raw["cost_assumptions"]["false_positive_inr"]),
                false_negative_cost=float(raw["cost_assumptions"]["false_negative_inr"]),
                train_cases=int(raw["training"]["train_cases"]),
                validation_cases=int(raw["training"]["validation_cases"]),
                calibration_folds=int(raw["training"]["calibration_folds"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, FusionTrainingError):
                raise
            raise FusionTrainingError("fusion model artifact is malformed") from exc


def load_fusion_model(path: str | Path) -> FusionModel:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FusionTrainingError("fusion model artifact is not valid JSON") from exc
    return FusionModel.from_dict(raw)


def _confusion(probabilities: Sequence[float], targets: Sequence[int], threshold: float) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for probability, target in zip(probabilities, targets, strict=True):
        predicted = probability >= threshold
        tp += int(predicted and target == 1)
        fp += int(predicted and target == 0)
        tn += int(not predicted and target == 0)
        fn += int(not predicted and target == 1)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _metrics(probabilities: Sequence[float], targets: Sequence[int], threshold: float) -> dict[str, Any]:
    confusion = _confusion(probabilities, targets, threshold)
    tp, fp, tn, fn = (confusion[key] for key in ("tp", "fp", "tn", "fn"))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1, "confusion": confusion}


def _cost_threshold(
    probabilities: Sequence[float],
    targets: Sequence[int],
    *,
    false_positive_cost: float,
    false_negative_cost: float,
) -> tuple[float, float]:
    candidates = [index / 100 for index in range(1, 100)]
    ranked: list[tuple[float, int, float]] = []
    for threshold in candidates:
        confusion = _confusion(probabilities, targets, threshold)
        cost = confusion["fp"] * false_positive_cost + confusion["fn"] * false_negative_cost
        ranked.append((cost, confusion["fp"] + confusion["fn"], -threshold))
    cost, _, negative_threshold = min(ranked)
    return -negative_threshold, cost


def train_fusion_model(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int = 5050,
    false_positive_cost: float = 80,
    false_negative_cost: float = 6200,
    min_image_quality: float = 0.46,
) -> tuple[FusionModel, dict[str, Any]]:
    materialized = list(rows)
    if any(row["split"] == "test" for row in materialized):
        raise FusionTrainingError("training input contains test cases; keep test features in a separate file")
    training = [row for row in materialized if row["split"] == "train"]
    validation = [row for row in materialized if row["split"] == "validation"]
    for name, selected in (("train", training), ("validation", validation)):
        if not selected or {int(row["target"]) for row in selected} != {0, 1}:
            raise FusionTrainingError(f"{name} features must include genuine and substitution cases")

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - optional local training dependency
        raise RuntimeError("Install training dependencies: pip install -e 'ml[train]'") from exc

    x_train = np.asarray([feature_vector(row["features"]) for row in training], dtype=float)
    y_train = np.asarray([int(row["target"]) for row in training], dtype=int)
    minimum_class = min(int((y_train == value).sum()) for value in (0, 1))
    folds = min(5, minimum_class)
    if folds < 3:
        raise FusionTrainingError("training split needs at least three cases per class")

    def estimator():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
        )

    cross_validation = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    out_of_fold_scores = cross_val_predict(
        estimator(), x_train, y_train, cv=cross_validation, method="decision_function", n_jobs=1
    )
    calibrator = LogisticRegression(max_iter=2000, random_state=seed)
    calibrator.fit(out_of_fold_scores.reshape(-1, 1), y_train)
    fitted = estimator()
    fitted.fit(x_train, y_train)
    scaler = fitted.named_steps["standardscaler"]
    classifier = fitted.named_steps["logisticregression"]

    provisional = FusionModel(
        seed=seed,
        means=tuple(float(value) for value in scaler.mean_),
        scales=tuple(float(value) for value in scaler.scale_),
        coefficients=tuple(float(value) for value in classifier.coef_[0]),
        intercept=float(classifier.intercept_[0]),
        calibration_coefficient=float(calibrator.coef_[0][0]),
        calibration_intercept=float(calibrator.intercept_[0]),
        threshold=0.5,
        min_image_quality=min_image_quality,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
        train_cases=len(training),
        validation_cases=len(validation),
        calibration_folds=folds,
    )
    validation_probabilities = [provisional.probability(row["features"]) for row in validation]
    validation_targets = [int(row["target"]) for row in validation]
    threshold, cost = _cost_threshold(
        validation_probabilities,
        validation_targets,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    )
    model = FusionModel(
        seed=provisional.seed,
        means=provisional.means,
        scales=provisional.scales,
        coefficients=provisional.coefficients,
        intercept=provisional.intercept,
        calibration_coefficient=provisional.calibration_coefficient,
        calibration_intercept=provisional.calibration_intercept,
        threshold=threshold,
        min_image_quality=provisional.min_image_quality,
        false_positive_cost=provisional.false_positive_cost,
        false_negative_cost=provisional.false_negative_cost,
        train_cases=provisional.train_cases,
        validation_cases=provisional.validation_cases,
        calibration_folds=provisional.calibration_folds,
    )
    oof_probabilities = [
        _sigmoid(model.calibration_intercept + model.calibration_coefficient * float(score))
        for score in out_of_fold_scores
    ]
    report = {
        "protocol": "fusion-train-validation-v1",
        "seed": seed,
        "feature_names": list(MODEL_FEATURES),
        "train": {
            "cases": len(training),
            "oof_average_precision": float(average_precision_score(y_train, oof_probabilities)),
        },
        "validation": {
            "cases": len(validation),
            "average_precision": float(average_precision_score(validation_targets, validation_probabilities)),
            "threshold": threshold,
            "expected_cost_inr": cost,
            **_metrics(validation_probabilities, validation_targets, threshold),
        },
        "test_accessed": False,
    }
    return model, report


def score_feature_rows(
    model: FusionModel,
    rows: Iterable[dict[str, Any]],
    *,
    include_splits: set[str] | None = None,
    exclude_case_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    allowed = include_splits or {"validation", "test"}
    excluded = exclude_case_ids or set()
    output: list[dict[str, Any]] = []
    for row in rows:
        if row["split"] not in allowed or row["case_id"] in excluded:
            continue
        probability = model.probability(row["features"])
        scored = dict(row)
        scored["baseline_probability"] = row.get("probability")
        scored["probability"] = probability
        scored["decision"] = model.decision(row["features"])
        scored["fusion_model"] = "swapshield-calibrated-logistic-fusion-v1"
        output.append(scored)
    return sorted(output, key=lambda row: str(row["case_id"]))


def training_report_markdown(report: dict[str, Any]) -> str:
    validation = report["validation"]
    confusion = validation["confusion"]
    return "\n".join(
        [
            "# SwapShield fusion training report",
            "",
            "The classifier was fitted on train and the operating threshold was selected on validation.",
            "The test split was not accessed.",
            "",
            "| Measure | Result |",
            "|---|---:|",
            f"| Training cases | {report['train']['cases']} |",
            f"| Validation cases | {validation['cases']} |",
            f"| Train OOF average precision | {report['train']['oof_average_precision']:.3f} |",
            f"| Validation average precision | {validation['average_precision']:.3f} |",
            f"| Validation precision | {validation['precision']:.3f} |",
            f"| Validation recall | {validation['recall']:.3f} |",
            f"| Validation F1 | {validation['f1']:.3f} |",
            f"| Selected threshold | {validation['threshold']:.2f} |",
            f"| Expected validation cost | INR {validation['expected_cost_inr']:.0f} |",
            "",
            f"Validation confusion matrix: TP={confusion['tp']}, FP={confusion['fp']}, "
            f"TN={confusion['tn']}, FN={confusion['fn']}.",
        ]
    )
