"""Strict manifest contract for the locked real-image benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SPLITS = frozenset({"train", "validation", "test"})
LABELS = frozenset({"genuine", "substitution"})


class DatasetValidationError(ValueError):
    """Raised when a real-image manifest is unsafe or leaks identities."""


def _required_text(value: Any, name: str, *, max_length: int = 160) -> str:
    if not isinstance(value, str):
        raise DatasetValidationError(f"{name} must be a string")
    cleaned = " ".join(value.split())
    if not cleaned or len(cleaned) > max_length:
        raise DatasetValidationError(f"{name} must contain 1-{max_length} characters")
    return cleaned


def _optional_text(value: Any, name: str, *, max_length: int = 160) -> str | None:
    if value is None:
        return None
    return _required_text(value, name, max_length=max_length)


def _relative_image_path(value: Any, name: str) -> str:
    text = _required_text(value, name, max_length=500)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise DatasetValidationError(f"{name} must stay inside the dataset directory")
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise DatasetValidationError(f"{name} must be JPEG, PNG, or WebP")
    return path.as_posix()


def _optional_weight(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetValidationError(f"{name} must be numeric")
    parsed = float(value)
    if parsed < 0 or (name.startswith("dispatch") and parsed == 0):
        raise DatasetValidationError(f"{name} is outside the allowed range")
    return parsed


@dataclass(frozen=True, slots=True)
class RealPairRecord:
    case_id: str
    split: str
    label: str
    category: str
    dispatch_item_id: str
    return_item_id: str | None
    dispatch_image: str
    return_image: str
    source: str
    source_license: str
    slices: tuple[str, ...] = ()
    dispatch_sku: str | None = None
    return_sku: str | None = None
    dispatch_serial: str | None = None
    return_serial: str | None = None
    dispatch_weight_grams: float | None = None
    return_weight_grams: float | None = None

    @property
    def target(self) -> int:
        return int(self.label == "substitution")

    @classmethod
    def from_mapping(cls, raw: Any, *, line_number: int) -> "RealPairRecord":
        if not isinstance(raw, dict):
            raise DatasetValidationError(f"line {line_number} must contain one JSON object")

        required = {
            "case_id",
            "split",
            "label",
            "category",
            "dispatch_item_id",
            "return_item_id",
            "dispatch_image",
            "return_image",
            "source",
            "source_license",
        }
        optional = {
            "slices",
            "dispatch_sku",
            "return_sku",
            "dispatch_serial",
            "return_serial",
            "dispatch_weight_grams",
            "return_weight_grams",
        }
        missing = sorted(required - set(raw))
        unexpected = sorted(set(raw) - required - optional)
        if missing or unexpected:
            raise DatasetValidationError(
                f"line {line_number} keys invalid; missing={missing}, unexpected={unexpected}"
            )

        split = _required_text(raw["split"], "split", max_length=16).lower()
        label = _required_text(raw["label"], "label", max_length=20).lower()
        if split not in SPLITS:
            raise DatasetValidationError(f"line {line_number} uses unsupported split {split!r}")
        if label not in LABELS:
            raise DatasetValidationError(f"line {line_number} uses unsupported label {label!r}")

        dispatch_item_id = _required_text(raw["dispatch_item_id"], "dispatch_item_id")
        return_item_id = _optional_text(raw["return_item_id"], "return_item_id")
        if label == "genuine" and return_item_id != dispatch_item_id:
            raise DatasetValidationError(f"line {line_number}: genuine pairs must use the same item identity")
        if label == "substitution" and return_item_id == dispatch_item_id:
            raise DatasetValidationError(
                f"line {line_number}: substitutions must use a different or empty return identity"
            )

        raw_slices = raw.get("slices", [])
        if not isinstance(raw_slices, list) or len(raw_slices) > 12:
            raise DatasetValidationError(f"line {line_number}: slices must be a list of at most 12 labels")
        slices = tuple(dict.fromkeys(_required_text(item, "slice", max_length=40) for item in raw_slices))

        return cls(
            case_id=_required_text(raw["case_id"], "case_id", max_length=80),
            split=split,
            label=label,
            category=_required_text(raw["category"], "category", max_length=80),
            dispatch_item_id=dispatch_item_id,
            return_item_id=return_item_id,
            dispatch_image=_relative_image_path(raw["dispatch_image"], "dispatch_image"),
            return_image=_relative_image_path(raw["return_image"], "return_image"),
            source=_required_text(raw["source"], "source", max_length=120),
            source_license=_required_text(raw["source_license"], "source_license", max_length=160),
            slices=slices,
            dispatch_sku=_optional_text(raw.get("dispatch_sku"), "dispatch_sku", max_length=80),
            return_sku=_optional_text(raw.get("return_sku"), "return_sku", max_length=80),
            dispatch_serial=_optional_text(raw.get("dispatch_serial"), "dispatch_serial", max_length=120),
            return_serial=_optional_text(raw.get("return_serial"), "return_serial", max_length=120),
            dispatch_weight_grams=_optional_weight(
                raw.get("dispatch_weight_grams"), "dispatch_weight_grams"
            ),
            return_weight_grams=_optional_weight(raw.get("return_weight_grams"), "return_weight_grams"),
        )


def load_real_manifest(path: str | Path) -> list[RealPairRecord]:
    manifest = Path(path)
    records: list[RealPairRecord] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(f"line {line_number} is not valid JSON") from exc
            records.append(RealPairRecord.from_mapping(raw, line_number=line_number))
    if not records:
        raise DatasetValidationError("manifest contains no records")
    return records


def validate_real_manifest(
    records: Iterable[RealPairRecord],
    *,
    dataset_root: str | Path | None = None,
    check_files: bool = False,
    require_evaluation_splits: bool = True,
) -> dict[str, Any]:
    materialized = list(records)
    case_ids: set[str] = set()
    identity_splits: dict[str, str] = {}
    split_counts = {split: {label: 0 for label in LABELS} for split in SPLITS}

    for record in materialized:
        if record.case_id in case_ids:
            raise DatasetValidationError(f"duplicate case_id {record.case_id!r}")
        case_ids.add(record.case_id)
        split_counts[record.split][record.label] += 1

        identities = {record.dispatch_item_id}
        if record.return_item_id:
            identities.add(record.return_item_id)
        for identity in identities:
            existing = identity_splits.get(identity)
            if existing is not None and existing != record.split:
                raise DatasetValidationError(
                    f"identity leakage: {identity!r} appears in both {existing!r} and {record.split!r}"
                )
            identity_splits[identity] = record.split

        if check_files:
            if dataset_root is None:
                raise DatasetValidationError("dataset_root is required when check_files is enabled")
            root = Path(dataset_root).resolve()
            for relative in (record.dispatch_image, record.return_image):
                candidate = (root / relative).resolve()
                if root not in candidate.parents:
                    raise DatasetValidationError(f"case {record.case_id!r} escapes the dataset directory")
                if not candidate.is_file():
                    raise DatasetValidationError(f"case {record.case_id!r} is missing {relative!r}")

    if require_evaluation_splits:
        for split in ("validation", "test"):
            missing = [label for label, count in split_counts[split].items() if count == 0]
            if missing:
                raise DatasetValidationError(f"{split} split is missing labels: {missing}")

    return {
        "cases": len(materialized),
        "identities": len(identity_splits),
        "splits": {
            split: {
                "cases": sum(counts.values()),
                "genuine": counts["genuine"],
                "substitution": counts["substitution"],
            }
            for split, counts in sorted(split_counts.items())
        },
    }
