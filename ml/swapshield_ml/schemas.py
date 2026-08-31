"""Strict, defense-only contracts for vision-language model output."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


ALLOWED_EVIDENCE_IDS = frozenset(
    {
        "dispatch_image",
        "return_image",
        "dispatch_record",
        "return_record",
        "weight_record",
    }
)
ALLOWED_ATTRIBUTES = frozenset(
    {
        "brand",
        "model_text",
        "serial_text",
        "color",
        "shape",
        "logo",
        "packaging",
        "accessories",
        "condition",
        "dimensions",
        "other_observable",
    }
)
ALLOWED_SEVERITIES = frozenset({"minor", "material"})
ASSESSMENT_KEYS = frozenset(
    {
        "evidence_sufficient",
        "same_product_likelihood",
        "mismatch_confidence",
        "observations",
        "missing_evidence",
    }
)
OBSERVATION_KEYS = frozenset(
    {"attribute", "dispatch_value", "return_value", "severity", "evidence_ids"}
)


class SchemaValidationError(ValueError):
    """Raised when an LLM response violates the evidence-only contract."""


@dataclass(frozen=True, slots=True)
class VisualObservation:
    attribute: str
    dispatch_value: str
    return_value: str
    severity: str
    evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_ids"] = list(self.evidence_ids)
        return payload


@dataclass(frozen=True, slots=True)
class VlmAssessment:
    evidence_sufficient: bool
    same_product_likelihood: float
    mismatch_confidence: float
    observations: tuple[VisualObservation, ...]
    missing_evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_sufficient": self.evidence_sufficient,
            "same_product_likelihood": self.same_product_likelihood,
            "mismatch_confidence": self.mismatch_confidence,
            "observations": [item.to_dict() for item in self.observations],
            "missing_evidence": list(self.missing_evidence),
        }


def _bounded_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{name} must be a number")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise SchemaValidationError(f"{name} must be between 0 and 1")
    return parsed


def _short_text(value: Any, name: str, *, max_length: int = 240) -> str:
    if not isinstance(value, str):
        raise SchemaValidationError(f"{name} must be a string")
    cleaned = " ".join(value.split())
    if not cleaned or len(cleaned) > max_length:
        raise SchemaValidationError(f"{name} must contain 1-{max_length} characters")
    return cleaned


def _decode_json_only(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise SchemaValidationError("malformed JSON code fence")
        cleaned = "\n".join(lines[1:-1]).strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError("model response is not JSON-only") from exc
    if not isinstance(payload, dict):
        raise SchemaValidationError("model response must be a JSON object")
    return payload


def parse_vlm_json(text: str, *, allow_missing_empty_lists: bool = False) -> VlmAssessment:
    """Parse a Qwen response and reject all fields outside the allowlist.

    A local model occasionally omits an empty list despite otherwise following
    the schema. Batch inference may opt into filling only the two list fields;
    direct callers retain the fully strict default.
    """
    payload = _decode_json_only(text)
    if allow_missing_empty_lists:
        for name in ("observations", "missing_evidence"):
            payload.setdefault(name, [])
    if set(payload) != ASSESSMENT_KEYS:
        unexpected = sorted(set(payload) - ASSESSMENT_KEYS)
        missing = sorted(ASSESSMENT_KEYS - set(payload))
        raise SchemaValidationError(f"assessment keys invalid; unexpected={unexpected}, missing={missing}")

    sufficient = payload["evidence_sufficient"]
    if not isinstance(sufficient, bool):
        raise SchemaValidationError("evidence_sufficient must be a boolean")

    raw_observations = payload["observations"]
    if not isinstance(raw_observations, list) or len(raw_observations) > 12:
        raise SchemaValidationError("observations must be a list with at most 12 items")

    observations: list[VisualObservation] = []
    for index, raw in enumerate(raw_observations):
        if not isinstance(raw, dict) or set(raw) != OBSERVATION_KEYS:
            raise SchemaValidationError(f"observation {index} has invalid keys")
        attribute = _short_text(raw["attribute"], f"observations[{index}].attribute", max_length=40)
        severity = _short_text(raw["severity"], f"observations[{index}].severity", max_length=12)
        if attribute not in ALLOWED_ATTRIBUTES:
            raise SchemaValidationError(f"observation {index} uses an unsupported attribute")
        if severity not in ALLOWED_SEVERITIES:
            raise SchemaValidationError(f"observation {index} uses an unsupported severity")
        evidence_ids = raw["evidence_ids"]
        if not isinstance(evidence_ids, list) or not 1 <= len(evidence_ids) <= 3:
            raise SchemaValidationError(f"observation {index} must cite 1-3 evidence IDs")
        if any(not isinstance(item, str) or item not in ALLOWED_EVIDENCE_IDS for item in evidence_ids):
            raise SchemaValidationError(f"observation {index} cites unsupported evidence")
        observations.append(
            VisualObservation(
                attribute=attribute,
                dispatch_value=_short_text(raw["dispatch_value"], f"observations[{index}].dispatch_value"),
                return_value=_short_text(raw["return_value"], f"observations[{index}].return_value"),
                severity=severity,
                evidence_ids=tuple(dict.fromkeys(evidence_ids)),
            )
        )

    raw_missing = payload["missing_evidence"]
    if not isinstance(raw_missing, list) or len(raw_missing) > 8:
        raise SchemaValidationError("missing_evidence must be a list with at most 8 items")
    missing = tuple(_short_text(item, "missing_evidence item", max_length=120) for item in raw_missing)

    return VlmAssessment(
        evidence_sufficient=sufficient,
        same_product_likelihood=_bounded_number(
            payload["same_product_likelihood"], "same_product_likelihood"
        ),
        mismatch_confidence=_bounded_number(payload["mismatch_confidence"], "mismatch_confidence"),
        observations=tuple(observations),
        missing_evidence=missing,
    )


def ground_assessment(
    assessment: VlmAssessment, available_evidence: set[str] | frozenset[str]
) -> VlmAssessment:
    """Drop claims that are not backed by evidence available for this request."""
    available = frozenset(available_evidence) & ALLOWED_EVIDENCE_IDS
    grounded = tuple(
        observation
        for observation in assessment.observations
        if set(observation.evidence_ids).issubset(available)
    )
    removed = len(assessment.observations) - len(grounded)
    sufficient = assessment.evidence_sufficient and removed == 0
    missing = assessment.missing_evidence
    if removed:
        missing = (*missing, "One or more model observations lacked available evidence")
    return VlmAssessment(
        evidence_sufficient=sufficient,
        same_product_likelihood=assessment.same_product_likelihood,
        mismatch_confidence=assessment.mismatch_confidence if grounded else 0.0,
        observations=grounded,
        missing_evidence=missing,
    )
