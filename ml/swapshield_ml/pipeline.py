"""Composable end-to-end verification pipeline with no autonomous adverse action."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any, Callable, Protocol

from .contracts import Decision, RiskFeatures, RiskResult
from .quality import ImageQuality, assess_image_quality
from .risk import score_return
from .schemas import VlmAssessment, ground_assessment


class PairEncoder(Protocol):
    def compare(self, dispatch_images: list[Any], return_images: list[Any]) -> float: ...


class VisualVerifier(Protocol):
    def verify(self, dispatch_image: Any, return_image: Any) -> VlmAssessment: ...


@dataclass(frozen=True, slots=True)
class VerificationRequest:
    dispatch_image: Any
    return_image: Any
    dispatch_sku: str
    return_sku: str
    dispatch_serial: str | None = None
    return_serial: str | None = None
    dispatch_weight_grams: float | None = None
    return_weight_grams: float | None = None


@dataclass(frozen=True, slots=True)
class VerificationOutput:
    risk: RiskResult
    features: RiskFeatures
    vlm_assessment: VlmAssessment
    dispatch_quality: ImageQuality
    return_quality: ImageQuality
    evidence_sources: tuple[str, ...]
    policy_note: str = "Recommendation only; a human confirms every adverse outcome."

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk": {
                "probability": self.risk.probability,
                "score": self.risk.score,
                "decision": self.risk.decision.value,
                "reasons": list(self.risk.reasons),
            },
            "features": asdict(self.features),
            "vlm_assessment": self.vlm_assessment.to_dict(),
            "quality": {
                "dispatch": self.dispatch_quality.to_dict(),
                "return": self.return_quality.to_dict(),
            },
            "evidence_sources": list(self.evidence_sources),
            "policy_note": self.policy_note,
        }


def _normalize_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = "".join(character for character in value.upper() if character.isalnum())
    return normalized or None


def identifier_mismatch(request: VerificationRequest) -> float:
    dispatch_sku = _normalize_identifier(request.dispatch_sku)
    return_sku = _normalize_identifier(request.return_sku)
    if dispatch_sku and return_sku and dispatch_sku != return_sku:
        return 1.0

    dispatch_serial = _normalize_identifier(request.dispatch_serial)
    return_serial = _normalize_identifier(request.return_serial)
    if dispatch_serial and return_serial:
        return 0.0 if dispatch_serial == return_serial else 1.0
    if dispatch_serial or return_serial:
        return 0.35
    return 0.0


def relative_weight_delta(request: VerificationRequest) -> float:
    dispatch = request.dispatch_weight_grams
    returned = request.return_weight_grams
    if dispatch is None or returned is None:
        return 0.0
    if dispatch <= 0 or returned < 0:
        raise ValueError("weights must be non-negative and dispatch weight must be greater than zero")
    return abs(returned - dispatch) / dispatch


class VerifierPipeline:
    def __init__(
        self,
        pair_encoder: PairEncoder,
        visual_verifier: VisualVerifier,
        *,
        threshold: float = 0.68,
        min_image_quality: float = 0.46,
        quality_fn: Any = assess_image_quality,
        risk_fn: Callable[[RiskFeatures], RiskResult] | None = None,
    ) -> None:
        self.pair_encoder = pair_encoder
        self.visual_verifier = visual_verifier
        self.threshold = threshold
        self.min_image_quality = min_image_quality
        self.quality_fn = quality_fn
        self.risk_fn = risk_fn or (
            lambda features: score_return(
                features,
                threshold=self.threshold,
                min_image_quality=self.min_image_quality,
            )
        )
        self._inference_lock = Lock()

    def verify(self, request: VerificationRequest) -> VerificationOutput:
        dispatch_quality = self.quality_fn(request.dispatch_image)
        return_quality = self.quality_fn(request.return_image)
        quality = min(dispatch_quality.score, return_quality.score)

        # GPU model calls are serialized to avoid concurrent requests exhausting
        # the 8 GB card during a live demo.
        with self._inference_lock:
            similarity = self.pair_encoder.compare([request.dispatch_image], [request.return_image])
            raw_assessment = self.visual_verifier.verify(request.dispatch_image, request.return_image)

        assessment = ground_assessment(raw_assessment, {"dispatch_image", "return_image"})
        vlm_mismatch = max(
            assessment.mismatch_confidence,
            1.0 - assessment.same_product_likelihood,
        )
        features = RiskFeatures(
            vision_similarity=similarity,
            vlm_mismatch=vlm_mismatch,
            serial_mismatch=identifier_mismatch(request),
            weight_delta=relative_weight_delta(request),
            image_quality=quality,
        )
        risk = self.risk_fn(features)

        if not assessment.evidence_sufficient and risk.decision is not Decision.RECAPTURE:
            missing = "; ".join(assessment.missing_evidence) or "clearer product images"
            risk = RiskResult(
                probability=risk.probability,
                score=risk.score,
                decision=Decision.RECAPTURE,
                reasons=(*risk.reasons, f"Visual verifier requested more evidence: {missing}"),
            )

        evidence_sources = ["dispatch_image", "return_image", "dispatch_record", "return_record"]
        if request.dispatch_weight_grams is not None and request.return_weight_grams is not None:
            evidence_sources.append("weight_record")

        return VerificationOutput(
            risk=risk,
            features=features,
            vlm_assessment=assessment,
            dispatch_quality=dispatch_quality,
            return_quality=return_quality,
            evidence_sources=tuple(evidence_sources),
        )
