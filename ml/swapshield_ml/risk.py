from math import exp

from .contracts import Decision, RiskFeatures, RiskResult


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def score_return(
    features: RiskFeatures,
    threshold: float = 0.68,
    min_image_quality: float = 0.46,
) -> RiskResult:
    """Score a return with the deterministic POC baseline.

    This is intentionally a transparent baseline. It will be replaced by the
    calibrated fusion model after real DINOv2/Qwen3-VL features are available.
    """
    features.validate()
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not 0.0 <= min_image_quality <= 1.0:
        raise ValueError("min_image_quality must be between 0 and 1")

    logit = (
        -3.15
        + (1.0 - features.vision_similarity) * 4.1
        + features.vlm_mismatch * 2.75
        + features.serial_mismatch * 2.3
        + min(features.weight_delta / 0.35, 1.0) * 1.65
    )
    probability = _sigmoid(logit)
    reasons: list[str] = []
    if features.vision_similarity < 0.68:
        reasons.append("Visual identity is materially different from dispatch evidence")
    if features.vlm_mismatch > 0.62:
        reasons.append("Vision-language verifier found product-level discrepancies")
    if features.serial_mismatch > 0.70:
        reasons.append("Serial or model identifier does not match the order record")
    if features.weight_delta > 0.18:
        reasons.append("Parcel weight differs beyond the configured tolerance")
    if features.image_quality < min_image_quality:
        reasons.append("Image quality is below the evidence threshold; capture clearer product photos")
    if not reasons:
        reasons.append("All available return evidence is consistent with dispatch")

    if features.image_quality < min_image_quality:
        decision = Decision.RECAPTURE
    elif probability >= threshold:
        decision = Decision.REVIEW
    else:
        decision = Decision.APPROVE

    return RiskResult(
        probability=probability,
        score=round(probability * 100),
        decision=decision,
        reasons=tuple(reasons),
    )
