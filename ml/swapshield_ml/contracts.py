from dataclasses import dataclass
from enum import StrEnum


class Decision(StrEnum):
    APPROVE = "approve"
    REVIEW = "review"
    RECAPTURE = "recapture"


@dataclass(frozen=True, slots=True)
class RiskFeatures:
    vision_similarity: float
    vlm_mismatch: float
    serial_mismatch: float
    weight_delta: float
    image_quality: float

    def validate(self) -> None:
        bounded = {
            "vision_similarity": self.vision_similarity,
            "vlm_mismatch": self.vlm_mismatch,
            "serial_mismatch": self.serial_mismatch,
            "image_quality": self.image_quality,
        }
        for name, value in bounded.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.weight_delta < 0.0:
            raise ValueError("weight_delta must be non-negative")


@dataclass(frozen=True, slots=True)
class RiskResult:
    probability: float
    score: int
    decision: Decision
    reasons: tuple[str, ...]

