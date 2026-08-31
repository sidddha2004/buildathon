"""Fast, local image-quality checks used to trigger safe abstention."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ImageQuality:
    score: float
    brightness: float
    contrast: float
    sharpness: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def assess_image_quality(image: Any) -> ImageQuality:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional GPU/runtime dependency
        raise RuntimeError("Install numpy before running image-quality checks") from exc

    gray = np.asarray(image.convert("L").resize((384, 384)), dtype=np.float32)
    mean = float(gray.mean())
    std = float(gray.std())
    gradient_x = float(np.abs(np.diff(gray, axis=1)).mean())
    gradient_y = float(np.abs(np.diff(gray, axis=0)).mean())

    brightness = max(0.0, min(1.0, 1.0 - abs(mean - 127.5) / 127.5))
    contrast = max(0.0, min(1.0, std / 55.0))
    sharpness = max(0.0, min(1.0, (gradient_x + gradient_y) / 28.0))
    # A weak dimension should meaningfully lower confidence without letting one
    # heuristic alone force a recapture.
    score = 0.30 * brightness + 0.30 * contrast + 0.40 * sharpness
    return ImageQuality(
        score=round(score, 6),
        brightness=round(brightness, 6),
        contrast=round(contrast, 6),
        sharpness=round(sharpness, 6),
    )
