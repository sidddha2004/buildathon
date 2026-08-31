from dataclasses import asdict, dataclass
from .contracts import Decision, RiskFeatures
from .risk import score_return


class _SeededRandom:
    """The same 32-bit LCG used by the browser baseline."""

    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFFFFFF

    def random(self) -> float:
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        return self._state / 4294967296


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    sample_size: int
    positive_rate: float
    precision: float
    recall: float
    f1: float
    false_positives: int
    false_negatives: int
    false_positive_cost_inr: int
    missed_loss_inr: int

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def evaluate_seeded_baseline(
    *,
    threshold: float = 0.68,
    seed: int = 5050,
    sample_size: int = 480,
    review_cost_inr: int = 80,
    missed_swap_loss_inr: int = 6200,
) -> EvaluationSummary:
    random = _SeededRandom(seed)
    tp = fp = fn = positives = 0

    for _ in range(sample_size):
        is_swap = random.random() < 0.18
        positives += int(is_swap)
        hard_negative = not is_swap and random.random() < 0.14
        if is_swap:
            features = RiskFeatures(
                vision_similarity=0.43 + random.random() * 0.42,
                vlm_mismatch=0.42 + random.random() * 0.52,
                serial_mismatch=1.0 if random.random() < 0.44 else random.random() * 0.28,
                weight_delta=random.random() * 0.34,
                image_quality=0.52 + random.random() * 0.48,
            )
        else:
            features = RiskFeatures(
                vision_similarity=(0.58 + random.random() * 0.25) if hard_negative else (0.79 + random.random() * 0.20),
                vlm_mismatch=(0.28 + random.random() * 0.42) if hard_negative else random.random() * 0.30,
                serial_mismatch=0.82 if random.random() < 0.025 else random.random() * 0.12,
                weight_delta=random.random() * (0.19 if hard_negative else 0.09),
                image_quality=0.52 + random.random() * 0.48,
            )

        predicted_swap = score_return(features, threshold).decision is Decision.REVIEW
        tp += int(predicted_swap and is_swap)
        fp += int(predicted_swap and not is_swap)
        fn += int(not predicted_swap and is_swap)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    return EvaluationSummary(
        sample_size=sample_size,
        positive_rate=positives / sample_size,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positives=fp,
        false_negatives=fn,
        false_positive_cost_inr=fp * review_cost_inr,
        missed_loss_inr=fn * missed_swap_loss_inr,
    )
