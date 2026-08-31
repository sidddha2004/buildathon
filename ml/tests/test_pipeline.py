import unittest

from ml.swapshield_ml.contracts import Decision
from ml.swapshield_ml.pipeline import VerificationRequest, VerifierPipeline
from ml.swapshield_ml.quality import ImageQuality
from ml.swapshield_ml.schemas import VisualObservation, VlmAssessment


class FakeEncoder:
    def __init__(self, similarity: float) -> None:
        self.similarity = similarity

    def compare(self, dispatch_images, return_images) -> float:
        self.calls = (dispatch_images, return_images)
        return self.similarity


class FakeVerifier:
    def __init__(self, assessment: VlmAssessment) -> None:
        self.assessment = assessment

    def verify(self, dispatch_image, return_image) -> VlmAssessment:
        self.calls = (dispatch_image, return_image)
        return self.assessment


def quality(score: float = 0.9):
    return lambda image: ImageQuality(score, 0.9, 0.9, 0.9)


def assessment(*, sufficient: bool = True, same: float = 0.95, mismatch: float = 0.05):
    observations = ()
    if mismatch > 0.5:
        observations = (
            VisualObservation(
                "model_text",
                "WH-1000XM5",
                "WH-1000XM4",
                "material",
                ("dispatch_image", "return_image"),
            ),
        )
    return VlmAssessment(
        evidence_sufficient=sufficient,
        same_product_likelihood=same,
        mismatch_confidence=mismatch,
        observations=observations,
        missing_evidence=() if sufficient else ("rear serial label",),
    )


def request(**overrides):
    values = {
        "dispatch_image": object(),
        "return_image": object(),
        "dispatch_sku": "SKU-100",
        "return_sku": "SKU-100",
        "dispatch_serial": "ABC-123",
        "return_serial": "ABC-123",
        "dispatch_weight_grams": 1000.0,
        "return_weight_grams": 1010.0,
    }
    values.update(overrides)
    return VerificationRequest(**values)


class PipelineTests(unittest.TestCase):
    def test_clear_match_is_recommended_for_approval(self) -> None:
        pipeline = VerifierPipeline(FakeEncoder(0.96), FakeVerifier(assessment()), quality_fn=quality())
        output = pipeline.verify(request())
        self.assertEqual(output.risk.decision, Decision.APPROVE)
        self.assertEqual(output.evidence_sources[-1], "weight_record")

    def test_independent_mismatches_route_to_human_review(self) -> None:
        pipeline = VerifierPipeline(
            FakeEncoder(0.42),
            FakeVerifier(assessment(same=0.08, mismatch=0.94)),
            quality_fn=quality(),
        )
        output = pipeline.verify(
            request(return_sku="SKU-200", return_serial="XYZ-999", return_weight_grams=690.0)
        )
        self.assertEqual(output.risk.decision, Decision.REVIEW)
        self.assertGreater(output.risk.probability, 0.9)

    def test_insufficient_vlm_evidence_forces_recapture(self) -> None:
        pipeline = VerifierPipeline(
            FakeEncoder(0.96),
            FakeVerifier(assessment(sufficient=False, same=0.96, mismatch=0.04)),
            quality_fn=quality(),
        )
        output = pipeline.verify(request())
        self.assertEqual(output.risk.decision, Decision.RECAPTURE)
        self.assertIn("rear serial label", output.risk.reasons[-1])


if __name__ == "__main__":
    unittest.main()
