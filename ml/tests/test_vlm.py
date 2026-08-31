import json
import unittest

from ml.swapshield_ml.vlm import QwenVisualVerifier, VlmCudaOutOfMemoryError


def valid_response() -> str:
    return json.dumps(
        {
            "evidence_sufficient": True,
            "same_product_likelihood": 0.9,
            "mismatch_confidence": 0.1,
            "observations": [],
            "missing_evidence": [],
        }
    )


class ScriptedVerifier(QwenVisualVerifier):
    def __init__(self, responses: list[str | Exception], *, retries: int = 1) -> None:
        super().__init__(schema_retries=retries)
        self.responses = responses
        self.calls = 0

    def _generate(self, messages):
        response = self.responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


class VlmReliabilityTests(unittest.TestCase):
    def test_schema_failure_is_retried_once(self) -> None:
        verifier = ScriptedVerifier(["not json", valid_response()])
        assessment = verifier.verify(object(), object())
        self.assertTrue(assessment.evidence_sufficient)
        self.assertEqual(verifier.calls, 2)

    def test_repeated_schema_failure_becomes_safe_recapture_evidence(self) -> None:
        verifier = ScriptedVerifier(["not json", '{"wrong": true}'])
        assessment = verifier.verify(object(), object())
        self.assertFalse(assessment.evidence_sufficient)
        self.assertEqual(assessment.mismatch_confidence, 0.0)
        self.assertIn("structured validation", assessment.missing_evidence[0])

    def test_cuda_oom_is_retried_at_lower_pixel_budget(self) -> None:
        verifier = ScriptedVerifier([VlmCudaOutOfMemoryError("oom"), valid_response()])
        assessment = verifier.verify(object(), object())
        self.assertTrue(assessment.evidence_sufficient)
        self.assertEqual(verifier.calls, 2)

    def test_repeated_cuda_oom_becomes_safe_recapture_evidence(self) -> None:
        verifier = ScriptedVerifier(
            [VlmCudaOutOfMemoryError("oom"), VlmCudaOutOfMemoryError("oom again")]
        )
        assessment = verifier.verify(object(), object())
        self.assertFalse(assessment.evidence_sufficient)
        self.assertIn("GPU memory", assessment.missing_evidence[0])

    def test_large_image_is_resized_to_pixel_budget(self) -> None:
        from PIL import Image

        verifier = ScriptedVerifier([valid_response()])
        resized = verifier._resize_for_pixel_budget(Image.new("RGB", (2_000, 1_000)), 262_144)
        self.assertLessEqual(resized.width * resized.height, 262_144)
        self.assertAlmostEqual(resized.width / resized.height, 2.0, places=1)


if __name__ == "__main__":
    unittest.main()
