import json
import unittest

from ml.swapshield_ml.schemas import (
    SchemaValidationError,
    VisualObservation,
    VlmAssessment,
    ground_assessment,
    parse_vlm_json,
)


def valid_payload() -> dict[str, object]:
    return {
        "evidence_sufficient": True,
        "same_product_likelihood": 0.18,
        "mismatch_confidence": 0.91,
        "observations": [
            {
                "attribute": "model_text",
                "dispatch_value": "WH-1000XM5",
                "return_value": "WH-1000XM4",
                "severity": "material",
                "evidence_ids": ["dispatch_image", "return_image"],
            }
        ],
        "missing_evidence": [],
    }


class VlmSchemaTests(unittest.TestCase):
    def test_valid_json_and_json_fence_are_accepted(self) -> None:
        raw = json.dumps(valid_payload())
        self.assertEqual(parse_vlm_json(raw).mismatch_confidence, 0.91)
        self.assertEqual(parse_vlm_json(f"```json\n{raw}\n```").observations[0].attribute, "model_text")

    def test_action_taking_field_is_rejected(self) -> None:
        payload = valid_payload()
        payload["action"] = "reject_refund"
        with self.assertRaises(SchemaValidationError):
            parse_vlm_json(json.dumps(payload))

    def test_batch_mode_fills_only_missing_empty_lists(self) -> None:
        payload = valid_payload()
        del payload["missing_evidence"]
        with self.assertRaises(SchemaValidationError):
            parse_vlm_json(json.dumps(payload))
        assessment = parse_vlm_json(json.dumps(payload), allow_missing_empty_lists=True)
        self.assertEqual(assessment.missing_evidence, ())

    def test_batch_mode_does_not_fill_required_scalar_fields(self) -> None:
        payload = valid_payload()
        del payload["mismatch_confidence"]
        with self.assertRaises(SchemaValidationError):
            parse_vlm_json(json.dumps(payload), allow_missing_empty_lists=True)

    def test_unsupported_evidence_reference_is_rejected(self) -> None:
        payload = valid_payload()
        payload["observations"][0]["evidence_ids"] = ["customer_profile"]  # type: ignore[index]
        with self.assertRaises(SchemaValidationError):
            parse_vlm_json(json.dumps(payload))

    def test_grounder_drops_claim_without_available_evidence(self) -> None:
        assessment = VlmAssessment(
            evidence_sufficient=True,
            same_product_likelihood=0.3,
            mismatch_confidence=0.8,
            observations=(
                VisualObservation("brand", "Sony", "Soni", "material", ("return_image",)),
            ),
            missing_evidence=(),
        )
        grounded = ground_assessment(assessment, {"dispatch_image"})
        self.assertFalse(grounded.evidence_sufficient)
        self.assertEqual(grounded.observations, ())
        self.assertEqual(grounded.mismatch_confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
