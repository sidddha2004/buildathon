import json
import unittest

from ml.swapshield_ml.auditor import (
    AuditorValidationError,
    EvidenceAuditor,
    deterministic_audit,
    parse_auditor_json,
)


def verification(decision: str = "review", sufficient: bool = True) -> dict:
    return {
        "risk": {"probability": 0.96, "decision": decision, "reasons": ["Visual mismatch"]},
        "features": {"vision_similarity": 0.4, "vlm_mismatch": 0.9},
        "vlm_assessment": {
            "evidence_sufficient": sufficient,
            "same_product_likelihood": 0.08,
            "mismatch_confidence": 0.92,
            "observations": [],
            "missing_evidence": [] if sufficient else ["Clear rear label image"],
        },
        "evidence_sources": ["dispatch_image", "return_image", "dispatch_record"],
    }


def valid_response() -> str:
    return json.dumps(
        {
            "recommendation_support": "supported",
            "evidence_consistent": True,
            "contradictions": [],
            "missing_evidence": [],
            "reviewer_summary": "The supplied mismatch evidence supports human review.",
            "checked_evidence_ids": ["dispatch_image", "return_image"],
        }
    )


class ScriptedAuditor(EvidenceAuditor):
    def __init__(self, response: str) -> None:
        super().__init__(endpoint="https://example.invalid/v1/chat/completions", model="auditor-model")
        self.response = response

    def _request(self, compact):
        return self.response


class AuditorTests(unittest.TestCase):
    def test_valid_structured_audit_is_accepted(self) -> None:
        assessment = parse_auditor_json(valid_response(), {"dispatch_image", "return_image"})
        self.assertEqual(assessment.recommendation_support, "supported")

    def test_extra_action_field_is_rejected(self) -> None:
        raw = json.loads(valid_response())
        raw["reject_refund"] = True
        with self.assertRaises(AuditorValidationError):
            parse_auditor_json(json.dumps(raw), {"dispatch_image", "return_image"})

    def test_unavailable_evidence_id_is_rejected(self) -> None:
        raw = json.loads(valid_response())
        raw["checked_evidence_ids"] = ["customer_history"]
        with self.assertRaises(AuditorValidationError):
            parse_auditor_json(json.dumps(raw), {"dispatch_image", "return_image"})

    def test_accusatory_language_is_rejected(self) -> None:
        raw = json.loads(valid_response())
        raw["reviewer_summary"] = "The customer committed fraud."
        with self.assertRaises(AuditorValidationError):
            parse_auditor_json(json.dumps(raw), {"dispatch_image", "return_image"})

    def test_recapture_fallback_preserves_missing_evidence(self) -> None:
        assessment = deterministic_audit(verification("recapture", sufficient=False))
        self.assertEqual(assessment.recommendation_support, "needs_more_evidence")
        self.assertIn("Clear rear label image", assessment.missing_evidence)

    def test_unconfigured_auditor_uses_safe_fallback(self) -> None:
        assessment = EvidenceAuditor().audit(verification())
        self.assertEqual(assessment.source, "deterministic_fallback")
        self.assertEqual(assessment.api_status, "not_configured")

    def test_configured_auditor_marks_successful_api_result(self) -> None:
        assessment = ScriptedAuditor(valid_response()).audit(verification())
        self.assertEqual(assessment.source, "llm_api")
        self.assertEqual(assessment.api_status, "used")
        self.assertEqual(assessment.model, "auditor-model")

    def test_invalid_api_result_falls_back_without_stopping_verification(self) -> None:
        assessment = ScriptedAuditor("not json").audit(verification())
        self.assertEqual(assessment.source, "deterministic_fallback")
        self.assertEqual(assessment.api_status, "failed")


if __name__ == "__main__":
    unittest.main()
