import unittest

from ml.swapshield_ml.contracts import Decision, RiskFeatures
from ml.swapshield_ml.evaluation import evaluate_seeded_baseline
from ml.swapshield_ml.risk import score_return


class RiskBaselineTests(unittest.TestCase):
    def test_clear_match_is_approved(self) -> None:
        result = score_return(RiskFeatures(0.95, 0.08, 0.0, 0.01, 0.92))
        self.assertEqual(result.decision, Decision.APPROVE)

    def test_multiple_mismatches_route_to_review(self) -> None:
        result = score_return(RiskFeatures(0.55, 0.90, 1.0, 0.30, 0.91))
        self.assertEqual(result.decision, Decision.REVIEW)
        self.assertGreaterEqual(len(result.reasons), 3)

    def test_poor_image_abstains_even_when_risk_is_high(self) -> None:
        result = score_return(RiskFeatures(0.50, 0.85, 0.9, 0.25, 0.20))
        self.assertEqual(result.decision, Decision.RECAPTURE)

    def test_seeded_evaluation_is_reproducible(self) -> None:
        first = evaluate_seeded_baseline()
        second = evaluate_seeded_baseline()
        self.assertEqual(first, second)
        self.assertEqual(first.sample_size, 480)
        self.assertEqual(first.false_positives, 5)
        self.assertEqual(first.false_negatives, 4)


if __name__ == "__main__":
    unittest.main()
