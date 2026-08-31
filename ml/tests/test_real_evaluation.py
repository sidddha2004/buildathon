import unittest

from ml.swapshield_ml.real_evaluation import RealPrediction, evaluate_real_predictions


def prediction(case_id: str, split: str, target: int, probability: float, decision: str = "approve"):
    return RealPrediction(case_id, split, target, probability, decision, "mouse", ("clear",), 100.0)


class RealEvaluationTests(unittest.TestCase):
    def test_threshold_is_selected_on_validation_and_applied_to_test(self) -> None:
        records = [
            prediction("VG1", "validation", 0, 0.10),
            prediction("VG2", "validation", 0, 0.20),
            prediction("VS1", "validation", 1, 0.80, "review"),
            prediction("VS2", "validation", 1, 0.90, "review"),
            prediction("TG1", "test", 0, 0.15),
            prediction("TG2", "test", 0, 0.25),
            prediction("TS1", "test", 1, 0.85, "review"),
            prediction("TS2", "test", 1, 0.95, "review"),
        ]
        report = evaluate_real_predictions(records, bootstrap_samples=50)
        self.assertEqual(report["test"]["precision"], 1.0)
        self.assertEqual(report["test"]["recall"], 1.0)
        self.assertEqual(report["threshold_selection"]["split"], "validation")

    def test_recapture_is_reported_separately(self) -> None:
        records = [
            prediction("VG", "validation", 0, 0.1),
            prediction("VS", "validation", 1, 0.9, "review"),
            prediction("TG", "test", 0, 0.1, "recapture"),
            prediction("TS", "test", 1, 0.9, "review"),
        ]
        report = evaluate_real_predictions(records, bootstrap_samples=20)
        self.assertEqual(report["test"]["recapture_rate"], 0.5)
        self.assertEqual(report["test"]["genuine_recapture_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
