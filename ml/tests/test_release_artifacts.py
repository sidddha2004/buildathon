import json
import unittest
from pathlib import Path

from ml.swapshield_ml.fusion import load_fusion_model


ROOT = Path(__file__).resolve().parents[2]


class ReleaseArtifactTests(unittest.TestCase):
    def test_versioned_fusion_model_loads(self) -> None:
        model = load_fusion_model(ROOT / "evaluation" / "results" / "fusion-model.json")
        self.assertEqual(model.train_cases, 176)
        self.assertEqual(model.validation_cases, 32)
        self.assertEqual(model.threshold, 0.92)
        self.assertEqual(model.calibration_folds, 5)

    def test_locked_report_is_internally_consistent(self) -> None:
        report = json.loads(
            (ROOT / "evaluation" / "results" / "real-report.json").read_text(encoding="utf-8")
        )
        test = report["test"]
        confusion = test["confusion"]
        self.assertEqual(sum(confusion.values()), test["cases"])
        self.assertEqual(confusion, {"fn": 2, "fp": 0, "tn": 15, "tp": 13})
        self.assertEqual(
            test["missed_substitution_loss"],
            confusion["fn"] * report["cost_assumptions"]["false_negative_inr"],
        )
        category_cases = sum(
            value["cases"]
            for name, value in test["slices"].items()
            if name.startswith("category:")
        )
        self.assertEqual(category_cases, test["cases"])


if __name__ == "__main__":
    unittest.main()
