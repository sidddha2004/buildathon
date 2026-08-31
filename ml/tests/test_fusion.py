import json
import tempfile
import unittest
from pathlib import Path

from ml.swapshield_ml.contracts import RiskFeatures
from ml.swapshield_ml.fusion import (
    FusionModel,
    FusionTrainingError,
    load_fusion_model,
    score_feature_rows,
    train_fusion_model,
)


def row(case_id: str, split: str, target: int, strength: float) -> dict:
    return {
        "case_id": case_id,
        "split": split,
        "label": "substitution" if target else "genuine",
        "target": target,
        "category": "chair",
        "slices": ["same-category"],
        "probability": strength,
        "decision": "review" if strength >= 0.5 else "approve",
        "latency_ms": 100.0,
        "features": {
            "vision_similarity": 1.0 - strength * 0.7,
            "vlm_mismatch": strength * 0.9,
            "serial_mismatch": 0.0,
            "weight_delta": 0.0,
            "image_quality": 0.8,
        },
    }


class FusionTests(unittest.TestCase):
    def feature_rows(self) -> list[dict]:
        rows = []
        for split, count in (("train", 40), ("validation", 12)):
            for index in range(count):
                target = index % 2
                strength = (0.72 + (index % 5) * 0.03) if target else (0.12 + (index % 5) * 0.03)
                rows.append(row(f"{split}-{index}", split, target, strength))
        return rows

    def test_training_is_reproducible_and_artifact_round_trips(self) -> None:
        rows = self.feature_rows()
        first, report = train_fusion_model(rows, seed=5050)
        second, _ = train_fusion_model(rows, seed=5050)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertFalse(report["test_accessed"])
        self.assertEqual(report["validation"]["recall"], 1.0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.json"
            path.write_text(json.dumps(first.to_dict()), encoding="utf-8")
            loaded = load_fusion_model(path)
        self.assertIsInstance(loaded, FusionModel)
        self.assertAlmostEqual(
            loaded.probability(rows[0]["features"]),
            first.probability(rows[0]["features"]),
        )
        risk = loaded.score_risk(RiskFeatures(**rows[-1]["features"]))
        self.assertGreaterEqual(risk.probability, 0.5)

    def test_test_rows_are_rejected_during_training(self) -> None:
        rows = self.feature_rows() + [row("locked-test", "test", 1, 0.9)]
        with self.assertRaisesRegex(FusionTrainingError, "contains test cases"):
            train_fusion_model(rows)

    def test_scoring_can_exclude_smoke_cases(self) -> None:
        model, _ = train_fusion_model(self.feature_rows())
        test_rows = [row("keep", "test", 1, 0.9), row("smoke", "test", 0, 0.1)]
        scored = score_feature_rows(model, test_rows, exclude_case_ids={"smoke"})
        self.assertEqual([item["case_id"] for item in scored], ["keep"])
        self.assertIn("baseline_probability", scored[0])


if __name__ == "__main__":
    unittest.main()
