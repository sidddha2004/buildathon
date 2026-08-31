import unittest

from ml.swapshield_ml.real_dataset import (
    DatasetValidationError,
    RealPairRecord,
    validate_real_manifest,
)


def record(
    case_id: str,
    split: str,
    label: str,
    dispatch_id: str,
    return_id: str | None,
) -> RealPairRecord:
    return RealPairRecord.from_mapping(
        {
            "case_id": case_id,
            "split": split,
            "label": label,
            "category": "computer mouse",
            "dispatch_item_id": dispatch_id,
            "return_item_id": return_id,
            "dispatch_image": f"images/{case_id}-dispatch.jpg",
            "return_image": f"images/{case_id}-return.jpg",
            "source": "self-captured",
            "source_license": "owned",
            "slices": ["different-angle"],
        },
        line_number=1,
    )


class RealDatasetTests(unittest.TestCase):
    def test_valid_item_disjoint_manifest(self) -> None:
        records = [
            record("VAL-G", "validation", "genuine", "VAL-A", "VAL-A"),
            record("VAL-S", "validation", "substitution", "VAL-B", "VAL-C"),
            record("TEST-G", "test", "genuine", "TEST-A", "TEST-A"),
            record("TEST-S", "test", "substitution", "TEST-B", "TEST-C"),
        ]
        summary = validate_real_manifest(records)
        self.assertEqual(summary["cases"], 4)
        self.assertEqual(summary["splits"]["test"]["substitution"], 1)

    def test_identity_leakage_is_rejected(self) -> None:
        records = [
            record("VAL-G", "validation", "genuine", "SHARED", "SHARED"),
            record("VAL-S", "validation", "substitution", "VAL-B", "VAL-C"),
            record("TEST-G", "test", "genuine", "SHARED", "SHARED"),
            record("TEST-S", "test", "substitution", "TEST-B", "TEST-C"),
        ]
        with self.assertRaisesRegex(DatasetValidationError, "identity leakage"):
            validate_real_manifest(records)

    def test_inconsistent_label_is_rejected(self) -> None:
        with self.assertRaisesRegex(DatasetValidationError, "genuine pairs"):
            record("BAD", "validation", "genuine", "ITEM-A", "ITEM-B")

    def test_path_traversal_is_rejected(self) -> None:
        raw = {
            "case_id": "BAD-PATH",
            "split": "validation",
            "label": "genuine",
            "category": "mouse",
            "dispatch_item_id": "ITEM-A",
            "return_item_id": "ITEM-A",
            "dispatch_image": "../secret.jpg",
            "return_image": "images/return.jpg",
            "source": "self-captured",
            "source_license": "owned",
        }
        with self.assertRaisesRegex(DatasetValidationError, "inside the dataset"):
            RealPairRecord.from_mapping(raw, line_number=1)


if __name__ == "__main__":
    unittest.main()
