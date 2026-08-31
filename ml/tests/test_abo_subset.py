import tempfile
import unittest
from pathlib import Path

from ml.swapshield_ml.abo_subset import (
    ABOSubsetError,
    ListingProduct,
    SpinView,
    build_manifest_records,
    evenly_spaced_views,
    image_object_key,
    select_products,
)
from ml.swapshield_ml.real_dataset import RealPairRecord, validate_real_manifest


def views(spin_id: str, count: int = 6) -> list[SpinView]:
    return [
        SpinView(index * (360 / count), f"{spin_id}-image-{index}", f"aa/{spin_id}/{index}.jpg")
        for index in range(count)
    ]


class ABOSubsetTests(unittest.TestCase):
    def test_even_views_cover_the_spin(self) -> None:
        chosen = evenly_spaced_views(views("SPIN", 6), 3)
        self.assertEqual([item.azimuth for item in chosen], [0, 120, 240])

    def test_selection_and_pairs_are_deterministic_and_leakage_free(self) -> None:
        categories = ("CHAIR", "LAMP")
        listings = []
        spin_views = {}
        for category in categories:
            for index in range(12):
                item_id = f"{category}-{index:02d}"
                spin_id = f"SPIN-{item_id}"
                listings.append(ListingProduct(item_id, spin_id, category))
                spin_views[spin_id] = views(spin_id)

        first = select_products(
            listings,
            spin_views,
            categories=categories,
            item_count=20,
            views_per_item=3,
            seed=5050,
        )
        second = select_products(
            listings,
            spin_views,
            categories=categories,
            item_count=20,
            views_per_item=3,
            seed=5050,
        )
        self.assertEqual(first, second)
        raw_records = build_manifest_records(first)
        records = [RealPairRecord.from_mapping(raw, line_number=index) for index, raw in enumerate(raw_records, 1)]
        summary = validate_real_manifest(records)
        self.assertEqual(summary["cases"], 40)
        self.assertEqual(summary["identities"], 20)
        self.assertEqual(summary["splits"]["test"]["genuine"], 4)
        self.assertEqual(summary["splits"]["test"]["substitution"], 4)
        for raw in raw_records:
            self.assertNotIn("dispatch_sku", raw)
            self.assertNotIn("return_sku", raw)

    def test_download_paths_cannot_escape_dataset(self) -> None:
        malicious = SpinView(0, "bad", "../../secret.jpg")
        with self.assertRaisesRegex(ABOSubsetError, "unsafe ABO object path"):
            image_object_key(malicious)

    def test_existing_files_can_be_validated(self) -> None:
        categories = ("CHAIR",)
        listings = []
        spin_views = {}
        for index in range(6):
            item_id = f"ITEM-{index}"
            spin_id = f"SPIN-{index}"
            listings.append(ListingProduct(item_id, spin_id, "CHAIR"))
            spin_views[spin_id] = views(spin_id, 3)
        products = select_products(
            listings,
            spin_views,
            categories=categories,
            item_count=6,
            views_per_item=3,
            seed=1,
        )
        raw_records = build_manifest_records(products)
        records = [RealPairRecord.from_mapping(raw, line_number=index) for index, raw in enumerate(raw_records, 1)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for record in records:
                for relative in (record.dispatch_image, record.return_image):
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
            summary = validate_real_manifest(records, dataset_root=root, check_files=True)
            self.assertEqual(summary["cases"], 12)


if __name__ == "__main__":
    unittest.main()
