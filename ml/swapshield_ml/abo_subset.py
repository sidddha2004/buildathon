"""Bounded Amazon Berkeley Objects subset builder for the real benchmark.

The builder downloads the small public metadata files, selects item-disjoint
360-degree product identities, and fetches only the image views needed by the
generated manifest. It intentionally never downloads the 40 GB spin archive.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import random
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from ml.swapshield_ml.real_dataset import load_real_manifest, validate_real_manifest


ABO_BASE_URL = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/"
ABO_DATASET_URL = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html"
ABO_SOURCE = "Amazon Berkeley Objects (ABO)"
ABO_LICENSE = "CC BY 4.0"
LISTING_SHARDS = tuple(f"listings/metadata/listings_{suffix}.json.gz" for suffix in "0123456789abcdef")
SPIN_METADATA_KEY = "spins/metadata/spins.csv.gz"
DEFAULT_CATEGORIES = ("CHAIR", "SOFA", "TABLE", "LAMP")
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class ABOSubsetError(RuntimeError):
    """Raised when a safe, balanced ABO subset cannot be built."""


@dataclass(frozen=True, slots=True)
class ListingProduct:
    item_id: str
    spin_id: str
    category: str


@dataclass(frozen=True, slots=True)
class SpinView:
    azimuth: float
    image_id: str
    path: str


@dataclass(frozen=True, slots=True)
class SelectedProduct:
    item_id: str
    spin_id: str
    category: str
    split: str
    views: tuple[SpinView, ...]


def _safe_remote_key(key: str, *, expected_suffixes: set[str] | None = None) -> str:
    normalized = PurePosixPath(key)
    if normalized.is_absolute() or ".." in normalized.parts or not normalized.parts:
        raise ABOSubsetError(f"unsafe ABO object path: {key!r}")
    if expected_suffixes and normalized.suffix.lower() not in expected_suffixes:
        raise ABOSubsetError(f"unsupported ABO object type: {key!r}")
    return normalized.as_posix()


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    if not cleaned:
        raise ABOSubsetError(f"unsafe empty identifier derived from {value!r}")
    return cleaned[:120]


def object_url(key: str) -> str:
    safe_key = _safe_remote_key(key)
    return ABO_BASE_URL + quote(safe_key, safe="/")


def download_file(
    key: str,
    destination: Path,
    *,
    retries: int = 3,
    timeout: int = 60,
) -> bool:
    """Download one public ABO object atomically; return False when cached."""

    _safe_remote_key(key)
    if destination.is_file() and destination.stat().st_size > 0:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = Request(object_url(key), headers={"User-Agent": "SwapShield-ABO-Subset/0.4"})
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response, partial.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            if partial.stat().st_size == 0:
                raise ABOSubsetError(f"empty download for {key}")
            partial.replace(destination)
            return True
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(attempt)
    raise ABOSubsetError(f"failed to download {key} after {retries} attempts") from last_error


def _english_value(values: Any) -> str | None:
    if not isinstance(values, list):
        return None
    fallback: str | None = None
    for entry in values:
        if not isinstance(entry, dict) or not isinstance(entry.get("value"), str):
            continue
        value = " ".join(entry["value"].split())
        if not value:
            continue
        fallback = fallback or value
        if str(entry.get("language_tag", "")).lower().startswith("en"):
            return value
    return fallback


def parse_listing_files(paths: Iterable[Path], categories: Sequence[str]) -> list[ListingProduct]:
    wanted = {category.upper() for category in categories}
    products: dict[str, ListingProduct] = {}
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ABOSubsetError(f"invalid JSON in {path.name}:{line_number}") from exc
                item_id = raw.get("item_id")
                spin_id = raw.get("spin_id")
                category = _english_value(raw.get("product_type"))
                if not all(isinstance(value, str) and value for value in (item_id, spin_id, category)):
                    continue
                normalized_category = category.upper()
                if normalized_category in wanted and item_id not in products:
                    products[item_id] = ListingProduct(item_id, spin_id, normalized_category)
    return sorted(products.values(), key=lambda item: item.item_id)


def parse_spin_views(path: Path, spin_ids: set[str]) -> dict[str, list[SpinView]]:
    views: dict[str, list[SpinView]] = {spin_id: [] for spin_id in spin_ids}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"spin_id", "azimuth", "image_id", "path"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ABOSubsetError("ABO spin metadata schema is missing required columns")
        for raw in reader:
            spin_id = raw["spin_id"]
            if spin_id not in views:
                continue
            relative = _safe_remote_key(raw["path"], expected_suffixes=ALLOWED_IMAGE_SUFFIXES)
            try:
                azimuth = float(raw["azimuth"])
            except ValueError as exc:
                raise ABOSubsetError(f"invalid azimuth for spin {spin_id}") from exc
            views[spin_id].append(SpinView(azimuth, raw["image_id"], relative))
    for spin_id in views:
        views[spin_id].sort(key=lambda view: (view.azimuth, view.image_id))
    return views


def evenly_spaced_views(views: Sequence[SpinView], count: int) -> tuple[SpinView, ...]:
    if count < 2:
        raise ABOSubsetError("at least two views per product are required")
    if len(views) < count:
        raise ABOSubsetError(f"product has {len(views)} views; {count} required")
    indices = [math.floor(index * len(views) / count) for index in range(count)]
    return tuple(views[index] for index in indices)


def _category_targets(item_count: int, categories: Sequence[str]) -> dict[str, int]:
    normalized = tuple(dict.fromkeys(category.upper() for category in categories))
    if not normalized:
        raise ABOSubsetError("at least one category is required")
    if item_count < len(normalized) * 6:
        raise ABOSubsetError(
            f"--items must be at least {len(normalized) * 6} so every category has two identities per split"
        )
    base, remainder = divmod(item_count, len(normalized))
    return {category: base + int(index < remainder) for index, category in enumerate(normalized)}


def select_products(
    listings: Sequence[ListingProduct],
    views_by_spin: dict[str, list[SpinView]],
    *,
    categories: Sequence[str],
    item_count: int,
    views_per_item: int,
    seed: int,
) -> list[SelectedProduct]:
    targets = _category_targets(item_count, categories)
    by_category: dict[str, list[ListingProduct]] = {category: [] for category in targets}
    for product in listings:
        if product.category in by_category and len(views_by_spin.get(product.spin_id, ())) >= views_per_item:
            by_category[product.category].append(product)

    selected: list[SelectedProduct] = []
    for category, target in targets.items():
        candidates = sorted(by_category[category], key=lambda item: item.item_id)
        random.Random(f"{seed}:{category}").shuffle(candidates)
        if len(candidates) < target:
            raise ABOSubsetError(
                f"category {category} has only {len(candidates)} eligible spin products; {target} requested"
            )
        chosen = candidates[:target]
        validation_count = max(2, round(target * 0.15))
        test_count = max(2, round(target * 0.15))
        train_count = target - validation_count - test_count
        if train_count < 2:
            raise ABOSubsetError(f"category {category} does not have enough training identities")
        split_names = (
            ["train"] * train_count
            + ["validation"] * validation_count
            + ["test"] * test_count
        )
        for product, split in zip(chosen, split_names, strict=True):
            selected.append(
                SelectedProduct(
                    item_id=product.item_id,
                    spin_id=product.spin_id,
                    category=product.category,
                    split=split,
                    views=evenly_spaced_views(views_by_spin[product.spin_id], views_per_item),
                )
            )
    return sorted(selected, key=lambda item: (item.split, item.category, item.item_id))


def image_object_key(view: SpinView) -> str:
    return _safe_remote_key(f"spins/original/{view.path}", expected_suffixes=ALLOWED_IMAGE_SUFFIXES)


def local_image_path(product: SelectedProduct, view: SpinView) -> str:
    suffix = PurePosixPath(view.path).suffix.lower()
    filename = f"{_safe_component(view.image_id)}{suffix}"
    return (Path("images") / product.split / _safe_component(product.item_id) / filename).as_posix()


def build_manifest_records(products: Sequence[SelectedProduct]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[SelectedProduct]] = {}
    for product in products:
        grouped.setdefault((product.split, product.category), []).append(product)

    records: list[dict[str, Any]] = []
    for (split, category), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: item.item_id)
        if len(ordered) < 2:
            raise ABOSubsetError(f"{split}/{category} needs at least two identities")
        for index, product in enumerate(ordered):
            substitute = ordered[(index + 1) % len(ordered)]
            common = {
                "split": split,
                "category": category.lower().replace("_", " "),
                "dispatch_item_id": product.item_id,
                "dispatch_image": local_image_path(product, product.views[0]),
                "source": ABO_SOURCE,
                "source_license": ABO_LICENSE,
            }
            records.append(
                {
                    "case_id": f"ABO-{split[:3].upper()}-{_safe_component(product.item_id)}-G",
                    **common,
                    "label": "genuine",
                    "return_item_id": product.item_id,
                    "return_image": local_image_path(product, product.views[1]),
                    "slices": ["abo-360", "different-angle"],
                }
            )
            records.append(
                {
                    "case_id": f"ABO-{split[:3].upper()}-{_safe_component(product.item_id)}-S",
                    **common,
                    "label": "substitution",
                    "return_item_id": substitute.item_id,
                    "return_image": local_image_path(substitute, substitute.views[-1]),
                    "slices": ["abo-360", "same-category", "hard-negative"],
                }
            )
    return sorted(records, key=lambda item: item["case_id"])


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def download_selected_images(
    products: Sequence[SelectedProduct],
    output_dir: Path,
    *,
    workers: int = 6,
    progress: Callable[[str], None] = print,
) -> tuple[int, int]:
    jobs: dict[tuple[str, str], tuple[str, Path]] = {}
    for product in products:
        for view in product.views:
            relative = local_image_path(product, view)
            jobs[(product.item_id, view.image_id)] = (image_object_key(view), output_dir / relative)

    downloaded = 0
    cached = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 16))) as pool:
        futures = {pool.submit(download_file, key, destination): destination for key, destination in jobs.values()}
        for future in as_completed(futures):
            was_downloaded = future.result()
            downloaded += int(was_downloaded)
            cached += int(not was_downloaded)
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == len(futures):
                progress(f"Images: {completed}/{len(futures)} ({downloaded} downloaded, {cached} cached)")
    return downloaded, cached


def write_attribution(output_dir: Path) -> None:
    text = f"""# Dataset attribution

This local benchmark subset contains images from **{ABO_SOURCE}**, licensed
under **{ABO_LICENSE}**. The subset was selected and paired by SwapShield; it is
not an official Amazon benchmark split.

- Source: {ABO_DATASET_URL}
- License: https://creativecommons.org/licenses/by/4.0/
- Paper: *ABO: Dataset and Benchmarks for Real-World 3D Object Understanding*

Keep this file with any copy or redistribution of the downloaded images.
"""
    (output_dir / "ATTRIBUTION.md").write_text(text, encoding="utf-8")


def materialize_abo_subset(
    output_dir: Path,
    *,
    item_count: int = 120,
    views_per_item: int = 3,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    seed: int = 5050,
    workers: int = 6,
    overwrite_manifest: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists() and not overwrite_manifest:
        raise ABOSubsetError(f"{manifest_path} already exists; pass --overwrite to rebuild it")

    cache_dir = output_dir / ".cache"
    progress("Downloading/caching ABO metadata (about 91 MB total)...")
    listing_paths: list[Path] = []
    for index, key in enumerate(LISTING_SHARDS, start=1):
        destination = cache_dir / PurePosixPath(key).name
        download_file(key, destination)
        listing_paths.append(destination)
        progress(f"Listing metadata: {index}/{len(LISTING_SHARDS)}")
    spin_metadata = cache_dir / PurePosixPath(SPIN_METADATA_KEY).name
    download_file(SPIN_METADATA_KEY, spin_metadata)

    normalized_categories = tuple(dict.fromkeys(category.upper() for category in categories))
    listings = parse_listing_files(listing_paths, normalized_categories)
    if not listings:
        raise ABOSubsetError(f"no listings found for categories {normalized_categories}")
    progress(f"Eligible listing candidates before view checks: {len(listings)}")
    views = parse_spin_views(spin_metadata, {item.spin_id for item in listings})
    products = select_products(
        listings,
        views,
        categories=normalized_categories,
        item_count=item_count,
        views_per_item=views_per_item,
        seed=seed,
    )
    downloaded, cached = download_selected_images(products, output_dir, workers=workers, progress=progress)
    records = build_manifest_records(products)
    write_jsonl(records, manifest_path)
    write_attribution(output_dir)

    parsed = load_real_manifest(manifest_path)
    validation = validate_real_manifest(parsed, dataset_root=output_dir, check_files=True)
    summary = {
        "dataset": ABO_SOURCE,
        "license": ABO_LICENSE,
        "seed": seed,
        "categories": list(normalized_categories),
        "selected_items": len(products),
        "views_per_item": views_per_item,
        "images_downloaded": downloaded,
        "images_reused_from_cache": cached,
        "manifest": manifest_path.name,
        "validation": validation,
        "selection": [
            {
                "item_id": product.item_id,
                "spin_id": product.spin_id,
                "category": product.category,
                "split": product.split,
                "views": [asdict(view) for view in product.views],
            }
            for product in products
        ],
    }
    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
