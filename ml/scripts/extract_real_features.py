"""Run the real GPU verifier over a validated manifest and cache predictions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from ml.swapshield_ml.pipeline import VerificationRequest, VerifierPipeline
from ml.swapshield_ml.real_dataset import load_real_manifest, validate_real_manifest
from ml.swapshield_ml.vision import DinoPairEncoder
from ml.swapshield_ml.vlm import QwenVisualVerifier


def _existing_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                seen.add(str(json.loads(line)["case_id"]))
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache DINOv2 + Qwen3-VL predictions for real pairs")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", action="append", choices=["train", "validation", "test"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()

    if args.output.exists() and not args.resume:
        raise SystemExit(f"{args.output} already exists; remove it or pass --resume")
    records = load_real_manifest(args.manifest)
    validate_real_manifest(records, dataset_root=args.manifest.parent, check_files=True)
    selected_splits = set(args.split or ["train", "validation"])
    selected = [record for record in records if record.split in selected_splits]
    completed = _existing_case_ids(args.output) if args.resume else set()
    selected = [record for record in selected if record.case_id not in completed]
    if args.max_cases is not None:
        selected = selected[: args.max_cases]

    from PIL import Image

    vision_model = os.getenv("SWAPSHIELD_VISION_MODEL", "facebook/dinov2-small")
    vlm_model = os.getenv("SWAPSHIELD_VLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
    pipeline = VerifierPipeline(
        DinoPairEncoder(vision_model, os.getenv("SWAPSHIELD_DEVICE", "cuda")),
        QwenVisualVerifier(
            vlm_model,
            four_bit=os.getenv("SWAPSHIELD_VLM_4BIT", "true").lower() == "true",
            max_pixels=int(os.getenv("SWAPSHIELD_VLM_MAX_PIXELS", "262144")),
            oom_retry_pixels=int(os.getenv("SWAPSHIELD_VLM_OOM_RETRY_PIXELS", "131072")),
        ),
        threshold=float(os.getenv("SWAPSHIELD_REVIEW_THRESHOLD", "0.68")),
        min_image_quality=float(os.getenv("SWAPSHIELD_MIN_IMAGE_QUALITY", "0.46")),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = args.manifest.parent
    with args.output.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(selected, start=1):
            with Image.open(root / record.dispatch_image) as dispatch_source:
                dispatch = dispatch_source.convert("RGB")
            with Image.open(root / record.return_image) as return_source:
                returned = return_source.convert("RGB")

            dispatch_sku = record.dispatch_sku or "UNSPECIFIED-SKU"
            return_sku = record.return_sku or dispatch_sku
            started = perf_counter()
            output = pipeline.verify(
                VerificationRequest(
                    dispatch_image=dispatch,
                    return_image=returned,
                    dispatch_sku=dispatch_sku,
                    return_sku=return_sku,
                    dispatch_serial=record.dispatch_serial,
                    return_serial=record.return_serial,
                    dispatch_weight_grams=record.dispatch_weight_grams,
                    return_weight_grams=record.return_weight_grams,
                )
            )
            latency_ms = (perf_counter() - started) * 1000
            payload = {
                "case_id": record.case_id,
                "split": record.split,
                "label": record.label,
                "target": record.target,
                "category": record.category,
                "slices": list(record.slices),
                "probability": output.risk.probability,
                "decision": output.risk.decision.value,
                "latency_ms": round(latency_ms, 3),
                "features": output.to_dict()["features"],
                "quality": output.to_dict()["quality"],
                "models": {"vision": vision_model, "vlm": vlm_model},
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            print(f"[{index}/{len(selected)}] cached {record.case_id} -> {output.risk.decision.value}")


if __name__ == "__main__":
    main()
