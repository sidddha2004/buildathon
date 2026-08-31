"""Run one local dispatch/return comparison from the command line."""

from __future__ import annotations

import argparse
import json
import os

from ml.swapshield_ml.pipeline import VerificationRequest, VerifierPipeline
from ml.swapshield_ml.vision import DinoPairEncoder
from ml.swapshield_ml.vlm import QwenVisualVerifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a dispatch image with a returned-item image")
    parser.add_argument("dispatch_image")
    parser.add_argument("return_image")
    parser.add_argument("--dispatch-sku", required=True)
    parser.add_argument("--return-sku", required=True)
    parser.add_argument("--dispatch-serial")
    parser.add_argument("--return-serial")
    parser.add_argument("--dispatch-weight", type=float)
    parser.add_argument("--return-weight", type=float)
    args = parser.parse_args()

    from PIL import Image

    dispatch = Image.open(args.dispatch_image).convert("RGB")
    returned = Image.open(args.return_image).convert("RGB")
    pipeline = VerifierPipeline(
        DinoPairEncoder(
            os.getenv("SWAPSHIELD_VISION_MODEL", "facebook/dinov2-small"),
            os.getenv("SWAPSHIELD_DEVICE", "cuda"),
        ),
        QwenVisualVerifier(
            os.getenv("SWAPSHIELD_VLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct"),
            four_bit=os.getenv("SWAPSHIELD_VLM_4BIT", "true").lower() == "true",
            max_pixels=int(os.getenv("SWAPSHIELD_VLM_MAX_PIXELS", "262144")),
            oom_retry_pixels=int(os.getenv("SWAPSHIELD_VLM_OOM_RETRY_PIXELS", "131072")),
        ),
        threshold=float(os.getenv("SWAPSHIELD_REVIEW_THRESHOLD", "0.68")),
        min_image_quality=float(os.getenv("SWAPSHIELD_MIN_IMAGE_QUALITY", "0.46")),
    )
    result = pipeline.verify(
        VerificationRequest(
            dispatch_image=dispatch,
            return_image=returned,
            dispatch_sku=args.dispatch_sku,
            return_sku=args.return_sku,
            dispatch_serial=args.dispatch_serial,
            return_serial=args.return_serial,
            dispatch_weight_grams=args.dispatch_weight,
            return_weight_grams=args.return_weight,
        )
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
