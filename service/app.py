"""FastAPI adapter for the local RTX inference pipeline."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional convenience outside the GPU extra
    pass

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from ml.swapshield_ml.auditor import EvidenceAuditor
from ml.swapshield_ml.pipeline import VerificationRequest, VerifierPipeline
from ml.swapshield_ml.vision import DinoPairEncoder
from ml.swapshield_ml.vlm import QwenVisualVerifier


MAX_UPLOAD_BYTES = int(os.getenv("SWAPSHIELD_MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("SWAPSHIELD_MAX_IMAGE_PIXELS", "30000000"))
_pipeline: VerifierPipeline | None = None
_auditor: EvidenceAuditor | None = None

app = FastAPI(
    title="SwapShield Local Verifier",
    version="1.0.0",
    description="Defense-only return evidence comparison. No endpoint can reject refunds or move money.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def get_pipeline() -> VerifierPipeline:
    global _pipeline
    if _pipeline is None:
        threshold = float(os.getenv("SWAPSHIELD_REVIEW_THRESHOLD", "0.68"))
        min_image_quality = float(os.getenv("SWAPSHIELD_MIN_IMAGE_QUALITY", "0.46"))
        risk_fn = None
        fusion_model_path = os.getenv(
            "SWAPSHIELD_FUSION_MODEL", "evaluation/results/fusion-model.json"
        ).strip()
        if fusion_model_path:
            from ml.swapshield_ml.fusion import load_fusion_model

            fusion_model = load_fusion_model(fusion_model_path)
            threshold = fusion_model.threshold
            min_image_quality = fusion_model.min_image_quality
            risk_fn = fusion_model.score_risk
        _pipeline = VerifierPipeline(
            DinoPairEncoder(
                model_name=os.getenv("SWAPSHIELD_VISION_MODEL", "facebook/dinov2-small"),
                device=os.getenv("SWAPSHIELD_DEVICE", "cuda"),
            ),
            QwenVisualVerifier(
                model_name=os.getenv("SWAPSHIELD_VLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct"),
                four_bit=os.getenv("SWAPSHIELD_VLM_4BIT", "true").lower() == "true",
                max_pixels=int(os.getenv("SWAPSHIELD_VLM_MAX_PIXELS", "262144")),
                oom_retry_pixels=int(os.getenv("SWAPSHIELD_VLM_OOM_RETRY_PIXELS", "131072")),
            ),
            threshold=threshold,
            min_image_quality=min_image_quality,
            risk_fn=risk_fn,
        )
    return _pipeline


def get_auditor() -> EvidenceAuditor:
    global _auditor
    if _auditor is None:
        _auditor = EvidenceAuditor.from_env()
    return _auditor


async def _decode_image(upload: UploadFile):
    if upload.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WebP evidence is accepted")
    content = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Evidence image exceeds the configured size limit")
    try:
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        image = Image.open(BytesIO(content))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Evidence file is not a valid image") from exc


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "SwapShield Local Verifier",
        "status": "ready",
        "documentation": "/docs",
        "health": "/health",
        "policy": "recommendation_only",
    }


@app.get("/health")
def health() -> dict[str, object]:
    pipeline = _pipeline
    fusion_path = os.getenv(
        "SWAPSHIELD_FUSION_MODEL", "evaluation/results/fusion-model.json"
    ).strip()
    auditor = get_auditor()
    return {
        "status": "ready",
        "policy": "recommendation_only",
        "models": {
            "dinov2": "loaded" if pipeline and getattr(pipeline.pair_encoder, "is_loaded", False) else "lazy",
            "qwen3_vl": "loaded" if pipeline and getattr(pipeline.visual_verifier, "is_loaded", False) else "lazy",
            "fusion": "trained" if fusion_path and Path(fusion_path).exists() else "baseline",
        },
        "auditor": {
            "mode": "llm_api" if auditor.configured else "deterministic_fallback",
            "model": auditor.model or None,
            "authority": "advisory_only",
        },
    }


@app.post("/v1/verify")
async def verify_return(
    dispatch_image: UploadFile = File(...),
    return_image: UploadFile = File(...),
    dispatch_sku: str = Form(..., min_length=1, max_length=80),
    return_sku: str = Form(..., min_length=1, max_length=80),
    dispatch_serial: str | None = Form(default=None, max_length=120),
    return_serial: str | None = Form(default=None, max_length=120),
    dispatch_weight_grams: float | None = Form(default=None, gt=0),
    return_weight_grams: float | None = Form(default=None, ge=0),
) -> dict[str, object]:
    dispatch = await _decode_image(dispatch_image)
    returned = await _decode_image(return_image)
    request = VerificationRequest(
        dispatch_image=dispatch,
        return_image=returned,
        dispatch_sku=dispatch_sku,
        return_sku=return_sku,
        dispatch_serial=dispatch_serial,
        return_serial=return_serial,
        dispatch_weight_grams=dispatch_weight_grams,
        return_weight_grams=return_weight_grams,
    )
    try:
        result = await run_in_threadpool(get_pipeline().verify, request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    payload = result.to_dict()
    audit = await run_in_threadpool(get_auditor().audit, payload)
    payload["auditor_assessment"] = audit.to_dict()
    return payload
