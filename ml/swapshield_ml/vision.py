"""DINOv2 image-pair similarity with lazy model loading."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class DinoPairEncoder:
    def __init__(self, model_name: str = "facebook/dinov2-small", device: str = "cuda") -> None:
        self.model_name = model_name
        self.device = device
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.is_loaded:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover - exercised only with optional GPU dependencies
            raise RuntimeError("Install the gpu extra before loading DINOv2: pip install -e 'ml[gpu]'") from exc

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("SWAPSHIELD_DEVICE requests CUDA, but PyTorch cannot see a CUDA GPU")

        self._torch = torch
        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        dtype = torch.bfloat16 if self.device.startswith("cuda") and torch.cuda.is_bf16_supported() else None
        self._model = AutoModel.from_pretrained(self.model_name, torch_dtype=dtype)
        self._model.to(self.device)
        self._model.eval()

    def _embed(self, images: Sequence[Any]) -> Any:
        if not images:
            raise ValueError("at least one image is required")
        self.load()
        torch = self._torch
        assert torch is not None and self._processor is not None and self._model is not None
        inputs = self._processor(images=list(images), return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            output = self._model(**inputs)
            cls_embeddings = output.last_hidden_state[:, 0, :].float()
            return torch.nn.functional.normalize(cls_embeddings, dim=-1)

    def compare(self, dispatch_images: Sequence[Any], return_images: Sequence[Any]) -> float:
        """Return centroid cosine similarity, clamped to the risk model's 0-1 range."""
        dispatch = self._embed(dispatch_images)
        returned = self._embed(return_images)
        torch = self._torch
        assert torch is not None
        dispatch_centroid = torch.nn.functional.normalize(dispatch.mean(dim=0), dim=0)
        return_centroid = torch.nn.functional.normalize(returned.mean(dim=0), dim=0)
        similarity = torch.dot(dispatch_centroid, return_centroid).item()
        return round(max(0.0, min(1.0, float(similarity))), 6)
