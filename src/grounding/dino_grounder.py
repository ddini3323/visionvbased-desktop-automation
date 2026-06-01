from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

from PIL import Image

from .base import BaseGrounder, GroundingResult

logger = logging.getLogger(__name__)


def _make_not_found(reasoning: str) -> GroundingResult:
    return GroundingResult(x=0, y=0, confidence=0.0, bbox=(0, 0, 0, 0),
                           method="dino", found=False, reasoning=reasoning)


class DINOGrounder(BaseGrounder):
    """Stage 1: zero-shot object detection via GroundingDINO (HuggingFace transformers)."""

    MODEL_ID = "IDEA-Research/grounding-dino-tiny"
    SCORE_THRESHOLD = 0.3

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._load_error: Optional[str] = None
        self._loaded = False

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if self._load_error:
            return False
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            logger.info("DINOGrounder: loading model %s (first use may download ~700MB)...", self.MODEL_ID)
            self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.MODEL_ID)
            self._model.eval()
            self._loaded = True
            logger.info("DINOGrounder: model loaded")
            return True
        except ImportError:
            self._load_error = "transformers not installed"
            logger.warning("DINOGrounder: transformers not installed -- stage skipped")
            return False
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("DINOGrounder: model load failed: %s -- stage skipped", exc)
            return False

    def ground(self, screenshot: Image.Image, target: str) -> GroundingResult:
        if not self._ensure_loaded():
            return _make_not_found(self._load_error or "model unavailable")

        # DINO works best with short, period-terminated prompts
        text_prompt = self._build_prompt(target)
        image = screenshot.convert("RGB")
        W, H = image.size

        try:
            import torch
            inputs = self._processor(images=image, text=text_prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model(**inputs)

            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.SCORE_THRESHOLD,
                text_threshold=self.SCORE_THRESHOLD,
                target_sizes=[(H, W)],
            )
            det = results[0]
            if len(det["scores"]) == 0:
                return _make_not_found(f"No detections above threshold {self.SCORE_THRESHOLD}")

            best_idx = int(det["scores"].argmax())
            score = float(det["scores"][best_idx])
            box = det["boxes"][best_idx].tolist()
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            x1, x2 = max(0, x1), min(W, x2)
            y1, y2 = max(0, y1), min(H, y2)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            label = det["labels"][best_idx] if "labels" in det else "?"

            return GroundingResult(
                x=cx, y=cy,
                confidence=score,
                bbox=(x1, y1, x2, y2),
                method="dino",
                found=True,
                reasoning=f"DINO: '{label}' score={score:.3f} prompt='{text_prompt}'",
            )

        except Exception as exc:
            logger.warning("DINOGrounder.ground failed: %s", exc)
            return _make_not_found(str(exc))

    def _build_prompt(self, target: str) -> str:
        """Shorten verbose target to a compact DINO-friendly prompt."""
        # Strip everything from a preposition onwards
        short = re.split(r'\s+(?:on|in|at|near|from)\s+', target, maxsplit=1)[0]
        words = short.strip().split()
        prompt = " ".join(words[:3])
        return prompt if prompt.endswith(".") else prompt + "."
