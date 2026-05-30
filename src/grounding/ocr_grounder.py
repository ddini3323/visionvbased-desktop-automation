from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Optional

from PIL import Image

from .base import BaseGrounder, GroundingResult

logger = logging.getLogger(__name__)


def _make_not_found(reasoning: str) -> GroundingResult:
    return GroundingResult(x=0, y=0, confidence=0.0, bbox=(0, 0, 0, 0),
                           method="ocr", found=False, reasoning=reasoning)


class OCRGrounder(BaseGrounder):
    """Stage 2: find the icon text label via OCR, compute icon center above it."""

    def __init__(self, icon_label_offset_px: int = 40) -> None:
        self._offset = icon_label_offset_px
        self._reader = None
        self._load_error: Optional[str] = None
        self._loaded = False

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if self._load_error:
            return False
        try:
            import easyocr
            logger.info("OCRGrounder: initialising easyocr Reader (first use downloads models)...")
            self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            self._loaded = True
            logger.info("OCRGrounder: ready")
            return True
        except ImportError:
            self._load_error = "easyocr not installed"
            logger.warning("OCRGrounder: easyocr not installed -- stage skipped")
            return False
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("OCRGrounder: reader init failed: %s -- stage skipped", exc)
            return False

    def _matches(self, ocr_text: str, target: str) -> bool:
        short = target.strip().split()[0].lower()
        ocr_lower = ocr_text.strip().lower()
        if short in ocr_lower or ocr_lower in short:
            return True
        return SequenceMatcher(None, short, ocr_lower).ratio() > 0.75

    def ground(self, screenshot: Image.Image, target: str) -> GroundingResult:
        if not self._ensure_loaded():
            return _make_not_found(self._load_error or "reader unavailable")

        import numpy as np
        img_arr = np.array(screenshot.convert("RGB"))

        try:
            ocr_results = self._reader.readtext(img_arr)
        except Exception as exc:
            logger.warning("OCRGrounder.readtext failed: %s", exc)
            return _make_not_found(str(exc))

        best_conf = 0.0
        best_hit = None
        for (bbox_pts, text, conf) in ocr_results:
            if conf < 0.4:
                continue
            if not self._matches(text, target):
                continue
            if conf > best_conf:
                best_conf = conf
                best_hit = (bbox_pts, text, conf)

        if best_hit is None:
            return _make_not_found("Target text not found by OCR")

        bbox_pts, text, conf = best_hit
        xs = [pt[0] for pt in bbox_pts]
        ys = [pt[1] for pt in bbox_pts]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        label_cx = (x1 + x2) // 2
        label_cy = (y1 + y2) // 2

        icon_cx = label_cx
        icon_cy = label_cy - self._offset
        half = 24
        W, H = screenshot.size
        icon_bbox = (
            max(0, icon_cx - half), max(0, icon_cy - half),
            min(W, icon_cx + half), min(H, icon_cy + half),
        )

        return GroundingResult(
            x=icon_cx, y=icon_cy,
            confidence=conf,
            bbox=icon_bbox,
            method="ocr",
            found=True,
            reasoning=f"OCR: '{text}' (conf={conf:.2f}), icon ~{self._offset}px above label",
        )
