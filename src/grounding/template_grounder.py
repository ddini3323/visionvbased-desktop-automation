from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .base import BaseGrounder, GroundingResult

logger = logging.getLogger(__name__)

_SCALES: List[float] = [0.8, 1.0, 1.2]
_THRESHOLD = 0.65


def _make_not_found(reasoning: str) -> GroundingResult:
    return GroundingResult(x=0, y=0, confidence=0.0, bbox=(0, 0, 0, 0),
                           method="template", found=False, reasoning=reasoning)


def _slug(target: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", target.lower()).strip("_")


class TemplateGrounder(BaseGrounder):
    """Stage 3: cv2 multi-scale template matching against a stored icon crop."""

    def __init__(self, template_dir: str = "templates") -> None:
        self._tdir = Path(template_dir)

    def _find_template(self, target: str) -> Optional[np.ndarray]:
        short = target.strip().split()[0].lower()
        candidates = [
            self._tdir / f"{_slug(target)}.png",
            self._tdir / f"{short}.png",
            self._tdir / "notepad_icon.png",
            Path("icon_crop.png"),  # captured earlier, exists in project root
        ]
        for path in candidates:
            if path.exists():
                tmpl = cv2.imread(str(path))
                if tmpl is not None:
                    logger.debug("TemplateGrounder: using template %s", path)
                    return tmpl
        return None

    def ground(self, screenshot: Image.Image, target: str) -> GroundingResult:
        tmpl_bgr = self._find_template(target)
        if tmpl_bgr is None:
            return _make_not_found("No template file found")

        img_rgb = np.array(screenshot.convert("RGB"))
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        tmpl_gray = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY)

        H_img, W_img = img_gray.shape
        best_score = 0.0
        best_loc: Optional[Tuple[int, int]] = None
        best_tw = tmpl_gray.shape[1]
        best_th = tmpl_gray.shape[0]

        for scale in _SCALES:
            tw = max(1, int(tmpl_gray.shape[1] * scale))
            th = max(1, int(tmpl_gray.shape[0] * scale))
            if tw > W_img or th > H_img:
                continue
            scaled = cv2.resize(tmpl_gray, (tw, th), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(img_gray, scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best_loc = max_loc
                best_tw, best_th = tw, th

        if best_score < _THRESHOLD or best_loc is None:
            return _make_not_found(f"Best match {best_score:.3f} < threshold {_THRESHOLD}")

        x1 = best_loc[0]
        y1 = best_loc[1]
        x2 = x1 + best_tw
        y2 = y1 + best_th
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        return GroundingResult(
            x=cx, y=cy,
            confidence=best_score,
            bbox=(x1, y1, x2, y2),
            method="template",
            found=True,
            reasoning=f"Template match score={best_score:.3f}",
        )

    def save_template(self, screenshot: Image.Image, bbox: Tuple[int, int, int, int],
                      target: str) -> Optional[Path]:
        """Crop bbox from screenshot and save for future use."""
        x1, y1, x2, y2 = bbox
        pad = 4
        W, H = screenshot.size
        x1c = max(0, x1 - pad)
        y1c = max(0, y1 - pad)
        x2c = min(W, x2 + pad)
        y2c = min(H, y2 + pad)
        if x2c <= x1c or y2c <= y1c:
            return None
        img_rgb = np.array(screenshot.convert("RGB"))
        crop = img_rgb[y1c:y2c, x1c:x2c]
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        self._tdir.mkdir(parents=True, exist_ok=True)
        out_path = self._tdir / f"{_slug(target)}.png"
        cv2.imwrite(str(out_path), crop_bgr)
        logger.info("TemplateGrounder: saved template -> %s", out_path)
        return out_path
