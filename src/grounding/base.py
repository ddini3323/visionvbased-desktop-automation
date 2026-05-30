from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass
class GroundingResult:
    x: int
    y: int
    confidence: float
    bbox: Tuple[int, int, int, int]
    method: str
    found: bool
    reasoning: str
    annotated_image: Optional[Image.Image] = field(default=None, repr=False)


class BaseGrounder(abc.ABC):
    @abc.abstractmethod
    def ground(self, screenshot: Image.Image, target: str) -> GroundingResult:
        """Locate `target` in `screenshot`, return screen-space coordinates."""

    def annotate(self, image: Image.Image, result: GroundingResult) -> Image.Image:
        """Return a copy of `image` annotated with the detection result."""
        out = image.copy().convert("RGB")
        draw = ImageDraw.Draw(out)

        if result.found:
            x1, y1, x2, y2 = result.bbox
            # Red bounding box
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

            # Crosshair at center
            cx, cy = result.x, result.y
            arm = 15
            draw.line([(cx - arm, cy), (cx + arm, cy)], fill=(255, 0, 0), width=2)
            draw.line([(cx, cy - arm), (cx, cy + arm)], fill=(255, 0, 0), width=2)

            # Label background + text
            label = f"{result.method} {result.confidence:.2f}"
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except Exception:
                font = ImageFont.load_default()

            text_bbox = draw.textbbox((x1, max(y1 - 22, 0)), label, font=font)
            draw.rectangle(text_bbox, fill=(255, 0, 0))
            draw.text((x1, max(y1 - 22, 0)), label, fill=(255, 255, 255), font=font)
        else:
            # Stamp "NOT FOUND" across the centre
            w, h = out.size
            draw.text(
                (w // 2 - 80, h // 2 - 15),
                "NOT FOUND",
                fill=(255, 0, 0),
            )

        return out
