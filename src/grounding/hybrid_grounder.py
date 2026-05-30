from __future__ import annotations

import logging
import re
import time
from typing import List, Optional

from PIL import Image

from .base import BaseGrounder, GroundingResult

logger = logging.getLogger(__name__)


class HybridGrounder(BaseGrounder):
    """Cascaded grounding: DINO -> OCR -> Template -> VLM (ScreenSeekeR).

    Each stage is tried in order; the first success short-circuits the rest.
    After a successful detection the icon crop is saved as a template so Stage 3
    gets faster on subsequent runs.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        base_url: Optional[str] = None,
        max_retries: int = 3,
        template_dir: str = "templates",
    ) -> None:
        from .dino_grounder import DINOGrounder
        from .ocr_grounder import OCRGrounder
        from .template_grounder import TemplateGrounder
        from .vlm_grounder import VLMGrounder

        self._template_grounder = TemplateGrounder(template_dir=template_dir)
        self._stages: List[BaseGrounder] = [
            DINOGrounder(),
            OCRGrounder(),
            self._template_grounder,
            VLMGrounder(api_key=api_key, model=model, base_url=base_url, max_retries=max_retries),
        ]
        self._stage_names = ["dino", "ocr", "template", "vlm"]
        self._max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ground(self, screenshot: Image.Image, target: str) -> GroundingResult:
        dino_target = self._simplify_target(target)
        timings: List[str] = []

        for attempt in range(1, self._max_retries + 1):
            logger.info("HybridGrounder: attempt %d/%d", attempt, self._max_retries)

            for idx, stage in enumerate(self._stages):
                name = self._stage_names[idx]
                effective = dino_target if idx == 0 else target
                t0 = time.perf_counter()

                try:
                    result = stage.ground(screenshot, effective)
                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    logger.warning("Stage [%s] raised: %s", name, exc)
                    timings.append(f"{name}=ERR({elapsed:.2f}s)")
                    continue

                elapsed = time.perf_counter() - t0
                timings.append(f"{name}={elapsed:.2f}s")

                if result.found:
                    logger.info(
                        "HybridGrounder: SUCCESS via [%s] in %.2fs (attempt %d/%d)",
                        name, elapsed, attempt, self._max_retries,
                    )
                    # Auto-save template for Stage 3 on future calls
                    if idx != 2:  # skip if we ARE the template stage
                        self._save_template(screenshot, result, target)

                    timing_str = ", ".join(timings)
                    return GroundingResult(
                        x=result.x, y=result.y,
                        confidence=result.confidence,
                        bbox=result.bbox,
                        method=result.method,
                        found=True,
                        reasoning=f"{result.reasoning} | {timing_str}",
                    )

            logger.warning(
                "HybridGrounder: all stages failed on attempt %d. %s",
                attempt, ", ".join(timings),
            )

        timing_str = ", ".join(timings)
        return GroundingResult(
            x=0, y=0, confidence=0.0, bbox=(0, 0, 0, 0),
            method="hybrid", found=False,
            reasoning=f"All stages failed after {self._max_retries} attempts. {timing_str}",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _simplify_target(self, target: str) -> str:
        """Shorten verbose target string to a compact DINO-friendly prompt."""
        short = re.split(r"\s+(?:on|in|at|near|from)\s+", target, maxsplit=1)[0]
        words = short.strip().split()
        prompt = " ".join(words[:3])
        return prompt if prompt.endswith(".") else prompt + "."

    def _save_template(
        self, screenshot: Image.Image, result: GroundingResult, target: str
    ) -> None:
        try:
            self._template_grounder.save_template(screenshot, result.bbox, target)
        except Exception as exc:
            logger.warning("HybridGrounder: template save failed: %s", exc)
